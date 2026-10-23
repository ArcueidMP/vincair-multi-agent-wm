"""Frozen local-model composition on explicitly declared physics profiles.

The guarded finite-state runner below is adapted from abstraction_composition;
its historical source and results remain unchanged. Phase budgets, feedback
gains, handoff forces, tolerances, action libraries and goal pairs are fixed
before heldout evaluation. Models are trained only on local response programs.

Ordered-goal success is reported independently of full interface completion:
passing an active two-contact force gate is not required to demonstrate the
external manipulation goals. Both metrics retain every failed early stage.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.abstraction_composition import (
    GOAL_DWELL, RETREAT_DWELL, RECOVERY_DWELL, PHASES, VARIANTS, checkpoint_digest,
    _timer_readiness,
)
from vmas_pusht.abstraction_models import predict_checkpoint
from vmas_pusht.abstraction_transfer import (
    ANGLE_TOLERANCE, MACRO_STEPS, PRELOAD, SUPPORTS, TRANSLATION_TOLERANCE,
    _support_commands, effective_anchor, feedback_primary, load_checkpoints,
    main_candidates, reposition_action, task_metrics,
)
from vmas_pusht.boundary import _at_wall, _finite, _plain, _rotate, surface_gaps
from vmas_pusht.physical_abstraction import (
    HELPER_LOCAL, PRIMARY_LOCAL, coarse_state, model_features, norm_bound,
    physical_state, setup_scene,
)
from vmas_pusht.physics_profiles import get_profile, make_profile
from vmas_pusht.scenario import wrap
from vmas_pusht.snapshot import EntitySnapshot, SimulationSnapshot, capture, restore

MODEL_NAMES = ("coarse_small", "coarse_body_small", "full_matched", "history_matched")
PLANNERS = ("legacy_wm",) + tuple("new_" + name for name in MODEL_NAMES) + ("analytic_feedback", "zero_push")
CANDIDATE_BANKS = ("bank10", "bank13")


def is_model_planner(planner):
    return planner == "legacy_wm" or planner.startswith("new_")


def validate_checkpoints(checkpoints, supports, planners, seed):
    for planner in planners:
        if planner not in PLANNERS:
            raise ValueError(f"Unknown planner {planner}")
        for support in supports:
            if support not in SUPPORTS:
                raise ValueError(f"Unknown support {support}")
            if planner == "legacy_wm" and support not in checkpoints.get("legacy", {}):
                raise ValueError(f"Missing legacy model for {support}")
            if planner.startswith("new_"):
                key = f"{support}/{planner[4:]}/seed{seed}"
                if key not in checkpoints.get("new", {}):
                    raise ValueError(f"Missing retrained model {key}")


def candidates_for_bank(angles, bank="bank10"):
    if bank == "bank10":
        return main_candidates(angles)
    if bank != "bank13":
        raise ValueError(f"Unknown candidate bank {bank}")
    programs = [(0., 0.)] + list(itertools.product((-20., 0., 20.), (.15, .3, .6, .9)))
    direction = angles[:, None] + angles.new_tensor([math.radians(d) for d, _ in programs])[None]
    strengths = angles.new_tensor([a for _, a in programs])
    return torch.stack((direction.sin(), -direction.cos()), -1) * strengths[None, :, None]


def empty_history(geometry):
    history = geometry.new_zeros(len(geometry), 2, 17)
    history[:, :, :12] = geometry[:, None, :12]
    return history


def history_inputs(geometry_trace, action_trace):
    """Two complete past five-step intervals, oldest first; no future states."""
    now = len(action_trace)
    if len(geometry_trace) != now + 1:
        raise ValueError("Geometry trace must include the initial and each post-step state")
    history = empty_history(geometry_trace[-1])
    for slot, end in enumerate((now - MACRO_STEPS, now)):
        start = end - MACRO_STEPS
        if start >= 0:
            history[:, slot, :12] = geometry_trace[start]
            history[:, slot, 12:16] = torch.stack(action_trace[start:end]).mean(0).flatten(1)
            history[:, slot, 16] = 1.
    return history


def _entry(checkpoints, row, seed):
    if row["planner"] == "legacy_wm":
        return checkpoints["legacy"][row["support"]]
    return checkpoints["new"][f"{row['support']}/{row['planner'][4:]}/seed{seed}"]


def predict_rows(checkpoints, rows, coarse, full, history, mask, model_seed=0, legacy_substeps=20):
    """coarse/full/history are B,K,..., preserve all physical and action inputs."""
    n, k = coarse.shape[:2]
    prediction = coarse.new_zeros(n, k, 3)
    for support, planner in sorted({(r["support"], r["planner"]) for r in rows if is_model_planner(r["planner"])}):
        indices = torch.tensor([i for i, row in enumerate(rows)
            if row["support"] == support and row["planner"] == planner and bool(mask[i])], dtype=torch.long)
        if not len(indices):
            continue
        entry = _entry(checkpoints, rows[int(indices[0])], model_seed)
        c = coarse[indices].reshape(-1, 18).clone()
        if planner == "legacy_wm":
            c[:, 15] = float(legacy_substeps)
            values = predict_checkpoint(entry, c)
        else:
            from vmas_pusht.stable_local_wm import predict_entry
            values = predict_entry(entry, c, full=full[indices].reshape(-1, 32),
                                   history=history[indices].reshape(-1, 2, 17))
        prediction[indices] = values.reshape(-1, k, 3)
    return prediction


def query_frozen_model(env, anchor, relative_goal, initial_pose, rows, checkpoints,
                       query_mask, model_substeps=20, history=None, candidate_bank="bank10", model_seed=0):
    n = env.num_envs
    candidates = candidates_for_bank(env.scenario.bar.state.rot[:, 0], candidate_bank)
    count = candidates.shape[1]
    coarse, full = model_features(env, torch.zeros(n, 2), MACRO_STEPS, anchor)
    history = empty_history(coarse) if history is None else history
    inputs = coarse[:, None].expand(n, count, -1).clone()
    full_inputs = full[:, None].expand(n, count, -1).clone()
    inputs[:, :, 12:14] = candidates
    full_inputs[:, :, 12:14] = candidates
    history_inputs_ = history[:, None].expand(n, count, 2, 17)
    needed = query_mask & torch.tensor([is_model_planner(r["planner"]) for r in rows])
    for i, row in enumerate(rows):
        if bool(needed[i]):
            reference = (_entry(checkpoints, row, model_seed)["reference_substeps"]
                         if row["planner"].startswith("new_") else (20 if model_substeps is None else model_substeps))
            inputs[i, :, 15] = float(reference)
            full_inputs[i, :, 15] = float(reference)
    prediction = predict_rows(checkpoints, rows, inputs, full_inputs, history_inputs_, needed,
                              model_seed, 20 if model_substeps is None else model_substeps)
    predicted_pose = env.scenario.t_pose()[:, None] + prediction
    target = initial_pose[:, 2] + relative_goal
    error = wrap(predicted_pose[:, :, 2] - target[:, None])
    shift = (predicted_pose[:, :, :2] - initial_pose[:, None, :2]).norm(dim=-1)
    cost = (error / ANGLE_TOLERANCE).square() + (shift / TRANSLATION_TOLERANCE).square()
    chosen = cost.argmin(1)
    action = candidates[torch.arange(n), chosen]
    expected = prediction[torch.arange(n), chosen]
    chosen[~needed] = -1
    return action, chosen, expected, inputs, needed


def snapshot_record(env, indices, anchor, history, rows, stage):
    """Serialize selected true simulator states, including independently moving stem."""
    snapshot = asdict(capture(env))
    idx = torch.tensor(indices, dtype=torch.long)
    def select(value):
        return None if value is None else value.index_select(0, idx).clone()
    for entity in snapshot["entities"]:
        entity["state"] = {k: select(v) for k, v in entity["state"].items()}
        if entity["action"] is not None:
            entity["action"] = {k: select(v) for k, v in entity["action"].items()}
    snapshot["scenario"] = {k: select(v) for k, v in snapshot["scenario"].items()}
    snapshot["steps"] = select(snapshot["steps"])
    snapshot["batch_size"] = len(indices)
    return dict(snapshot=snapshot, anchor=anchor[idx].clone(), history=history[idx].clone(),
                rows=[dict(rows[i], snapshot_stage=stage, composition_row=i) for i in indices])


def snapshot_from_record(record):
    saved = dict(record["snapshot"])
    saved["entities"] = tuple(EntitySnapshot(**entry) for entry in saved["entities"])
    return SimulationSnapshot(**saved)


@torch.no_grad()
def run_group(*, profile="original", substeps=20, variant="guarded_all", checkpoints=None,
              supports=SUPPORTS, planners=PLANNERS, angles_deg=(0., 30.),
              goal_pair=(.04, .08), model_seed=0, candidate_bank="bank13", capture_counterfactual=True,
              model_training_profile=None, prelude_steps=10,
              nuisances=((0., 0., 0.), (.045, 0., .3), (-.045, 0., -.3)), rotation_steps=60,
              retreat_steps=20, max_recovery=80, model_substeps=20,
              goal_dwell=GOAL_DWELL, retreat_dwell=RETREAT_DWELL,
              recovery_dwell=RECOVERY_DWELL):
    from vmas_pusht.rebrace import RebraceController, helper_contact_forces

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if rotation_steps < MACRO_STEPS or rotation_steps % MACRO_STEPS:
        raise ValueError("rotation_steps must be a positive multiple of five")
    if min(retreat_steps, max_recovery, goal_dwell, retreat_dwell, recovery_dwell) < 1:
        raise ValueError("phase budgets and dwell requirements must be positive")
    if max_recovery < recovery_dwell + 1:
        raise ValueError("recovery budget must allow approach and handoff verification")
    if model_substeps is not None and model_substeps < 1:
        raise ValueError("model_substeps must be positive or None")
    if variant != "guarded_all":
        raise ValueError("The stable-physics audit fixes the guarded_all controller")
    profile = get_profile(profile)
    if len(goal_pair) != 2 or not 0 < goal_pair[0] < goal_pair[1]:
        raise ValueError("goal_pair must contain two ordered positive angles")
    if candidate_bank not in CANDIDATE_BANKS:
        raise ValueError("Unknown action bank")
    checkpoints = checkpoints or {}
    validate_checkpoints(checkpoints, supports, planners, model_seed)
    n_candidates = 10 if candidate_bank == "bank10" else 13
    if prelude_steps < 2 * MACRO_STEPS:
        raise ValueError("The common physical prelude must supply two real history windows")
    rows = [dict(support=support, planner=planner, initial_angle_deg=angle,
                 relative_goal_rad=goal_pair[1], first_goal_rad=goal_pair[0], goal_pair=list(goal_pair),
                 physics_profile=profile["name"], model_training_profile=model_training_profile,
                 model_seed=model_seed, candidate_bank=candidate_bank, nuisance_id=nid, initial_velocity=list(nuisance[:2]),
                 initial_angular_velocity=nuisance[2], friction_ratio=profile["scenario"]["friction_ratio"], substeps=substeps,
                 variant=variant, task="rotate_retreat_rebrace_rotate",
                 model_substeps_input=substeps if model_substeps is None else model_substeps,
                 model_conditioning="native_input" if model_substeps is None else "fixed_reference",
                 model_substeps_input_ood=None)
            for support, planner, angle, (nid, nuisance) in
            itertools.product(supports, planners, angles_deg, enumerate(nuisances))]
    for row in rows:
        if row["planner"].startswith("new_"):
            row["model_substeps_input"] = int(_entry(checkpoints, row, model_seed)["reference_substeps"])
            row["model_conditioning"] = "fixed_training_reference"
        elif row["planner"] == "legacy_wm":
            row["model_substeps_input"] = 20 if model_substeps is None else model_substeps
            row["model_conditioning"] = "fixed_legacy_reference"
        else:
            row["model_substeps_input"] = None
            row["model_conditioning"] = "not_applicable"
    n = len(rows)
    if not n:
        raise ValueError("empty composition experiment")
    before_hash = checkpoint_digest(checkpoints)
    env = make_profile(profile, num_envs=n, device="cpu", substeps=substeps, seed=101)
    angles = torch.tensor([math.radians(r["initial_angle_deg"]) for r in rows])
    anchor = setup_scene(env, angle=angles, hidden_v=[r["initial_velocity"] for r in rows],
                         hidden_omega=[r["initial_angular_velocity"] for r in rows])
    support_names = [r["support"] for r in rows]
    preload = torch.tensor([s == "preloaded_anchor" for s in support_names])
    anchor += preload[:, None] * _rotate(torch.tensor([-PRELOAD, PRELOAD]).expand(n, 2), angles)
    prelude_initial_pose = env.scenario.t_pose().clone()
    prelude_trajectory = [physical_state(env).clone()]
    prelude_geometry = [coarse_state(env).clone()]
    prelude_actions = []
    prelude_force, prelude_effort, prelude_work, prelude_travel, prelude_contacts = [torch.zeros(n, 2) for _ in range(5)]
    prelude_norm = torch.zeros(n, 2)
    prelude_finite, prelude_wall = _finite(env), _at_wall(env)
    for _ in range(prelude_steps):
        command = torch.stack((torch.zeros(n, 2), _support_commands(env, anchor, rows)), 1)
        old_pos = torch.stack([a.state.pos.clone() for a in env.world.agents], 1)
        old_vel = torch.stack([a.state.vel.clone() for a in env.world.agents], 1)
        env.step([command[:, 0], command[:, 1]])
        new_pos = torch.stack([a.state.pos for a in env.world.agents], 1)
        new_vel = torch.stack([a.state.vel for a in env.world.agents], 1)
        norms = command.norm(dim=-1)
        prelude_norm = torch.maximum(prelude_norm, norms)
        prelude_force += norms * env.scenario.agent_force * env.world.dt
        prelude_effort += norms.square() * env.world.dt
        prelude_work += (command * env.scenario.agent_force * (old_vel + new_vel) / 2).sum(-1).abs() * env.world.dt
        prelude_travel += (new_pos - old_pos).norm(dim=-1)
        prelude_contacts += (surface_gaps(env).amin(-1) <= .005)
        prelude_finite &= _finite(env)
        prelude_wall |= _at_wall(env)
        prelude_actions.append(command.clone())
        prelude_trajectory.append(physical_state(env).clone())
        prelude_geometry.append(coarse_state(env).clone())
    # Goal progress starts after preparation, so helper-only motion cannot be
    # credited toward either requested angle. Preparation resources still count.
    initial_pose = env.scenario.t_pose().clone()
    controller = RebraceController(env, support_names, max_steps=max_recovery,
                                  confirm_steps=recovery_dwell)
    is_wm = torch.tensor([is_model_planner(r["planner"]) for r in rows])
    is_zero = torch.tensor([r["planner"] == "zero_push" for r in rows])
    horizon = 2 * rotation_steps + retreat_steps + max_recovery
    phase = torch.zeros(n, dtype=torch.long)
    phase_age = torch.zeros(n, dtype=torch.long)
    phase_steps = torch.zeros(n, 4, dtype=torch.long)
    phase_end = torch.full((n, 4), -1, dtype=torch.long)
    gates = torch.zeros(n, 4, dtype=torch.bool)
    timed_out = torch.zeros(n, 4, dtype=torch.bool)
    dwell = torch.zeros(n, dtype=torch.long)
    timer_recovery_dwell = torch.zeros(n, dtype=torch.long)
    maximum_recovery_dwell = torch.zeros(n, dtype=torch.long)
    anchor_at_handoff = torch.full((n, 2), float("nan"))
    recovery_ready_ever = torch.zeros(n, dtype=torch.bool)
    ever_first = torch.zeros(n, dtype=torch.bool)
    ever_final = torch.zeros(n, dtype=torch.bool)
    first_final = torch.full((n,), -1, dtype=torch.long)
    terminal_pose = initial_pose.clone()
    terminal_state = physical_state(env).clone()
    actual_steps = torch.zeros(n, dtype=torch.long)
    main = torch.zeros(n, 2)
    retreat_target = torch.zeros(n, 2)
    finite, wall = prelude_finite.clone(), prelude_wall.clone()
    force_integral, effort, work, travel, contacts = [v.clone() for v in
        (prelude_force, prelude_effort, prelude_work, prelude_travel, prelude_contacts)]
    helper_force_contacts = torch.zeros(n, 4)
    norm_max = prelude_norm.clone()
    macro_start = initial_pose.clone()
    macro_prediction = torch.zeros(n, 3)
    macro_remaining = torch.zeros(n, dtype=torch.long)
    model_queries = torch.zeros(n, dtype=torch.long)
    traces = dict(trajectory=[terminal_state.clone()], helper_anchor=[anchor.clone()],
                  actions=[], phases=[], active=[], surface_gaps=[], goal_dwell=[],
                  helper_normal_forces=[], recovery_dwell=[], recovery_ready=[], recovery_metrics=[])
    selected_macros, predictions, model_inputs = [], [], []
    prediction_errors, prediction_valid = [], []
    geometry_trace = prelude_geometry.copy()
    history_action_trace = prelude_actions.copy()
    counterfactual_snapshots = []
    if capture_counterfactual:
        counterfactual_snapshots.append(snapshot_record(env, list(range(n)), anchor,
            history_inputs(geometry_trace, history_action_trace), rows, "post_prelude"))

    for step in range(horizon):
        active = phase < 4
        if not active.any():
            break
        old_phase = phase.clone()
        rotating = active & ((phase == 0) | (phase == 3))
        recovering = active & (phase == 2)
        retreating = active & (phase == 1)
        goal = torch.where(phase == 0, goal_pair[0], goal_pair[1])
        renew = rotating & (phase_age % MACRO_STEPS == 0)
        if renew.any():
            analytic = feedback_primary(env, goal, initial_pose)
            main[renew] = analytic[renew]
            chosen = torch.full((n,), -1, dtype=torch.long)
            expected = torch.zeros(n, 3)
            if (renew & is_wm).any():
                proposal, chosen, expected, inputs, queried = query_frozen_model(
                    env, anchor, goal, initial_pose, rows, checkpoints, renew, model_substeps,
                    history_inputs(geometry_trace, history_action_trace), candidate_bank, model_seed)
                main[renew & is_wm] = proposal[renew & is_wm]
                model_queries += queried
                model_inputs.append(dict(step=step, queried_rows=queried.clone(),
                                         inputs=inputs[queried].clone()))
            main[renew & is_zero] = 0
            macro_start[renew] = env.scenario.t_pose()[renew]
            macro_prediction[renew] = expected[renew]
            macro_remaining[renew] = MACRO_STEPS
            selected_macros.append(dict(step=step, rows=renew.clone(), chosen_index=chosen.clone()))
            predictions.append(dict(step=step, predicted_response=expected.clone(),
                                    wm_rows=(renew & is_wm).clone()))
        command = torch.zeros(n, 2, 2)
        command[rotating, 0] = main[rotating]
        command[rotating, 1] = _support_commands(env, anchor, rows)[rotating]
        if retreating.any():
            agent = env.world.agents[1]
            retreat_command = norm_bound(30. * (retreat_target - agent.state.pos) - 4. * agent.state.vel)
            command[retreating, 1] = retreat_command[retreating]
        timer_verify = recovering & (phase_age >= max_recovery - recovery_dwell)
        if recovering.any():
            if variant == "timer_baseline":
                start_verify = recovering & (phase_age == max_recovery - recovery_dwell)
                reference = effective_anchor(env, support_names)
                anchor[start_verify] = reference[start_verify]
                command[recovering, 0] = reposition_action(env, 0, PRIMARY_LOCAL)[recovering]
                command[recovering, 1] = reposition_action(env, 1, HELPER_LOCAL)[recovering]
                command[timer_verify, 1] = _support_commands(env, anchor, rows)[timer_verify]
            else:
                proposed = controller.step()
                command[recovering] = proposed[recovering]
        old_pos = torch.stack([a.state.pos.clone() for a in env.world.agents], 1)
        old_vel = torch.stack([a.state.vel.clone() for a in env.world.agents], 1)
        env.step([command[:, 0], command[:, 1]])
        new_pos = torch.stack([a.state.pos for a in env.world.agents], 1)
        new_vel = torch.stack([a.state.vel for a in env.world.agents], 1)
        norms = command.norm(dim=-1)
        norm_max = torch.maximum(norm_max, norms)
        force_integral += norms * env.scenario.agent_force * env.world.dt * active[:, None]
        effort += norms.square() * env.world.dt * active[:, None]
        work += (command * env.scenario.agent_force * (old_vel + new_vel) / 2).sum(-1).abs() * env.world.dt * active[:, None]
        travel += (new_pos - old_pos).norm(dim=-1) * active[:, None]
        gaps = surface_gaps(env)
        helper_forces = helper_contact_forces(env).norm(dim=-1)
        dual_force_contact = (helper_forces >= controller.config.minimum_contact_force).all(-1)
        contacts += (gaps.amin(-1) <= .005) * active[:, None]
        helper_force_contacts[active, old_phase[active]] += dual_force_contact[active]
        finite &= ~active | _finite(env)
        wall |= active & _at_wall(env)
        actual_steps += active
        phase_age += active
        phase_steps[active, old_phase[active]] += 1
        pose = env.scenario.t_pose()
        first_ok, _, _ = task_metrics(pose, initial_pose, torch.full((n,), goal_pair[0]))
        final_ok, _, _ = task_metrics(pose, initial_pose, torch.full((n,), goal_pair[1]))
        ever_first |= (old_phase == 0) & first_ok
        newly = active & final_ok & ~ever_final
        first_final[newly] = step + 1
        ever_final |= active & final_ok
        goal_ok = torch.where(old_phase == 0, first_ok, final_ok)
        retreat_ok = gaps[:, 1].amin(-1) > .01
        pass_now = torch.where(rotating, goal_ok, retreat_ok)
        dwell = torch.where(active & ~recovering & pass_now, dwell + 1,
                            torch.zeros_like(dwell))
        recovery_ready = torch.zeros(n, dtype=torch.bool)
        recovery_metrics = {}
        if recovering.any():
            if variant == "timer_baseline":
                recovery_metrics = _timer_readiness(env, anchor, support_names)
                candidate = recovery_metrics["instantaneous_ready"] & timer_verify
                timer_recovery_dwell = torch.where(candidate, timer_recovery_dwell + 1,
                                                   torch.zeros_like(timer_recovery_dwell))
                recovery_ready = timer_recovery_dwell >= recovery_dwell
                observed_recovery_dwell = timer_recovery_dwell
            else:
                recovery_metrics = controller.update(active_mask=recovering)
                recovery_ready = recovery_metrics["ready"] & recovering
                observed_recovery_dwell = recovery_metrics["dwell"]
                anchor[recovering] = recovery_metrics["anchor"][recovering]
            recovery_ready_ever |= recovery_ready
            maximum_recovery_dwell = torch.maximum(maximum_recovery_dwell,
                torch.where(recovering, observed_recovery_dwell, torch.zeros_like(maximum_recovery_dwell)))
        else:
            observed_recovery_dwell = torch.zeros(n, dtype=torch.long)
        limits = torch.tensor([rotation_steps, retreat_steps, max_recovery, rotation_steps])
        age_limit = torch.zeros(n, dtype=torch.long)
        age_limit[active] = limits[old_phase[active]]
        reached_budget = active & (phase_age >= age_limit)
        gate = torch.where(recovering, recovery_ready,
                           dwell >= torch.where(retreating, retreat_dwell, goal_dwell))
        if variant == "guarded_all":
            finish = active & (gate | reached_budget)
        elif variant == "guarded_recovery":
            finish = active & (reached_budget | (recovering & gate))
        else:
            finish = reached_budget
        gates[finish, old_phase[finish]] = gate[finish]
        timed_out[finish, old_phase[finish]] = ~gate[finish]
        phase_end[finish, old_phase[finish]] = step + 1
        # Evaluate complete macro predictions before switching phase; interrupted
        # macros are excluded explicitly rather than scored as full duration.
        macro_remaining[rotating] -= 1
        completed_macro = rotating & (macro_remaining == 0) & is_wm
        if completed_macro.any():
            response = pose - macro_start
            response[:, 2] = wrap(response[:, 2])
            error = macro_prediction - response
            prediction_errors.append(error.clone())
            prediction_valid.append(completed_macro.clone())
        traces["actions"].append(command.clone())
        history_action_trace.append(command.clone())
        geometry_trace.append(coarse_state(env).clone())
        traces["phases"].append(old_phase.clone())
        traces["active"].append(active.clone())
        traces["surface_gaps"].append(gaps.clone())
        traces["helper_normal_forces"].append(helper_forces.clone())
        traces["goal_dwell"].append(dwell.clone())
        traces["recovery_dwell"].append(observed_recovery_dwell.clone())
        traces["recovery_ready"].append(recovery_ready.clone())
        traces["recovery_metrics"].append({k: v.clone() if isinstance(v, torch.Tensor) else v
                                           for k, v in recovery_metrics.items()})
        entering_retreat = finish & (old_phase == 0)
        retreat_target[entering_retreat] = (env.world.agents[1].state.pos + _rotate(
            torch.tensor([.15, -.15]).expand(n, 2), env.scenario.bar.state.rot[:, 0]))[entering_retreat]
        entering_recovery = finish & (old_phase == 1)
        if entering_recovery.any() and variant != "timer_baseline":
            controller.reset(mask=entering_recovery)
        entering_final = finish & (old_phase == 2)
        anchor_at_handoff[entering_final] = anchor[entering_final]
        if capture_counterfactual and entering_final.any():
            counterfactual_snapshots.append(snapshot_record(env, entering_final.nonzero().flatten().tolist(),
                anchor, history_inputs(geometry_trace, history_action_trace), rows, "post_recovery"))
        finished = finish & (old_phase == 3)
        terminal_pose[finished] = pose[finished]
        terminal_state[finished] = physical_state(env)[finished]
        phase[finish] += 1
        phase_age[finish] = 0
        dwell[finish] = 0
        traces["trajectory"].append(physical_state(env).clone())
        traces["helper_anchor"].append(anchor.clone())

    if not (phase == 4).all():
        raise AssertionError("The bounded phase machine failed to terminate")
    if checkpoint_digest(checkpoints) != before_hash:
        raise AssertionError("Frozen model weights or normalizers changed")
    if norm_max.max() > 1 + 1e-6:
        raise AssertionError("A physical command exceeded the Euclidean force limit")
    terminal_ok, angle_error, translation = task_metrics(terminal_pose, initial_pose, torch.full((n,), goal_pair[1]))
    strict_success = gates.all(1) & terminal_ok & finite & ~wall
    # Task manipulation and a specific force-contact interface are distinct.
    # Passive support may achieve both goal dwells without the active brace gate.
    ordered_success = gates[:, 0] & gates[:, 3] & terminal_ok & finite & ~wall
    ordered_release_success = ordered_success & gates[:, 1]
    errors = torch.stack(prediction_errors, 1) if prediction_errors else torch.empty(n, 0, 3)
    valid = torch.stack(prediction_valid, 1) if prediction_valid else torch.empty(n, 0, dtype=torch.bool)
    for i, row in enumerate(rows):
        failures = [f"{PHASES[j]}_gate" for j in range(4) if not gates[i, j]]
        if not finite[i]: failures.append("nonfinite")
        if wall[i]: failures.append("arena_boundary")
        row.update(task_success=bool(ordered_success[i]), ordered_goal_success=bool(ordered_success[i]),
            ordered_goal_release_success=bool(ordered_release_success[i]),
            interface_completion_success=bool(strict_success[i]), terminal_success=bool(terminal_ok[i]),
            first_goal_success_at_phase_boundary=bool(gates[i, 0]), retreated_clear=bool(gates[i, 1]),
            contact_reacquired=bool(gates[i, 2]), final_goal_dwell_success=bool(gates[i, 3]),
            first_goal_ever_success=bool(ever_first[i]), ever_final_goal_success=bool(ever_final[i]),
            ever_contact_reacquired=bool(recovery_ready_ever[i]), first_success_step=int(first_final[i]),
            maximum_recovery_dwell=int(maximum_recovery_dwell[i]),
            phase_steps={name: int(phase_steps[i, j]) for j, name in enumerate(PHASES[:4])},
            phase_end_steps={name: int(phase_end[i, j]) for j, name in enumerate(PHASES[:4])},
            phase_timeout={name: bool(timed_out[i, j]) for j, name in enumerate(PHASES[:4])},
            failure_reasons=failures, interface_failure_reasons=failures,
            external_goal_failure_reasons=([name for condition, name in
                ((gates[i, 0], "first_goal_dwell"), (gates[i, 3], "final_goal_dwell"),
                 (terminal_ok[i], "terminal_goal"), (finite[i], "finite"), (~wall[i], "arena_clear"))
                if not condition]),
            task_steps=int(actual_steps[i]), prelude_steps=prelude_steps,
            total_steps=int(actual_steps[i]) + prelude_steps, maximum_steps=horizon + prelude_steps,
            simulated_time=float((actual_steps[i] + prelude_steps) * env.world.dt),
            maximum_simulated_time=(horizon + prelude_steps) * env.world.dt,
            prelude_bar_translation_units=float((initial_pose[i, :2] - prelude_initial_pose[i, :2]).norm()),
            prelude_bar_rotation_rad=float(wrap(initial_pose[i, 2] - prelude_initial_pose[i, 2])),
            prelude_force_integral_by_agent=prelude_force[i].tolist(),
            prelude_work_proxy_by_agent=prelude_work[i].tolist(),
            target_reference="physical pose after common zero-primary prelude",
            final_angle_error_rad=float(angle_error[i]), final_bar_translation_units=float(translation[i]),
            helper_anchor_at_handoff=anchor_at_handoff[i].tolist(),
            recovery_anchor_policy="original_geometry" if variant == "timer_baseline" else "bounded_preload_then_frozen_anchor",
            continuation_is_outside_reset_only_training_support=row["planner"] == "legacy_wm",
            recovery_and_changed_anchor_states_were_not_task_training_data=True,
            maximum_action_norm_by_agent=norm_max[i].tolist(), norm_force_integral_by_agent=force_integral[i].tolist(),
            squared_control_effort_by_agent=effort[i].tolist(), absolute_actuator_work_proxy_by_agent=work[i].tolist(),
            agent_travel_units=travel[i].tolist(),
            geometric_near_contact_fraction_by_agent=(contacts[i] / (actual_steps[i] + prelude_steps)).tolist(),
            helper_dual_force_contact_fraction=float(helper_force_contacts[i].sum() / actual_steps[i]),
            helper_dual_force_contact_fraction_by_phase={name: float(helper_force_contacts[i, j] / phase_steps[i, j])
                                                        for j, name in enumerate(PHASES[:4])},
            model_macro_queries=int(model_queries[i]), model_candidate_predictions=int(model_queries[i] * n_candidates),
            finite=bool(finite[i]), wall=bool(wall[i]))
        if is_wm[i]:
            e = errors[i, valid[i]]
            row["complete_predicted_macros"] = len(e)
            row["frozen_macro_prediction_translation_rmse_units"] = float(e[:, :2].square().sum(1).mean().sqrt()) if len(e) else None
            row["frozen_macro_prediction_angle_rmse_rad"] = float(e[:, 2].square().mean().sqrt()) if len(e) else None
    raw = {key: torch.stack(value, 1) if key != "recovery_metrics" else value for key, value in traces.items()}
    raw.update(rows=rows, physics_profile=profile, prelude_initial_pose=prelude_initial_pose,
               prelude_trajectory=torch.stack(prelude_trajectory, 1),
               prelude_actions=torch.stack(prelude_actions, 1), counterfactual_snapshots=counterfactual_snapshots, phase_names=PHASES, initial_pose=initial_pose, terminal_pose=terminal_pose,
               terminal_state=terminal_state, phase_gates=gates, phase_timeouts=timed_out,
               selected_macros=selected_macros, model_predictions=predictions, model_inputs=model_inputs,
               macro_prediction_errors=errors, macro_prediction_valid=valid,
               final_env_steps=torch.as_tensor(env.steps).clone(), frozen_model_weights_sha256=before_hash,
               recovery_controller_configuration=controller.configuration(),
               resource_accounting={
                   "helper_learned_parameters": 0,
                   "post_handoff_helper_memory_floats": 2,
                   "post_handoff_helper_memory": "fixed world-frame anchor x,y",
                   "recovery_coordinator_memory": "2D anchor; phase/elapsed/dwell integer counters; ready/accepted/failed, primary/helper staging booleans",
                   "primary_planner_memory": "initial target pose, current stage, current five-step macro",
                   "coordinator_gains": {"kp": controller.config.kp, "kd": controller.config.kd},
                   "recovery_sensing": controller.configuration()["recovery_observation"],
                   "post_handoff_sensing": controller.configuration()["post_handoff_helper_observation"],
                   "physical_action_budget_norm": 1.,
                   "primary_analytic_gains": {"angle": 8., "angular_velocity": .8},
                   "energy_metric": "absolute actuator work proxy; separate squared effort and integrated force",
               })
    return raw



def summarize(rows):
    keys = ("physics_profile", "model_training_profile", "substeps", "candidate_bank", "support", "planner", "model_seed", "first_goal_rad", "relative_goal_rad")
    result = []
    for key in sorted({tuple(row[k] for k in keys) for row in rows}):
        selected = [row for row in rows if tuple(row[k] for k in keys) == key]
        n = len(selected)
        entry = dict(zip(keys, key))
        entry.update(episodes=n,
            mean_prelude_bar_translation_units=sum(row["prelude_bar_translation_units"] for row in selected) / n,
            mean_prelude_bar_rotation_rad=sum(row["prelude_bar_rotation_rad"] for row in selected) / n,
            ordered_goal_success_count=sum(row["ordered_goal_success"] for row in selected),
            ordered_goal_release_success_count=sum(row["ordered_goal_release_success"] for row in selected),
            interface_completion_success_count=sum(row["interface_completion_success"] for row in selected),
            terminal_goal_success_count=sum(row["terminal_success"] for row in selected),
            mean_steps=sum(row["total_steps"] for row in selected) / n,
            mean_final_angle_error_rad=sum(row["final_angle_error_rad"] for row in selected) / n,
            mean_final_bar_translation_units=sum(row["final_bar_translation_units"] for row in selected) / n,
            mean_force_integral_by_agent=[sum(row["norm_force_integral_by_agent"][j] for row in selected) / n for j in range(2)],
            mean_work_proxy_by_agent=[sum(row["absolute_actuator_work_proxy_by_agent"][j] for row in selected) / n for j in range(2)],
            finite_all=all(row["finite"] for row in selected), wall_count=sum(row["wall"] for row in selected),
            gate_counts={gate: sum(row[gate] for row in selected) for gate in
                ("first_goal_success_at_phase_boundary", "retreated_clear", "contact_reacquired", "final_goal_dwell_success")})
        measured = [row for row in selected if row.get("complete_predicted_macros", 0)]
        if measured:
            count = sum(row["complete_predicted_macros"] for row in measured)
            entry["complete_predicted_macros"] = count
            for output, metric in (("macro_translation_rmse_units", "frozen_macro_prediction_translation_rmse_units"),
                                   ("macro_angle_rmse_rad", "frozen_macro_prediction_angle_rmse_rad")):
                entry[output] = math.sqrt(sum(row[metric] ** 2 * row["complete_predicted_macros"] for row in measured) / count)
        result.append(entry)
    return result


def merge_snapshot_records(records):
    """Pack different real task times into one replay batch without resetting them.

    Grouping is only for evaluation throughput; each row retains its full state,
    helper anchor, history and source task/stage labels. No averaging occurs.
    """
    if not records:
        raise ValueError("No snapshots to merge")
    first = records[0]["snapshot"]
    if any(record["snapshot"]["physics"] != first["physics"] for record in records):
        raise ValueError("Cannot merge snapshots with different physics")
    def cat(values):
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("Snapshot tensor presence changed")
        return torch.cat(values, 0)
    entities = []
    for index, entity in enumerate(first["entities"]):
        entries = [record["snapshot"]["entities"][index] for record in records]
        state = {key: cat([entry["state"][key] for entry in entries]) for key in entity["state"]}
        action = None if entity["action"] is None else {
            key: cat([entry["action"][key] for entry in entries]) for key in entity["action"]}
        entities.append(dict(name=entity["name"], state=state, action=action))
    snapshot = dict(batch_size=sum(record["snapshot"]["batch_size"] for record in records),
        physics=first["physics"], entities=tuple(entities),
        scenario={key: cat([record["snapshot"]["scenario"][key] for record in records]) for key in first["scenario"]},
        steps=cat([record["snapshot"]["steps"] for record in records]))
    return dict(snapshot=snapshot, anchor=cat([record["anchor"] for record in records]),
        history=cat([record["history"] for record in records]),
        rows=[row for record in records for row in record["rows"]])


@torch.no_grad()
def counterfactual_group(group, checkpoints, *, model_seed=0, legacy_substeps=20,
                         directions=(-35., 35.), strengths=(.45, .75)):
    """Frozen predictions for unseen five-step pushes from true task snapshots.

    Every branch starts with exactly the captured full physical state, including
    the separate stem. All branches use the same frozen helper law. Expanded
    batches can have floating point drift, so this is not a bitwise replay claim.
    These evaluation labels are never fed to training or model selection.
    """
    programs = list(itertools.product(directions, strengths))
    if not programs or any(abs(strength) > 1 for _, strength in programs):
        raise ValueError("Counterfactual programs require finite bounded forces")
    profile = group["physics_profile"]
    substeps = group["rows"][0]["substeps"]
    before = checkpoint_digest(checkpoints)
    output_rows, blocks = [], []
    for record in ([merge_snapshot_records(group["counterfactual_snapshots"])] if group["counterfactual_snapshots"] else []):
        selected = [i for i, row in enumerate(record["rows"]) if is_model_planner(row["planner"])]
        if not selected:
            continue
        index = torch.tensor([i for i in selected for _ in programs])
        n = len(index)
        env = make_profile(profile, num_envs=n, substeps=substeps, seed=101)
        restore(env, snapshot_from_record(record), index)
        rows = [dict(record["rows"][i], intervention_direction_deg=d, intervention_strength=s)
                for i in selected for d, s in programs]
        anchor = record["anchor"][index]
        history = record["history"][index]
        angles = env.scenario.bar.state.rot[:, 0]
        direction = angles + torch.tensor([math.radians(row["intervention_direction_deg"]) for row in rows])
        strength = torch.tensor([row["intervention_strength"] for row in rows])
        primary = torch.stack((direction.sin(), -direction.cos()), 1) * strength[:, None]
        coarse, full = model_features(env, primary, MACRO_STEPS, anchor)
        for i, row in enumerate(rows):
            reference = (_entry(checkpoints, row, model_seed)["reference_substeps"]
                         if row["planner"].startswith("new_") else legacy_substeps)
            coarse[i, 15] = float(reference)
            full[i, 15] = float(reference)
        prediction = predict_rows(checkpoints, rows, coarse[:, None], full[:, None],
                                  history[:, None], torch.ones(n, dtype=torch.bool),
                                  model_seed, legacy_substeps)[:, 0]
        initial = env.scenario.t_pose().clone()
        for _ in range(MACRO_STEPS):
            helper = _support_commands(env, anchor, rows)
            env.step([primary, helper])
        response = env.scenario.t_pose() - initial
        response[:, 2] = wrap(response[:, 2])
        errors = prediction - response
        finite = _finite(env)
        for i, row in enumerate(rows):
            output_rows.append(dict(physics_profile=profile["name"], substeps=substeps,
                planner=row["planner"], support=row["support"], model_seed=model_seed,
                model_training_profile=row["model_training_profile"],
                candidate_bank=row["candidate_bank"], first_goal_rad=row["first_goal_rad"],
                relative_goal_rad=row["relative_goal_rad"], snapshot_stage=row["snapshot_stage"],
                composition_row=row["composition_row"], initial_angle_deg=row["initial_angle_deg"],
                nuisance_id=row["nuisance_id"], direction_deg=row["intervention_direction_deg"],
                strength=row["intervention_strength"], prediction=prediction[i].tolist(),
                response=response[i].tolist(), error=errors[i].tolist(), finite=bool(finite[i])))
        blocks.append(dict(rows=rows, coarse=coarse, full=full, history=history,
                           prediction=prediction, response=response, errors=errors))
    if checkpoint_digest(checkpoints) != before:
        raise AssertionError("Frozen model changed during counterfactual evaluation")
    return dict(rows=output_rows, blocks=blocks, programs=programs)


def summarize_counterfactual(rows):
    keys = ("physics_profile", "model_training_profile", "substeps", "support", "planner", "model_seed", "snapshot_stage")
    output = []
    for key in sorted({tuple(row[k] for k in keys) for row in rows}):
        selected = [row for row in rows if tuple(row[k] for k in keys) == key]
        errors = torch.tensor([row["error"] for row in selected])
        response = torch.tensor([row["response"] for row in selected])
        entry = dict(zip(keys, key))
        entry.update(rows=len(selected), rmse_xy_units_rad=errors.square().mean(0).sqrt().tolist(),
            translation_rmse_units=float(errors[:, :2].square().sum(1).mean().sqrt()),
            angular_rmse_rad=float(errors[:, 2].square().mean().sqrt()),
            zero_response_rmse_xy_units_rad=response.square().mean(0).sqrt().tolist(),
            finite_all=all(row["finite"] for row in selected),
            within_task_fraction_count=int(((errors[:, :2].norm(dim=1) < TRANSLATION_TOLERANCE / 3)
                                            & (errors[:, 2].abs() < ANGLE_TOLERANCE / 3)).sum()))
        output.append(entry)
    return output


def compare_refinement(groups):
    """Pair exact protocol/controller/initial condition rows across resolutions."""
    index = {}
    for group in groups:
        for row in group["rows"]:
            key = (row["physics_profile"], row["candidate_bank"], row["support"], row["planner"],
                   row["model_seed"], tuple(row["goal_pair"]), row["initial_angle_deg"],
                   tuple(row["initial_velocity"]), row["initial_angular_velocity"])
            index.setdefault(key, {})[row["substeps"]] = row
    result = []
    for key, resolutions in sorted(index.items()):
        numbers = sorted(resolutions)
        for low, high in zip(numbers[:-1], numbers[1:]):
            a, b = resolutions[low], resolutions[high]
            result.append(dict(physics_profile=key[0], model_training_profile=a["model_training_profile"], candidate_bank=key[1], support=key[2],
                planner=key[3], model_seed=key[4], goal_pair=list(key[5]), initial_angle_deg=key[6],
                initial_velocity=list(key[7]), initial_angular_velocity=key[8],
                low_substeps=low, high_substeps=high,
                ordered_success_disagreement=a["ordered_goal_success"] != b["ordered_goal_success"],
                interface_success_disagreement=a["interface_completion_success"] != b["interface_completion_success"],
                final_angle_error_gap_rad=abs(a["final_angle_error_rad"] - b["final_angle_error_rad"]),
                translation_gap_units=abs(a["final_bar_translation_units"] - b["final_bar_translation_units"]),
                steps_gap=abs(a["total_steps"] - b["total_steps"])))
    return result


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="original")
    parser.add_argument("--new-models", type=Path)
    parser.add_argument("--training-profile", default="unspecified", help="Provenance label of local training physics")
    parser.add_argument("--legacy-models", type=Path, default=Path("data/novelty/abstraction_models.pt"))
    parser.add_argument("--legacy-preload-models", type=Path, default=Path("data/novelty/abstraction_models_preload.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/novelty/stable_composition"))
    parser.add_argument("--substeps", type=int, nargs="+", default=[80, 160])
    parser.add_argument("--angles", type=float, nargs="+", default=[0., 30.])
    parser.add_argument("--supports", choices=SUPPORTS, nargs="+", default=list(SUPPORTS))
    parser.add_argument("--planners", choices=PLANNERS, nargs="+", default=list(PLANNERS))
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--banks", choices=CANDIDATE_BANKS, nargs="+", default=["bank13"])
    parser.add_argument("--goals", choices=("standard", "holdout"), nargs="+", default=["standard", "holdout"])
    parser.add_argument("--counterfactual", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One angle, zero nuisance, budgets5/2/6/5; never scientific result")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    profile = get_profile(args.profile)
    legacy, model_provenance, skipped = load_checkpoints(args.legacy_models, args.legacy_preload_models)
    checkpoints = dict(legacy=legacy, new={})
    if args.new_models:
        checkpoints["new"] = torch.load(args.new_models, map_location="cpu")["checkpoints"]
        model_provenance[str(args.new_models.resolve())] = dict(sha256=file_hash(args.new_models))
    for seed in args.model_seeds:
        validate_checkpoints(checkpoints, args.supports, args.planners, seed)
    sources = [Path(__file__), Path(__file__).with_name("physics_profiles.py"),
               Path(__file__).with_name("abstraction_composition.py"), Path(__file__).with_name("rebrace.py"),
               Path(__file__).with_name("abstraction_transfer.py"), Path(__file__).with_name("physical_abstraction.py"),
               Path(__file__).with_name("stable_local_wm.py"), Path(__file__).with_name("snapshot.py"),
               Path(__file__).with_name("scenario.py")]
    hashes = {str(path): file_hash(path) for path in sources if path.exists()}
    goal_map = dict(standard=(.04, .08), holdout=(.06, .12))
    groups, counterfactuals = [], []
    for ss, bank, goal_name, seed in itertools.product(args.substeps, args.banks, args.goals, args.model_seeds):
        kwargs = dict(profile=profile, substeps=ss, checkpoints=checkpoints, supports=args.supports,
            planners=args.planners, angles_deg=args.angles, goal_pair=goal_map[goal_name],
            model_seed=seed, candidate_bank=bank, capture_counterfactual=args.counterfactual,
            model_training_profile=args.training_profile)
        if args.smoke:
            kwargs.update(angles_deg=[0.], nuisances=[(0., 0., 0.)], rotation_steps=5, retreat_steps=2, max_recovery=6)
        group = run_group(**kwargs)
        groups.append(group)
        print(json.dumps(dict(profile=profile["name"], substeps=ss, bank=bank, goals=goal_name,
            seed=seed, episodes=len(group["rows"]), summary=summarize(group["rows"]))), flush=True)
        if args.counterfactual:
            counterfactuals.append(counterfactual_group(group, checkpoints, model_seed=seed))
    for path, metadata in model_provenance.items():
        if file_hash(path) != metadata["sha256"]:
            raise AssertionError(f"Model file changed during evaluation: {path}")
    for path, sha in hashes.items():
        if file_hash(path) != sha:
            raise AssertionError(f"Evaluation source changed during evaluation: {path}")
    rows = [row for group in groups for row in group["rows"]]
    cf_rows = [row for result in counterfactuals for row in result["rows"]]
    report = dict(schema_version=1, physics_profile=profile, smoke_only=args.smoke,
        arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        frozen_protocol=dict(goal_pairs_rad={key: list(goal_map[key]) for key in args.goals},
            phase_budget_steps=[60, 20, 80, 60], prelude_steps=10, goal_reference="pose after prelude", macro_steps=MACRO_STEPS,
            angle_tolerance_rad=ANGLE_TOLERANCE, translation_tolerance_units=TRANSLATION_TOLERANCE,
            helper_gains=dict(kp=30., kd=4., preload=.01), model_resolution_feature="fixed training reference",
            action_banks={"bank10": "zero + directions -20/0/20 times strengths .3/.6/.9",
                          "bank13": "zero + directions -20/0/20 times strengths .15/.3/.6/.9"},
            counterfactual_programs="heldout directions -35/+35 times strengths .45/.75; five constant steps",
            goal_dwell=GOAL_DWELL, retreat_dwell=RETREAT_DWELL, recovery_dwell=RECOVERY_DWELL),
        model_artifacts=model_provenance, skipped_legacy_artifacts=skipped, source_sha256=hashes,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        elapsed_seconds=time.perf_counter() - started, episode_count=len(rows), summary=summarize(rows), rows=rows,
        counterfactual_summary=summarize_counterfactual(cf_rows), counterfactual_rows=cf_rows,
        refinement_comparisons=compare_refinement(groups),
        limitations=[
            "External ordered-goal success requires both rotation dwell gates and final goal within tolerance. Interface completion additionally requires retreat and active contact-recovery gates; passive is not penalized by that gate in the external-goal metric.",
            "The helper is a goal-independent handwritten finite PD controller, not a learned abstraction. Physical materials changed before all rollouts; historical setup remains a separate control.",
            "Local continuation windows are trained, but retreat/rebrace trajectories and newly set recovery anchors remain transfer conditions. World-anchor recovery uses the same bounded preload handoff rule as the historical protocol.",
            "No task goals, whole composition trajectories or test counterfactual labels were used to train or choose the new models. Bank13 was declared before task outcomes; it is an explicit action-library variant, not silently mixed with bank10.",
            "History model adds two past geometry/action records; full model adds physical velocity/friction sensing. Parameters alone do not account for sensing, memory or computation.",
            "Ten common zero-primary physical preparation steps supply history; their resources count, while goals are relative to the resulting post-preparation pose. Preparation displacement is reported separately.",
            "All early-stage failures continue through later stages and remain failed. Task evaluation uses real simulator transitions without body resets after initialization.",
            "Counterfactual branching uses exact saved state but expanded batch computations may introduce floating-point drift; these branches do not alter the composed controller.",
            "Comparing success alone does not establish abstraction. Prediction error, zero-push control, task authority, numerical refinement and strong full/history baselines must be considered together.",
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".pt")
    torch.save(dict(groups=groups, counterfactuals=counterfactuals, model_artifacts=model_provenance,
                    source_sha256=hashes), raw_path)
    torch.load(raw_path, map_location="cpu")
    report["raw_sha256"] = file_hash(raw_path)
    args.output.with_suffix(".json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), episode_count=len(rows), elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
