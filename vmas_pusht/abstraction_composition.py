"""Physical composition with bounded recovery and explicit, persistent gates.

This is a new evaluation protocol; the historical 150-step experiment is left
unchanged. Every row is retained, including failed early stages and recovery
timeouts. No simulator state is reset after setup. Finished batch rows continue
as zero-action padding in the raw simulator trace, but their recorded endpoint
and resource totals stop at their own completion step.

The primary convergence protocol keeps the frozen model's substeps input fixed
at 20 while refining the simulator. Passing model_substeps=None is an explicitly
different controller and may query beyond its training support (5, 10, 20).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import time
from pathlib import Path

import torch

from vmas_pusht.abstraction_models import predict_checkpoint
from vmas_pusht.abstraction_transfer import (
    ANGLE_TOLERANCE, MACRO_STEPS, PLANNERS, PRELOAD, SUPPORTS,
    TRANSLATION_TOLERANCE, _support_commands, effective_anchor,
    feedback_primary, load_checkpoints, main_candidates, reposition_action,
    task_metrics,
)
from vmas_pusht.boundary import _at_wall, _finite, _plain, _rotate, surface_gaps
from vmas_pusht.physical_abstraction import (
    HELPER_LOCAL, PRIMARY_LOCAL, model_features, norm_bound, physical_state,
    setup_scene,
)
from vmas_pusht.scenario import make, wrap

VARIANTS = ("timer_baseline", "corrected_timer", "guarded_recovery", "guarded_all")
PHASES = ("rotate_first", "retreat", "recovery", "rotate_final", "done")
GOAL_DWELL = 5
RETREAT_DWELL = 3
RECOVERY_DWELL = 5


def checkpoint_digest(checkpoints):
    """Include normalizers as well as learned weights in the freeze check."""
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
    visit(checkpoints)
    return digest.hexdigest()


def query_frozen_model(env, anchor, relative_goal, initial_pose, rows,
                       checkpoints, query_mask, model_substeps=20):
    """Evaluate ten fixed actions only for queried WM rows; never alter physics."""
    n = env.num_envs
    candidates = main_candidates(env.scenario.bar.state.rot[:, 0])
    coarse, _ = model_features(env, torch.zeros(n, 2), MACRO_STEPS, anchor)
    if model_substeps is not None:
        coarse[:, 15] = float(model_substeps)
    inputs = coarse[:, None].expand(n, 10, -1).clone()
    inputs[:, :, 12:14] = candidates
    prediction = torch.zeros(n, 10, 3)
    needed = query_mask & torch.tensor([r["planner"] == "frozen_wm" for r in rows])
    for support in sorted(set(r["support"] for r in rows)):
        idx = torch.tensor([i for i, r in enumerate(rows)
                            if r["support"] == support and bool(needed[i])], dtype=torch.long)
        if len(idx):
            prediction[idx] = predict_checkpoint(checkpoints[support],
                inputs[idx].reshape(-1, 18)).reshape(-1, 10, 3)
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


def _timer_readiness(env, anchor, supports):
    """Conservative old-geometry baseline gate, verified under handoff commands.

    Thresholds mirror the controller contract: persistent helper dual contact,
    a near-contact primary, and low relative point velocity. The timer baseline
    does not get an inward recovery target or an adaptive recovery endpoint.
    """
    from vmas_pusht.rebrace import readiness_metrics
    return readiness_metrics(env, anchor, supports)


@torch.no_grad()
def run_group(*, friction=1.5, substeps=20, variant="guarded_all", checkpoints=None,
              supports=SUPPORTS, planners=PLANNERS, angles_deg=(0., 30.),
              nuisances=((0., 0., 0.), (.03, 0., .2)), rotation_steps=60,
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
    checkpoints = checkpoints or {}
    for support in supports:
        if "frozen_wm" in planners and support not in checkpoints:
            raise ValueError(f"no frozen checkpoint for {support}")
    rows = [dict(support=support, planner=planner, initial_angle_deg=angle,
                 relative_goal_rad=.08, nuisance_id=nid, initial_velocity=list(nuisance[:2]),
                 initial_angular_velocity=nuisance[2], friction_ratio=friction, substeps=substeps,
                 variant=variant, task="rotate_retreat_rebrace_rotate",
                 model_substeps_input=substeps if model_substeps is None else model_substeps,
                 model_conditioning="native_input" if model_substeps is None else "fixed_reference",
                 model_substeps_input_ood=(substeps if model_substeps is None else model_substeps) not in (5, 10, 20))
            for support, planner, angle, (nid, nuisance) in
            itertools.product(supports, planners, angles_deg, enumerate(nuisances))]
    n = len(rows)
    if not n:
        raise ValueError("empty composition experiment")
    before_hash = checkpoint_digest(checkpoints)
    env = make(num_envs=n, device="cpu", friction_ratio=friction, substeps=substeps, seed=101)
    angles = torch.tensor([math.radians(r["initial_angle_deg"]) for r in rows])
    anchor = setup_scene(env, angle=angles, hidden_v=[r["initial_velocity"] for r in rows],
                         hidden_omega=[r["initial_angular_velocity"] for r in rows])
    support_names = [r["support"] for r in rows]
    preload = torch.tensor([s == "preloaded_anchor" for s in support_names])
    anchor += preload[:, None] * _rotate(torch.tensor([-PRELOAD, PRELOAD]).expand(n, 2), angles)
    initial_pose = env.scenario.t_pose().clone()
    controller = RebraceController(env, support_names, max_steps=max_recovery,
                                  confirm_steps=recovery_dwell)
    is_wm = torch.tensor([r["planner"] == "frozen_wm" for r in rows])
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
    finite, wall = _finite(env), _at_wall(env)
    force_integral, effort, work, travel, contacts = [torch.zeros(n, 2) for _ in range(5)]
    helper_force_contacts = torch.zeros(n, 4)
    norm_max = torch.zeros(n, 2)
    macro_start = initial_pose.clone()
    macro_prediction = torch.zeros(n, 3)
    macro_remaining = torch.zeros(n, dtype=torch.long)
    model_queries = torch.zeros(n, dtype=torch.long)
    traces = dict(trajectory=[terminal_state.clone()], helper_anchor=[anchor.clone()],
                  actions=[], phases=[], active=[], surface_gaps=[], goal_dwell=[],
                  helper_normal_forces=[], recovery_dwell=[], recovery_ready=[], recovery_metrics=[])
    selected_macros, predictions, model_inputs = [], [], []
    prediction_errors, prediction_valid = [], []

    for step in range(horizon):
        active = phase < 4
        if not active.any():
            break
        old_phase = phase.clone()
        rotating = active & ((phase == 0) | (phase == 3))
        recovering = active & (phase == 2)
        retreating = active & (phase == 1)
        goal = torch.where(phase == 0, .04, .08)
        renew = rotating & (phase_age % MACRO_STEPS == 0)
        if renew.any():
            analytic = feedback_primary(env, goal, initial_pose)
            main[renew] = analytic[renew]
            chosen = torch.full((n,), -1, dtype=torch.long)
            expected = torch.zeros(n, 3)
            if (renew & is_wm).any():
                proposal, chosen, expected, inputs, queried = query_frozen_model(
                    env, anchor, goal, initial_pose, rows, checkpoints, renew, model_substeps)
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
        first_ok, _, _ = task_metrics(pose, initial_pose, torch.full((n,), .04))
        final_ok, _, _ = task_metrics(pose, initial_pose, torch.full((n,), .08))
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
    terminal_ok, angle_error, translation = task_metrics(terminal_pose, initial_pose, torch.full((n,), .08))
    strict_success = gates.all(1) & terminal_ok & finite & ~wall
    errors = torch.stack(prediction_errors, 1) if prediction_errors else torch.empty(n, 0, 3)
    valid = torch.stack(prediction_valid, 1) if prediction_valid else torch.empty(n, 0, dtype=torch.bool)
    for i, row in enumerate(rows):
        failures = [f"{PHASES[j]}_gate" for j in range(4) if not gates[i, j]]
        if not finite[i]: failures.append("nonfinite")
        if wall[i]: failures.append("arena_boundary")
        row.update(task_success=bool(strict_success[i]), terminal_success=bool(terminal_ok[i]),
            first_goal_success_at_phase_boundary=bool(gates[i, 0]), retreated_clear=bool(gates[i, 1]),
            contact_reacquired=bool(gates[i, 2]), final_goal_dwell_success=bool(gates[i, 3]),
            first_goal_ever_success=bool(ever_first[i]), ever_final_goal_success=bool(ever_final[i]),
            ever_contact_reacquired=bool(recovery_ready_ever[i]), first_success_step=int(first_final[i]),
            maximum_recovery_dwell=int(maximum_recovery_dwell[i]),
            phase_steps={name: int(phase_steps[i, j]) for j, name in enumerate(PHASES[:4])},
            phase_end_steps={name: int(phase_end[i, j]) for j, name in enumerate(PHASES[:4])},
            phase_timeout={name: bool(timed_out[i, j]) for j, name in enumerate(PHASES[:4])},
            failure_reasons=failures, total_steps=int(actual_steps[i]), maximum_steps=horizon,
            simulated_time=float(actual_steps[i] * env.world.dt), maximum_simulated_time=horizon * env.world.dt,
            final_angle_error_rad=float(angle_error[i]), final_bar_translation_units=float(translation[i]),
            helper_anchor_at_handoff=anchor_at_handoff[i].tolist(),
            recovery_anchor_policy="original_geometry" if variant == "timer_baseline" else "bounded_preload_then_frozen_anchor",
            continuation_is_outside_reset_only_training_support=True,
            maximum_action_norm_by_agent=norm_max[i].tolist(), norm_force_integral_by_agent=force_integral[i].tolist(),
            squared_control_effort_by_agent=effort[i].tolist(), absolute_actuator_work_proxy_by_agent=work[i].tolist(),
            agent_travel_units=travel[i].tolist(),
            geometric_near_contact_fraction_by_agent=(contacts[i] / actual_steps[i]).tolist(),
            helper_dual_force_contact_fraction=float(helper_force_contacts[i].sum() / actual_steps[i]),
            helper_dual_force_contact_fraction_by_phase={name: float(helper_force_contacts[i, j] / phase_steps[i, j])
                                                        for j, name in enumerate(PHASES[:4])},
            model_macro_queries=int(model_queries[i]), model_candidate_predictions=int(model_queries[i] * 10),
            finite=bool(finite[i]), wall=bool(wall[i]))
        if is_wm[i]:
            e = errors[i, valid[i]]
            row["complete_predicted_macros"] = len(e)
            row["frozen_macro_prediction_translation_rmse_units"] = float(e[:, :2].square().sum(1).mean().sqrt()) if len(e) else None
            row["frozen_macro_prediction_angle_rmse_rad"] = float(e[:, 2].square().mean().sqrt()) if len(e) else None
    raw = {key: torch.stack(value, 1) if key != "recovery_metrics" else value for key, value in traces.items()}
    raw.update(rows=rows, phase_names=PHASES, initial_pose=initial_pose, terminal_pose=terminal_pose,
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
                   "energy_metric": "absolute actuator work proxy; separate squared effort and integrated force",
               })
    return raw


def summarize(rows):
    result = []
    keys = ("variant", "support", "planner", "substeps")
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        selected = [r for r in rows if tuple(r[k] for k in keys) == key]
        n = len(selected)
        entry = dict(zip(keys, key))
        entry.update(episodes=n, success_count=sum(r["task_success"] for r in selected),
                     terminal_goal_success_count=sum(r["terminal_success"] for r in selected),
                     ever_final_goal_success_count=sum(r["ever_final_goal_success"] for r in selected),
                     mean_steps=sum(r["total_steps"] for r in selected) / n,
                     maximum_steps=max(r["total_steps"] for r in selected),
                     wall_count=sum(r["wall"] for r in selected), finite_all=all(r["finite"] for r in selected),
                     mean_final_angle_error_rad=sum(r["final_angle_error_rad"] for r in selected) / n,
                     mean_final_bar_translation_units=sum(r["final_bar_translation_units"] for r in selected) / n,
                     mean_helper_force_integral=sum(r["norm_force_integral_by_agent"][1] for r in selected) / n,
                     gate_counts={gate: sum(r[gate] for r in selected) for gate in
                         ("first_goal_success_at_phase_boundary", "retreated_clear", "contact_reacquired", "final_goal_dwell_success")},
                     timeout_counts={phase: sum(r["phase_timeout"][phase] for r in selected) for phase in PHASES[:4]})
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("data/novelty/abstraction_models.pt"))
    parser.add_argument("--preload-models", type=Path, default=Path("data/novelty/abstraction_models_preload.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/novelty/composition_v2"))
    parser.add_argument("--frictions", type=float, nargs="+", default=[1.5])
    parser.add_argument("--substeps", type=int, nargs="+", default=[20])
    parser.add_argument("--angles", type=float, nargs="+", default=[0., 30.])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--supports", choices=SUPPORTS, nargs="+", default=list(SUPPORTS))
    parser.add_argument("--planners", choices=PLANNERS, nargs="+", default=list(PLANNERS))
    parser.add_argument("--max-recovery", type=int, default=80)
    parser.add_argument("--model-substeps", type=int, default=20)
    parser.add_argument("--native-model-substeps", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    checkpoints, provenance, skipped = load_checkpoints(args.models, args.preload_models)
    if "frozen_wm" in args.planners and any(s not in checkpoints for s in args.supports):
        parser.error("Missing requested frozen model; no rows are silently removed")
    sources = [Path(__file__), Path(__file__).with_name("rebrace.py"),
               Path(__file__).with_name("abstraction_transfer.py"), Path(__file__).with_name("physical_abstraction.py"),
               Path(__file__).with_name("abstraction_models.py"), Path(__file__).with_name("scenario.py")]
    source_hash = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    groups = []
    for friction, substeps, variant in itertools.product(args.frictions, args.substeps, args.variants):
        group = run_group(friction=friction, substeps=substeps, variant=variant, checkpoints=checkpoints,
                          supports=args.supports, planners=args.planners, angles_deg=args.angles,
                          max_recovery=args.max_recovery,
                          model_substeps=None if args.native_model_substeps else args.model_substeps)
        groups.append(group)
        print(json.dumps(dict(friction=friction, substeps=substeps, variant=variant,
                              summary=summarize(group["rows"]))), flush=True)
    rows = [row for group in groups for row in group["rows"]]
    for path, entry in provenance.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != entry["sha256"]:
            raise AssertionError("Frozen model artifact changed during execution")
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path, sha in source_hash.items()):
        raise AssertionError("Evaluation source changed during execution")
    report = dict(schema_version=2, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        frozen_protocol=dict(rotation_goals_rad=[.04, .08], phase_budget_steps=[60, 20, args.max_recovery, 60],
            goal_dwell_steps=GOAL_DWELL, retreat_clear_dwell_steps=RETREAT_DWELL, recovery_dwell_steps=RECOVERY_DWELL,
            angle_tolerance_rad=ANGLE_TOLERANCE, translation_tolerance_units=TRANSLATION_TOLERANCE,
            force_norm_bound=1., action_macro_steps=MACRO_STEPS,
            timer_baseline_recovery="outside-point PD then final five steps under actual frozen handoff law",
            model_substeps_input=None if args.native_model_substeps else args.model_substeps),
        model_artifacts=provenance, skipped_model_artifacts=skipped, source_sha256=source_hash,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        elapsed_seconds=time.perf_counter() - started, episode_count=len(rows), summary=summarize(rows), rows=rows,
        limitations=[
            "All early-stage failures and recovery timeouts are retained; later stages run for diagnosis but cannot repair a failed sequential gate.",
            "Success requires all four persistent gates, finite state and no arena-boundary contact. Terminal angle alone is not success.",
            "This finite contact-rotation microtask is not the full original PushT benchmark.",
            "The helper remains a goal-independent handwritten controller. Recovery uses geometric T pose and velocity sensing, two-dimensional anchor memory, phase and dwell counters; this is not learned complexity reduction.",
            "Recovery can introduce a new world anchor; continuation and repositioned configurations are outside reset-only model training support.",
            "Passive support receives zero force during both rotation stages and handoff verification; it receives explicit finite forces only during retreat and approach.",
            "Fixed-reference model substeps conditioning isolates physics refinement. Native-input runs change the frozen policy and are not the primary convergence check.",
            "Raw traces include zero-action padding after per-row completion; only steps marked active contribute to endpoints or resource metrics.",
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".pt")
    torch.save(dict(groups=groups, model_artifacts=provenance, source_sha256=source_hash), raw_path)
    torch.load(raw_path, map_location="cpu")
    report["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(episode_count=len(rows), elapsed_seconds=report["elapsed_seconds"],
                         output=str(args.output), summary=report["summary"]), indent=2))


if __name__ == "__main__":
    main()
