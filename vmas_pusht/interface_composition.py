"""Full composed tasks using frozen local outcome models and real action options.

All task stages run from the initial scene through ordinary env.step calls.
Navigation and contact acquisition are part of the rotation-stage budget. The
external ordered-goal score remains separate from the recovery contact gate.
Only independently trained, frozen local outcome models may select WM options.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
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
    GOAL_DWELL, RETREAT_DWELL, RECOVERY_DWELL, PHASES, checkpoint_digest,
)
from vmas_pusht.abstraction_transfer import SUPPORTS, PRELOAD, _support_commands, task_metrics
from vmas_pusht.boundary import _at_wall, _finite, _plain, _rotate, surface_gaps
from vmas_pusht.physical_abstraction import coarse_state, norm_bound, physical_state, setup_scene
from vmas_pusht.physics_profiles import get_profile, make_profile
from vmas_pusht.rebrace import RebraceController, helper_contact_forces
from vmas_pusht.recovery_attribution import PhysicsObserver
from vmas_pusht.scenario import wrap
from vmas_pusht.stable_composition import history_inputs, snapshot_record


MODEL_KINDS = dict(wm_coarse="coarse_body_small", wm_full="full_matched",
    wm_history="history_matched", wm_large="full_large", wm_full_direct="full_matched")
PLANNERS = tuple(MODEL_KINDS) + ("feedback_upper", "feedback_lower", "true_option")
DEFAULT_PLANNERS = tuple(MODEL_KINDS) + ("feedback_upper", "feedback_lower")


def select_options(*args, **kwargs):
    """Late import allows the execution contract to be tested independently."""
    from vmas_pusht.option_planner import select_options as selector
    return selector(*args, **kwargs)


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    if isinstance(value, list):
        return [_clone(v) for v in value]
    return value


def _interface_memory(interface):
    return dict(primitive=asdict(interface.state),
        **{key: getattr(interface, key).clone() for key in interface._tensor_fields})


def _validate_models(model_pack, supports, planners, model_seed):
    checkpoints = (model_pack or {}).get("checkpoints", {})
    for planner in planners:
        if planner not in PLANNERS:
            raise ValueError(f"Unknown planner {planner}")
        for support in supports:
            if support not in SUPPORTS:
                raise ValueError(f"Unknown support {support}")
            if planner in MODEL_KINDS:
                key = f"{support}/{MODEL_KINDS[planner]}/seed{model_seed}"
                if key not in checkpoints:
                    raise ValueError(f"Missing frozen local option model: {key}")
    return checkpoints


def _measurement_totals(measurements, masks):
    if not measurements:
        return {}
    # Every source measurement is per real environment step. Padding after a
    # row finishes cannot contribute to its work or force accounting.
    selected = ("actuator_signed_work", "actuator_absolute_work", "actuator_positive_work",
        "actuator_negative_work", "force_norm_integral", "squared_force_integral", "agent_path_length")
    output = {}
    for key in selected:
        values = torch.stack([entry[key] for entry in measurements], 1)
        output[key] = (values * torch.stack(masks, 1)[:, :, None]).sum(1)
    return output


@torch.no_grad()
def run_group(*, profile="dry_light", substeps=160, model_pack=None,
              planners=DEFAULT_PLANNERS, supports=SUPPORTS, angles_deg=(0., 30.),
              goal_pair=(.04, .08), model_seed=0,
              nuisances=((0., 0., 0.), (.045, 0., .3), (-.045, 0., -.3)),
              prelude_steps=10, rotation_steps=60, retreat_steps=20, max_recovery=80,
              goal_dwell=GOAL_DWELL, retreat_dwell=RETREAT_DWELL,
              recovery_dwell=RECOVERY_DWELL, observe=True, capture_counterfactual=True):
    """Execute every stage, retaining failed prefixes and exact physical clocks.

    The selector reads task goals; the action interface and recovery controller
    do not. Options are atomic except for the original goal-dwell and deadline
    exits. Interrupted outcomes are saved but not treated as full option labels.
    """
    from vmas_pusht.action_interface import ActionInterface

    if prelude_steps < 10:
        raise ValueError("Two real history windows require at least ten prelude steps")
    if rotation_steps < 5 or rotation_steps % 5:
        raise ValueError("Rotation budget must be a positive multiple of five")
    if min(retreat_steps, max_recovery, goal_dwell, retreat_dwell, recovery_dwell) < 1:
        raise ValueError("Phase budgets and dwell requirements must be positive")
    if max_recovery < recovery_dwell + 1:
        raise ValueError("Recovery budget must permit acquisition and confirmation")
    if len(goal_pair) != 2 or not 0 < goal_pair[0] < goal_pair[1]:
        raise ValueError("Expected two ordered positive rotation goals")
    if substeps < 1:
        raise ValueError("Substeps must be positive")
    checkpoints = _validate_models(model_pack, supports, planners, model_seed)
    before_digest = checkpoint_digest(checkpoints)
    profile = get_profile(profile)
    rows = [dict(support=support, planner=planner, initial_angle_deg=angle,
        first_goal_rad=goal_pair[0], relative_goal_rad=goal_pair[1], goal_pair=list(goal_pair),
        model_seed=model_seed, physics_profile=profile["name"], substeps=substeps,
        nuisance_id=nid, initial_velocity=list(nuisance[:2]), initial_angular_velocity=nuisance[2],
        task="rotate_retreat_rebrace_rotate", variant="guarded_all_with_atomic_options",
        candidate_bank="atomic_bank13_push_durations_5_10_20",
        model_kind=MODEL_KINDS.get(planner), expanded_final_interface=planner != "wm_full_direct")
        for support, planner, angle, (nid, nuisance) in
        itertools.product(supports, planners, angles_deg, enumerate(nuisances))]
    n = len(rows)
    if not n:
        raise ValueError("Empty task matrix")
    env = make_profile(profile, num_envs=n, substeps=substeps, seed=101)
    angles = torch.tensor([math.radians(row["initial_angle_deg"]) for row in rows])
    anchor = setup_scene(env, angle=angles, hidden_v=[row["initial_velocity"] for row in rows],
                         hidden_omega=[row["initial_angular_velocity"] for row in rows])
    support_names = [row["support"] for row in rows]
    preload = torch.tensor([support == "preloaded_anchor" for support in support_names])
    anchor += preload[:, None] * _rotate(torch.tensor([-PRELOAD, PRELOAD]).expand(n, 2), angles)
    prelude_initial_pose = env.scenario.t_pose().clone()
    prelude_trajectory = [physical_state(env).clone()]
    geometry = [coarse_state(env).clone()]
    history_actions, prelude_actions, prelude_measurements = [], [], []
    all_measurements, measurement_masks = [], []
    prelude_masks = []
    force_integral, squared_effort, work_proxy, travel = [torch.zeros(n, 2) for _ in range(4)]
    norm_max = torch.zeros(n, 2)
    finite, wall = _finite(env), _at_wall(env)
    with PhysicsObserver(env) if observe else nullcontext() as observer:
        for _ in range(prelude_steps):
            command = torch.stack((torch.zeros(n, 2), _support_commands(env, anchor, rows)), 1)
            old_pos = torch.stack([a.state.pos.clone() for a in env.world.agents], 1)
            old_vel = torch.stack([a.state.vel.clone() for a in env.world.agents], 1)
            if observer is not None:
                observer.begin_step()
            env.step([command[:, 0], command[:, 1]])
            if observer is not None:
                measured = observer.end_step()
                prelude_measurements.append(measured)
                all_measurements.append(measured)
                measurement_masks.append(torch.ones(n, dtype=torch.bool))
                prelude_masks.append(torch.ones(n, dtype=torch.bool))
            new_pos = torch.stack([a.state.pos for a in env.world.agents], 1)
            new_vel = torch.stack([a.state.vel for a in env.world.agents], 1)
            norms = command.norm(dim=-1)
            norm_max = torch.maximum(norm_max, norms)
            force_integral += norms * env.scenario.agent_force * env.world.dt
            squared_effort += norms.square() * env.world.dt
            work_proxy += (command*env.scenario.agent_force*(old_vel+new_vel)/2).sum(-1).abs()*env.world.dt
            travel += (new_pos-old_pos).norm(dim=-1)
            finite &= _finite(env); wall |= _at_wall(env)
            prelude_actions.append(command.clone()); history_actions.append(command.clone())
            prelude_trajectory.append(physical_state(env).clone()); geometry.append(coarse_state(env).clone())

        initial_pose = env.scenario.t_pose().clone()
        initial_force = force_integral.clone()
        interface = ActionInterface(env, support_names, anchor)
        recovery = RebraceController(env, support_names, max_steps=max_recovery, confirm_steps=recovery_dwell)
        phase = torch.zeros(n, dtype=torch.long)
        phase_age = torch.zeros_like(phase)
        phase_steps = torch.zeros(n, 4, dtype=torch.long)
        phase_end = torch.full((n, 4), -1, dtype=torch.long)
        gates, timeouts = torch.zeros(n, 4, dtype=torch.bool), torch.zeros(n, 4, dtype=torch.bool)
        dwell = torch.zeros(n, dtype=torch.long)
        actual_steps = torch.zeros(n, dtype=torch.long)
        terminal_pose, terminal_state = initial_pose.clone(), physical_state(env).clone()
        retreat_target = torch.zeros(n, 2)
        anchor_at_handoff = torch.full((n, 2), float("nan"))
        max_recovery_dwell = torch.zeros(n, dtype=torch.long)
        pending = torch.full((n,), -1, dtype=torch.long)
        option_records, queries, task_measurements = [], [], []
        trace = dict(trajectory=[physical_state(env).clone()], helper_anchor=[anchor.clone()],
            actions=[], active=[], phases=[], goal_dwell=[], surface_gaps=[], helper_normal_forces=[],
            recovery_ready=[], recovery_dwell=[], interface_mode=[], interface_busy=[], interface_phase=[])
        controller_memory = []
        snapshots = []
        if capture_counterfactual:
            record = snapshot_record(env, list(range(n)), anchor, history_inputs(geometry, history_actions), rows, "post_prelude")
            record["interface_memory"] = _interface_memory(interface)
            snapshots.append(record)
        limits = torch.tensor([rotation_steps, retreat_steps, max_recovery, rotation_steps])
        horizon = 2*rotation_steps+retreat_steps+max_recovery

        def close_options(mask, step, interrupted, reason):
            pose = env.scenario.t_pose()
            for i in mask.nonzero().flatten().tolist():
                index = int(pending[i])
                if index < 0:
                    raise AssertionError("Finished option has no corresponding start record")
                item = option_records[index]
                delta = pose[i]-torch.tensor(item["start_pose"])
                delta[2] = wrap(delta[2])
                steps = step+1-item["start_step"]
                failed = bool(interface.failed[i])
                item.update(end_step=step+1, end_env_step=float(env.steps[i]),
                    elapsed_steps=steps, elapsed_seconds=steps*env.world.dt,
                    endpoint_pose=pose[i].tolist(), actual_response=delta.tolist(),
                    completed=not interrupted and not failed, failed=failed,
                    interrupted=interrupted, termination_reason=reason,
                    end_contact_mode=int(interface.mode[i]), end_route_phase=int(interface.state.phase[i]))
                prediction = item.get("predicted_outcome")
                if prediction is not None and not interrupted:
                    actual = torch.cat((delta, torch.tensor([steps*env.world.dt, float(not failed)])))
                    item["prediction_error"] = (torch.tensor(prediction)-actual).tolist()
                pending[i] = -1

        for step in range(horizon):
            active = phase < 4
            if not active.any():
                break
            old_phase = phase.clone()
            rotating = active & ((phase == 0) | (phase == 3))
            retreating, recovering = active & (phase == 1), active & (phase == 2)
            goals = torch.where(phase == 0, goal_pair[0], goal_pair[1])
            remaining = torch.where(rotating, rotation_steps-phase_age, torch.zeros_like(phase_age))
            query_mask = rotating & ~interface.busy
            if query_mask.any():
                routes_allowed = rotating & (phase == 3) & torch.tensor([r["planner"] != "wm_full_direct" for r in rows])
                options, diagnostics = select_options(env, interface, rows, initial_pose, goals,
                    remaining, history_inputs(geometry, history_actions), query_mask,
                    model_pack=model_pack, allow_routes=routes_allowed)
                start_mode = interface.mode.clone()
                interface.start(options, mask=query_mask)
                queries.append(dict(step=step, query_mask=query_mask.clone(), phase=phase.clone(),
                    allow_routes=routes_allowed.clone(), remaining_steps=remaining.clone(),
                    options=options.clone(), diagnostics=_clone(diagnostics)))
                predicted = diagnostics.get("selected_prediction")
                chosen = diagnostics.get("chosen_index")
                for i in query_mask.nonzero().flatten().tolist():
                    estimate = None
                    if predicted is not None and torch.is_tensor(predicted) and torch.isfinite(predicted[i]).all():
                        estimate = predicted[i].tolist()
                    option_records.append(dict(row=i, support=rows[i]["support"], planner=rows[i]["planner"],
                        task_phase=PHASES[int(phase[i])], start_step=step, start_env_step=float(env.steps[i]),
                        start_pose=env.scenario.t_pose()[i].tolist(), start_contact_mode=int(start_mode[i]),
                        option=options[i].tolist(), predicted_outcome=estimate,
                        chosen_index=None if chosen is None else int(chosen[i]),
                        available_steps=int(remaining[i]), route_allowed=bool(routes_allowed[i])))
                    pending[i] = len(option_records)-1
            command = torch.zeros(n, 2, 2)
            if rotating.any():
                proposed = interface.command(active_mask=rotating)
                command[rotating] = proposed[rotating]
            if retreating.any():
                helper = env.world.agents[1]
                retreat = norm_bound(30.*(retreat_target-helper.state.pos)-4.*helper.state.vel)
                command[retreating, 1] = retreat[retreating]
            if recovering.any():
                command[recovering] = recovery.step()[recovering]
            old_pos = torch.stack([a.state.pos.clone() for a in env.world.agents], 1)
            old_vel = torch.stack([a.state.vel.clone() for a in env.world.agents], 1)
            busy_before = interface.busy.clone()
            if observer is not None:
                observer.begin_step()
            env.step([command[:, 0], command[:, 1]])
            if observer is not None:
                measured = observer.end_step()
                task_measurements.append(measured); all_measurements.append(measured)
                measurement_masks.append(active.clone())
            if rotating.any():
                interface.observe(active_mask=rotating)
            new_pos = torch.stack([a.state.pos for a in env.world.agents], 1)
            new_vel = torch.stack([a.state.vel for a in env.world.agents], 1)
            norms = command.norm(dim=-1)
            norm_max = torch.maximum(norm_max, norms)
            force_integral += norms*env.scenario.agent_force*env.world.dt*active[:, None]
            squared_effort += norms.square()*env.world.dt*active[:, None]
            work_proxy += (command*env.scenario.agent_force*(old_vel+new_vel)/2).sum(-1).abs()*env.world.dt*active[:, None]
            travel += (new_pos-old_pos).norm(dim=-1)*active[:, None]
            finite &= ~active | _finite(env); wall |= active & _at_wall(env)
            actual_steps += active; phase_age += active
            phase_steps[active, old_phase[active]] += 1
            pose = env.scenario.t_pose()
            first_ok, _, _ = task_metrics(pose, initial_pose, torch.full((n,), goal_pair[0]))
            final_ok, _, _ = task_metrics(pose, initial_pose, torch.full((n,), goal_pair[1]))
            gaps = surface_gaps(env)
            goal_ok = torch.where(old_phase == 0, first_ok, final_ok)
            retreat_ok = gaps[:, 1].amin(-1) > .01
            pass_now = torch.where(rotating, goal_ok, retreat_ok)
            dwell = torch.where(active & ~recovering & pass_now, dwell+1, torch.zeros_like(dwell))
            recovery_ready = torch.zeros(n, dtype=torch.bool)
            observed_recovery_dwell = torch.zeros(n, dtype=torch.long)
            if recovering.any():
                metrics = recovery.update(active_mask=recovering)
                recovery_ready = metrics["ready"] & recovering
                observed_recovery_dwell = metrics["dwell"]
                anchor[recovering] = metrics["anchor"][recovering]
                max_recovery_dwell = torch.maximum(max_recovery_dwell,
                    torch.where(recovering, observed_recovery_dwell, torch.zeros_like(max_recovery_dwell)))
            age_limit = torch.zeros(n, dtype=torch.long)
            age_limit[active] = limits[old_phase[active]]
            reached_budget = active & (phase_age >= age_limit)
            gate = torch.where(recovering, recovery_ready,
                dwell >= torch.where(retreating, retreat_dwell, goal_dwell))
            finish = active & (gate | reached_budget)
            gates[finish, old_phase[finish]] = gate[finish]
            timeouts[finish, old_phase[finish]] = ~gate[finish]
            phase_end[finish, old_phase[finish]] = step+1

            ended = rotating & busy_before & ~interface.busy
            if ended.any():
                close_options(ended, step, False, "option_finished")
            interrupted = finish & rotating & interface.busy
            if interrupted.any():
                close_options(interrupted & gate, step, True, "stage_goal_dwell")
                close_options(interrupted & ~gate, step, True, "stage_deadline")
                interface.cancel(interrupted)
            trace["actions"].append(command.clone()); trace["active"].append(active.clone())
            trace["phases"].append(old_phase.clone()); trace["goal_dwell"].append(dwell.clone())
            trace["surface_gaps"].append(gaps.clone())
            trace["helper_normal_forces"].append(helper_contact_forces(env).norm(dim=-1))
            trace["recovery_ready"].append(recovery_ready.clone())
            trace["recovery_dwell"].append(observed_recovery_dwell.clone())
            trace["interface_mode"].append(interface.mode.clone())
            trace["interface_busy"].append(interface.busy.clone())
            trace["interface_phase"].append(interface.state.phase.clone())
            history_actions.append(command.clone()); geometry.append(coarse_state(env).clone())
            entering_retreat = finish & (old_phase == 0)
            retreat_target[entering_retreat] = (env.world.agents[1].state.pos+_rotate(
                torch.tensor([.15, -.15]).expand(n, 2), env.scenario.bar.state.rot[:, 0]))[entering_retreat]
            entering_recovery = finish & (old_phase == 1)
            if entering_recovery.any():
                recovery.reset(mask=entering_recovery)
            entering_final = finish & (old_phase == 2)
            if entering_final.any():
                anchor_at_handoff[entering_final] = anchor[entering_final]
                interface.reset_contact(entering_final, anchor=anchor, mode=3)
                if capture_counterfactual:
                    record = snapshot_record(env, entering_final.nonzero().flatten().tolist(), anchor,
                        history_inputs(geometry, history_actions), rows, "post_recovery")
                    # Memory is explicitly recorded in its original batch layout;
                    # selected physical rows are identified by composition_row.
                    record["interface_memory_full_batch"] = _interface_memory(interface)
                    snapshots.append(record)
            finished = finish & (old_phase == 3)
            terminal_pose[finished] = pose[finished]; terminal_state[finished] = physical_state(env)[finished]
            phase[finish] += 1; phase_age[finish] = 0; dwell[finish] = 0
            trace["trajectory"].append(physical_state(env).clone())
            trace["helper_anchor"].append(anchor.clone())
            controller_memory.append(dict(interface=_interface_memory(interface),
                task_phase=phase.clone(), phase_age=phase_age.clone(), task_goal_dwell=dwell.clone(),
                recovery={key: getattr(recovery, key).clone() for key in
                    ("anchor", "phase", "elapsed", "dwell", "ready", "accepted", "failed", "primary_staged", "helper_staged")}))

    if not (phase == 4).all() or (pending >= 0).any():
        raise AssertionError("The bounded task or an option record did not terminate")
    if checkpoint_digest(checkpoints) != before_digest:
        raise AssertionError("Frozen model weights or normalizers changed")
    if norm_max.max() > 1.000001:
        raise AssertionError("Unit Euclidean actuator bound exceeded")
    terminal_ok, angle_error, translation = task_metrics(terminal_pose, initial_pose, torch.full((n,), goal_pair[1]))
    ordered = gates[:, 0] & gates[:, 3] & terminal_ok & finite & ~wall
    strict = gates.all(1) & terminal_ok & finite & ~wall
    totals = _measurement_totals(all_measurements, measurement_masks)
    prelude_totals = _measurement_totals(prelude_measurements, prelude_masks)
    for i, row in enumerate(rows):
        own_options = [item for item in option_records if item["row"] == i]
        prediction_errors = [item["prediction_error"] for item in own_options if "prediction_error" in item]
        row.update(task_success=bool(ordered[i]), ordered_goal_success=bool(ordered[i]),
            ordered_goal_release_success=bool(ordered[i] & gates[i, 1]),
            interface_completion_success=bool(strict[i]), terminal_success=bool(terminal_ok[i]),
            first_goal_success_at_phase_boundary=bool(gates[i, 0]), retreated_clear=bool(gates[i, 1]),
            contact_reacquired=bool(gates[i, 2]), final_goal_dwell_success=bool(gates[i, 3]),
            maximum_recovery_dwell=int(max_recovery_dwell[i]),
            phase_steps={name: int(phase_steps[i, j]) for j, name in enumerate(PHASES[:4])},
            phase_end_steps={name: int(phase_end[i, j]) for j, name in enumerate(PHASES[:4])},
            phase_timeout={name: bool(timeouts[i, j]) for j, name in enumerate(PHASES[:4])},
            task_steps=int(actual_steps[i]), prelude_steps=prelude_steps,
            total_steps=int(actual_steps[i])+prelude_steps, maximum_steps=horizon+prelude_steps,
            simulated_time=(int(actual_steps[i])+prelude_steps)*env.world.dt,
            final_angle_error_rad=float(angle_error[i]), final_bar_translation_units=float(translation[i]),
            maximum_action_norm_by_agent=norm_max[i].tolist(), norm_force_integral_by_agent=force_integral[i].tolist(),
            prelude_force_integral_by_agent=initial_force[i].tolist(),
            squared_control_effort_by_agent=squared_effort[i].tolist(),
            absolute_actuator_work_proxy_by_agent=work_proxy[i].tolist(), agent_travel_units=travel[i].tolist(),
            finite=bool(finite[i]), wall=bool(wall[i]), helper_anchor_at_handoff=anchor_at_handoff[i].tolist(),
            option_queries=len(own_options), completed_options=sum(item["completed"] for item in own_options),
            failed_options=sum(item["failed"] for item in own_options),
            interrupted_options=sum(item["interrupted"] for item in own_options),
            route_options_started=sum(item["option"][0] > 0 for item in own_options),
            route_options_completed=sum(item["option"][0] > 0 and item["completed"] for item in own_options),
            route_contacts_acquired=sum(item["option"][0] > 0 and item["end_route_phase"] >= 3
                and not item["failed"] for item in own_options),
            target_reference="physical pose after common zero-primary prelude",
            external_goal_failure_reasons=[name for condition, name in
                ((gates[i, 0], "first_goal_dwell"), (gates[i, 3], "final_goal_dwell"),
                 (terminal_ok[i], "terminal_goal"), (finite[i], "finite"), (~wall[i], "arena_clear")) if not condition])
        for key, values in totals.items():
            row[key+"_by_agent"] = values[i].tolist()
        for key, values in prelude_totals.items():
            row["prelude_"+key+"_by_agent"] = values[i].tolist()
        if prediction_errors:
            error = torch.tensor(prediction_errors)
            row["complete_option_predictions"] = len(prediction_errors)
            row["option_prediction_translation_rmse_units"] = float(error[:, :2].square().sum(1).mean().sqrt())
            row["option_prediction_angular_rmse_rad"] = float(error[:, 2].square().mean().sqrt())
            row["option_prediction_elapsed_rmse_seconds"] = float(error[:, 3].square().mean().sqrt())
    raw = {key: torch.stack(value, 1) for key, value in trace.items()}
    raw.update(rows=rows, physics_profile=profile, initial_pose=initial_pose,
        prelude_initial_pose=prelude_initial_pose, prelude_trajectory=torch.stack(prelude_trajectory, 1),
        prelude_actions=torch.stack(prelude_actions, 1), prelude_measurements=prelude_measurements,
        measurements=task_measurements, counterfactual_snapshots=snapshots,
        terminal_pose=terminal_pose, terminal_state=terminal_state, phase_gates=gates,
        phase_timeouts=timeouts, phase_names=PHASES, option_records=option_records, queries=queries,
        controller_memory=controller_memory, recovery_controller_configuration=recovery.configuration(),
        final_env_steps=env.steps.clone(), frozen_model_weights_sha256=before_digest,
        resource_accounting=dict(helper_learned_parameters=0, post_handoff_helper_memory_floats=2,
            primary_interface="current face, route phases, anchor, dwell, elapsed, and held-force option memory",
            exact_substep_work=observe, observation_history="two real five-step windows; 34 stored values",
            physical_force_norm_bound=1., navigation_counts_toward_rotation_budget=True))
    return raw


def summarize(rows):
    keys = ("support", "planner", "model_seed", "substeps", "first_goal_rad", "relative_goal_rad")
    output = []
    for key in sorted({tuple(row[k] for k in keys) for row in rows}):
        selected = [row for row in rows if tuple(row[k] for k in keys) == key]
        output.append(dict(zip(keys, key), episodes=len(selected),
            first_goal_success=sum(row["first_goal_success_at_phase_boundary"] for row in selected),
            final_goal_dwell_success=sum(row["final_goal_dwell_success"] for row in selected),
            ordered_success=sum(row["ordered_goal_success"] for row in selected),
            interface_success=sum(row["interface_completion_success"] for row in selected),
            mean_total_steps=sum(row["total_steps"] for row in selected)/len(selected),
            mean_final_angle_error_rad=sum(row["final_angle_error_rad"] for row in selected)/len(selected),
            mean_final_translation_units=sum(row["final_bar_translation_units"] for row in selected)/len(selected),
            route_episodes=sum(row["route_options_started"] > 0 for row in selected),
            finite_all=all(row["finite"] for row in selected), wall_cases=sum(row["wall"] for row in selected)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/novelty/interface_composition"))
    parser.add_argument("--planners", nargs="+", choices=PLANNERS, default=list(DEFAULT_PLANNERS))
    parser.add_argument("--supports", nargs="+", choices=SUPPORTS, default=list(SUPPORTS))
    parser.add_argument("--angles", nargs="+", type=float, default=[0., 30.])
    parser.add_argument("--substeps", type=int, default=160)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--goals", nargs="+", choices=("standard", "holdout"), default=["standard", "holdout"])
    parser.add_argument("--no-observer", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.with_suffix(".pt").exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Choose a fresh output prefix to preserve existing results")
    if any(p in MODEL_KINDS for p in args.planners) and args.models is None:
        raise ValueError("WM planners require frozen model artifacts")
    model_pack = None
    model_digest = None
    if args.models is not None:
        from vmas_pusht.option_models import load_model_pack
        model_digest = hashlib.sha256(args.models.read_bytes()).hexdigest()
        model_pack = load_model_pack(args.models)
        if hashlib.sha256(args.models.read_bytes()).hexdigest() != model_digest:
            raise AssertionError("Model artifact changed while it was being loaded")
    paths = [Path(__file__)] + [Path("vmas_pusht")/(name+".py") for name in
        ("action_interface", "option_planner", "option_models", "rebrace", "recovery_primitives",
         "recovery_attribution", "snapshot", "scenario", "physics_profiles", "stable_composition")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if args.models is not None:
        hashes[str(args.models)] = model_digest
    started = time.perf_counter()
    groups = [run_group(substeps=args.substeps, model_pack=model_pack, planners=args.planners,
        supports=args.supports, angles_deg=args.angles, model_seed=args.model_seed,
        goal_pair=(.04, .08) if goal == "standard" else (.06, .12), observe=not args.no_observer)
        for goal in args.goals]
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise AssertionError("Frozen code/model artifact changed during evaluation: "+path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_suffix(".pt")
    torch.save(dict(groups=groups, source_sha256=hashes), raw)
    rows = [row for group in groups for row in group["rows"]]
    report = dict(arguments={key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()},
        source_sha256=hashes, git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        raw_path=str(raw), raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started, episode_count=len(rows), rows=rows, summary=summarize(rows),
        protocol=dict(full_task_from_initial_scene=True, training_uses_task_trajectories=False,
            prelude_steps=10, stage_budgets=[60, 20, 80, 60], force_norm_limit=1.,
            first_stage="direct options only", final_stage="expanded except wm_full_direct",
            goal_dwell_steps=5, recovery_dwell_steps=5, retreat_dwell_steps=3,
            outcome_prediction="one atomic option endpoint, elapsed time, and completion estimate"))
    args.output.with_suffix(".json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output), episodes=len(rows), summary=report["summary"],
        elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
