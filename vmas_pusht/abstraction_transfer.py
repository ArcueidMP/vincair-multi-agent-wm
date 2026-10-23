"""Frozen-interface transfer to local rotation goals and a composed microtask.

These are newly defined local angular tasks, not the original full PushT task.
The world model and helper are frozen before goals/planners change. A declared
analytic primary controller is a baseline, not a learned helper or a hidden
planner. Model-predictive queries during execution are outside the reset-only
training distribution; successful fitting of reset responses does not guarantee
that this continuation interface is valid.

No body state is reset, projected, or teleported after setup_scene. Composition
uses finite-force retreat, physical contact reacquisition, primary reposition,
and an explicit helper anchor-memory update from the observed current T pose.
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
from vmas_pusht.boundary import _at_wall, _finite, _rotate, surface_gaps
from vmas_pusht.physical_abstraction import (
    HELPER_LOCAL, PRIMARY_LOCAL, helper_action, model_features, norm_bound,
    physical_state, setup_scene,
)
from vmas_pusht.scenario import make, wrap

SUPPORTS = ("passive", "world_anchor", "preloaded_anchor")
PLANNERS = ("frozen_wm", "analytic_feedback", "zero_push")
MACRO_STEPS = 5
ANGLE_TOLERANCE = .015
TRANSLATION_TOLERANCE = .05
PRELOAD = .01


def main_candidates(angles):
    """Exactly ten distinct, finite, 5-step force macros; no resampling."""
    programs = [(0., 0.)] + list(itertools.product((-20., 0., 20.), (.3, .6, .9)))
    directions = torch.tensor([math.radians(d) for d, _ in programs], device=angles.device)
    strengths = torch.tensor([a for _, a in programs], device=angles.device)
    direction = angles[:, None] + directions[None]
    return torch.stack((direction.sin(), -direction.cos()), -1) * strengths[None, :, None]


def task_metrics(pose, initial_pose, relative_goal, angle_tolerance=ANGLE_TOLERANCE,
                 translation_tolerance=TRANSLATION_TOLERANCE):
    target = initial_pose[:, 2] + relative_goal
    angle_error = wrap(target - pose[:, 2]).abs()
    translation = (pose[:, :2] - initial_pose[:, :2]).norm(dim=-1)
    success = (angle_error < angle_tolerance) & (translation < translation_tolerance)
    return success, angle_error, translation


def feedback_primary(env, relative_goal, initial_pose):
    angle = env.scenario.bar.state.rot[:, 0]
    error = wrap(initial_pose[:, 2] + relative_goal - angle)
    omega = env.scenario.bar.state.ang_vel[:, 0]
    # This contact can push in the positive-angle direction, but cannot pull the
    # bar back through the top surface. Overshoots therefore receive zero force.
    strength = (8.0 * error - .8 * omega).clamp(0., .9)
    return torch.stack((angle.sin(), -angle.cos()), 1) * strength[:, None]


def wm_primary(env, anchor, relative_goal, initial_pose, rows, checkpoints):
    """Choose one declared macro without simulation lookahead or candidate fitting."""
    angles = env.scenario.bar.state.rot[:, 0]
    candidates = main_candidates(angles)
    n, count = candidates.shape[:2]
    coarse, _ = model_features(env, torch.zeros(n, 2), MACRO_STEPS, anchor)
    inputs = coarse[:, None].expand(n, count, -1).clone()
    inputs[:, :, 12:14] = candidates
    prediction = torch.empty(n, count, 3)
    # Predictions are required only on WM rows; analytic baseline rows are
    # never scored with a different support's model or used for model selection.
    needed = torch.tensor([r["planner"] == "frozen_wm" for r in rows])
    prediction[~needed] = 0
    for support in sorted(set(r["support"] for r in rows)):
        indices = torch.tensor([i for i, r in enumerate(rows)
                                if r["support"] == support and r["planner"] == "frozen_wm"])
        if len(indices):
            prediction[indices] = predict_checkpoint(checkpoints[support], inputs[indices].reshape(-1, 18)).reshape(-1, count, 3)
    predicted_pose = env.scenario.t_pose()[:, None] + prediction
    target_angle = initial_pose[:, 2] + relative_goal
    angle_error = wrap(predicted_pose[:, :, 2] - target_angle[:, None])
    translation = (predicted_pose[:, :, :2] - initial_pose[:, None, :2]).norm(dim=-1)
    cost = (angle_error / ANGLE_TOLERANCE).square() + (translation / TRANSLATION_TOLERANCE).square()
    chosen = cost.argmin(1)
    action = candidates[torch.arange(n), chosen]
    return action, chosen, prediction[torch.arange(n), chosen]


def contact_reference(env, local):
    bar = env.scenario.bar
    offset = _rotate(torch.tensor(local).expand(env.num_envs, 2), bar.state.rot[:, 0])
    target = bar.state.pos + offset
    velocity = bar.state.vel + bar.state.ang_vel * torch.stack((-offset[:, 1], offset[:, 0]), 1)
    return target, velocity


def reposition_action(env, agent_index, local):
    target, velocity = contact_reference(env, local)
    agent = env.world.agents[agent_index]
    return norm_bound(30. * (target - agent.state.pos) + 4. * (velocity - agent.state.vel))


def effective_anchor(env, supports):
    reference, _ = contact_reference(env, HELPER_LOCAL)
    offset = _rotate(torch.tensor([-PRELOAD, PRELOAD]).expand(env.num_envs, 2),
                     env.scenario.bar.state.rot[:, 0])
    preload = torch.tensor([s == "preloaded_anchor" for s in supports])
    return reference + preload[:, None] * offset


def _support_commands(env, anchor, rows):
    command = torch.zeros(env.num_envs, 2)
    # Preload is already encoded in anchor, exactly as in its training data.
    # Calling the preloaded helper variant again would apply the offset twice.
    actuated = torch.tensor([r["support"] != "passive" for r in rows])
    command[actuated] = helper_action(env, "world_anchor", anchor)[actuated]
    return command


def _state_dict_digest(checkpoints):
    h = hashlib.sha256()
    for support, checkpoint in sorted(checkpoints.items()):
        h.update(support.encode())
        for name, tensor in sorted(checkpoint["state_dict"].items()):
            h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def run_group(*, friction=1.5, substeps=10, checkpoints=None,
              supports=SUPPORTS, planners=PLANNERS, angles_deg=(0., 30.),
              goals=(.04, .08, .12), nuisances=((0., 0., 0.), (.03, 0., .2)),
              rotation_steps=60, retreat_steps=10, reacquire_steps=20, composed=False):
    if rotation_steps < MACRO_STEPS or rotation_steps % MACRO_STEPS:
        raise ValueError("rotation_steps must be a positive multiple of five")
    if composed and min(retreat_steps, reacquire_steps) < 1:
        raise ValueError("composition requires nonempty physical retreat and reacquisition")
    checkpoints = checkpoints or {}
    for support in supports:
        if "frozen_wm" in planners and support not in checkpoints:
            raise ValueError(f"no frozen checkpoint for {support}")
    before_hash = _state_dict_digest(checkpoints)
    goals = (.08,) if composed else goals
    rows = [dict(support=support, planner=planner, initial_angle_deg=angle,
                 relative_goal_rad=goal, nuisance_id=nid, initial_velocity=list(nuisance[:2]),
                 initial_angular_velocity=nuisance[2], friction_ratio=friction, substeps=substeps,
                 task="rotate_retreat_rebrace_rotate" if composed else "local_rotation")
            for support, planner, angle, goal, (nid, nuisance) in
            itertools.product(supports, planners, angles_deg, goals, enumerate(nuisances))]
    n = len(rows)
    if not n:
        raise ValueError("empty transfer experiment")
    env = make(num_envs=n, device="cpu", friction_ratio=friction, substeps=substeps, seed=101)
    angles = torch.tensor([math.radians(r["initial_angle_deg"]) for r in rows])
    physical_anchor = setup_scene(env, angle=angles,
                                 hidden_v=[r["initial_velocity"] for r in rows],
                                 hidden_omega=[r["initial_angular_velocity"] for r in rows])
    support_names = [r["support"] for r in rows]
    anchor = physical_anchor.clone()
    preload_mask = torch.tensor([s == "preloaded_anchor" for s in support_names])
    anchor += preload_mask[:, None] * _rotate(torch.tensor([-PRELOAD, PRELOAD]).expand(n, 2), angles)
    initial_pose = env.scenario.t_pose().clone()
    initial_state = physical_state(env).clone()
    final_goal = torch.tensor([r["relative_goal_rad"] for r in rows])
    horizon = 2 * rotation_steps + retreat_steps + reacquire_steps if composed else rotation_steps
    phase2_start = rotation_steps + retreat_steps + reacquire_steps
    actions, trajectories, anchors, phases = [], [initial_state], [anchor.clone()], []
    selected_macros, predictions = [], []
    force_integral, effort, work, travel, contacts = [torch.zeros(n, 2) for _ in range(5)]
    norm_max = torch.zeros(n, 2)
    first_success_step = torch.full((n,), -1, dtype=torch.long)
    ever_success = torch.zeros(n, dtype=torch.bool)
    first_goal_at_boundary = torch.zeros(n, dtype=torch.bool)
    first_goal_ever = torch.zeros(n, dtype=torch.bool)
    retreat_clear = torch.zeros(n, dtype=torch.bool)
    reacquired = torch.zeros(n, dtype=torch.bool)
    reacquired_at_boundary = torch.zeros(n, dtype=torch.bool)
    primary_repositioned = torch.zeros(n, dtype=torch.bool)
    finite, wall = _finite(env), _at_wall(env)
    main = torch.zeros(n, 2)
    retreat_target = None
    is_wm = torch.tensor([r["planner"] == "frozen_wm" for r in rows])
    is_zero = torch.tensor([r["planner"] == "zero_push" for r in rows])
    prediction_errors, prediction_valid = [], []
    macro_start_pose = None
    macro_prediction = None
    for step in range(horizon):
        if not composed or step < rotation_steps or step >= phase2_start:
            phase = "rotate_first" if composed and step < rotation_steps else "rotate_final"
            goal = torch.full((n,), .04) if phase == "rotate_first" else final_goal
            local_step = step if step < rotation_steps or not composed else step - phase2_start
            if local_step % MACRO_STEPS == 0:
                analytic = feedback_primary(env, goal, initial_pose)
                chosen = torch.full((n,), -1, dtype=torch.long)
                expected = torch.zeros(n, 3)
                if is_wm.any():
                    proposal, chosen, expected = wm_primary(env, anchor, goal, initial_pose, rows, checkpoints)
                    main = torch.where(is_wm[:, None], proposal, analytic)
                else:
                    main = analytic
                main[is_zero] = 0
                chosen[~is_wm] = -1
                macro_start_pose = env.scenario.t_pose().clone()
                macro_prediction = expected.clone()
                selected_macros.append(dict(step=step, chosen_index=chosen.clone()))
                predictions.append(dict(step=step, predicted_response=expected.clone(), wm_rows=is_wm.clone()))
            helper = _support_commands(env, anchor, rows)
        elif step < rotation_steps + retreat_steps:
            phase = "retreat"
            if step == rotation_steps:
                retreat_target = env.world.agents[1].state.pos + _rotate(
                    torch.tensor([.15, -.15]).expand(n, 2), env.scenario.bar.state.rot[:, 0])
            ag = env.world.agents[1]
            helper = norm_bound(30. * (retreat_target - ag.state.pos) - 4. * ag.state.vel)
            main = torch.zeros(n, 2)
        else:
            phase = "reacquire"
            helper = reposition_action(env, 1, HELPER_LOCAL)
            main = reposition_action(env, 0, PRIMARY_LOCAL)
        command = torch.stack((main, helper), 1)
        old_pos = torch.stack([a.state.pos.clone() for a in env.world.agents], 1)
        old_vel = torch.stack([a.state.vel.clone() for a in env.world.agents], 1)
        env.step([command[:, 0], command[:, 1]])
        new_pos = torch.stack([a.state.pos for a in env.world.agents], 1)
        new_vel = torch.stack([a.state.vel for a in env.world.agents], 1)
        norms = command.norm(dim=-1)
        norm_max = torch.maximum(norm_max, norms)
        force_integral += norms * env.scenario.agent_force * env.world.dt
        effort += norms.square() * env.world.dt
        work += (command * env.scenario.agent_force * (old_vel + new_vel) / 2).sum(-1).abs() * env.world.dt
        travel += (new_pos - old_pos).norm(dim=-1)
        gaps = surface_gaps(env)
        contacts += gaps.amin(-1) <= .005
        finite &= _finite(env); wall |= _at_wall(env)
        pose = env.scenario.t_pose()
        current_success, _, _ = task_metrics(pose, initial_pose, final_goal)
        newly = current_success & ~ever_success
        first_success_step[newly] = step + 1
        ever_success |= current_success
        if phase == "rotate_first":
            first_success, _, _ = task_metrics(pose, initial_pose, torch.full((n,), .04))
            first_goal_ever |= first_success
            if step == rotation_steps - 1:
                first_goal_at_boundary = first_success.clone()
        if composed and step == rotation_steps + retreat_steps - 1:
            retreat_clear = gaps[:, 1].amin(-1) > .01
        if phase == "reacquire":
            reacquired |= retreat_clear & (gaps[:, 1].amin(-1) <= .005)
            if step == phase2_start - 1:
                reacquired_at_boundary = retreat_clear & (gaps[:, 1].amin(-1) <= .005)
                target, _ = contact_reference(env, PRIMARY_LOCAL)
                primary_repositioned = (env.world.agents[0].state.pos - target).norm(dim=1) < .02
                anchor = effective_anchor(env, support_names)
        if phase.startswith("rotate") and (local_step + 1) % MACRO_STEPS == 0:
            response = pose - macro_start_pose
            response[:, 2] = wrap(response[:, 2])
            prediction_errors.append(macro_prediction - response)
            prediction_valid.append(is_wm.clone())
        trajectories.append(physical_state(env).clone())
        actions.append(command.clone()); anchors.append(anchor.clone()); phases.append(phase)
    terminal_success, angle_error, translation = task_metrics(env.scenario.t_pose(), initial_pose, final_goal)
    if composed:
        task_success = terminal_success & first_goal_at_boundary & retreat_clear & reacquired_at_boundary & primary_repositioned
    else:
        task_success = terminal_success
    steps_value = torch.as_tensor(env.steps).reshape(-1)
    if not torch.all(steps_value == horizon):
        raise AssertionError("Environment history was reset or step accounting differs")
    if not finite.all() or norm_max.max() > 1 + 1e-6:
        raise AssertionError("Nonfinite or unbounded-force execution")
    if before_hash != _state_dict_digest(checkpoints):
        raise AssertionError("Frozen model weights changed during transfer")
    errors = torch.stack(prediction_errors, 1) if prediction_errors else torch.empty(n, 0, 3)
    valid = torch.stack(prediction_valid, 1) if prediction_valid else torch.empty(n, 0, dtype=torch.bool)
    for i, row in enumerate(rows):
        row.update(terminal_success=bool(terminal_success[i]), task_success=bool(task_success[i]),
                   ever_final_goal_success=bool(ever_success[i]), first_success_step=int(first_success_step[i]),
                   final_angle_error_rad=float(angle_error[i]), final_bar_translation_units=float(translation[i]),
                   first_goal_success_at_phase_boundary=bool(first_goal_at_boundary[i]) if composed else None,
                   first_goal_ever_success=bool(first_goal_ever[i]) if composed else None,
                   retreated_clear=bool(retreat_clear[i]) if composed else None,
                   contact_reacquired=bool(reacquired_at_boundary[i]) if composed else None,
                   ever_contact_reacquired=bool(reacquired[i]) if composed else None,
                   primary_repositioned=bool(primary_repositioned[i]) if composed else None,
                   total_steps=horizon, simulated_time=horizon * env.world.dt,
                   maximum_action_norm_by_agent=norm_max[i].tolist(),
                   norm_force_integral_by_agent=force_integral[i].tolist(),
                   squared_control_effort_by_agent=effort[i].tolist(),
                   absolute_actuator_work_proxy_by_agent=work[i].tolist(),
                   agent_travel_units=travel[i].tolist(), contact_fraction_by_agent=(contacts[i] / horizon).tolist(),
                   finite=bool(finite[i]), wall=bool(wall[i]))
        if is_wm[i]:
            e = errors[i, valid[i]]
            row["frozen_macro_prediction_translation_rmse_units"] = float(e[:, :2].square().sum(1).mean().sqrt())
            row["frozen_macro_prediction_angle_rmse_rad"] = float(e[:, 2].square().mean().sqrt())
    raw = {"rows": rows, "initial_pose": initial_pose, "trajectory": torch.stack(trajectories, 1),
           "actions": torch.stack(actions, 1), "helper_anchor": torch.stack(anchors, 1),
           "phases": phases, "selected_macros": selected_macros, "model_predictions": predictions,
           "macro_prediction_errors": errors, "macro_prediction_valid": valid,
           "final_env_steps": steps_value, "frozen_model_weights_sha256": before_hash}
    return raw


def summarize(rows):
    summaries = []
    for task, support, planner, substeps in sorted(set((r["task"], r["support"], r["planner"], r["substeps"]) for r in rows)):
        picked = [r for r in rows if (r["task"], r["support"], r["planner"], r["substeps"]) == (task, support, planner, substeps)]
        n = len(picked)
        entry = dict(task=task, support=support, planner=planner, substeps=substeps, episodes=n,
                     success_count=sum(r["task_success"] for r in picked),
                     terminal_goal_success_count=sum(r["terminal_success"] for r in picked),
                     ever_final_goal_success_count=sum(r["ever_final_goal_success"] for r in picked),
                     mean_final_angle_error_rad=sum(r["final_angle_error_rad"] for r in picked) / n,
                     mean_bar_translation_units=sum(r["final_bar_translation_units"] for r in picked) / n,
                     mean_primary_force_integral=sum(r["norm_force_integral_by_agent"][0] for r in picked) / n,
                     mean_helper_force_integral=sum(r["norm_force_integral_by_agent"][1] for r in picked) / n,
                     helper_contact_fraction=sum(r["contact_fraction_by_agent"][1] for r in picked) / n,
                     finite_all=all(r["finite"] for r in picked), wall_count=sum(r["wall"] for r in picked))
        if task == "rotate_retreat_rebrace_rotate":
            entry["gate_counts"] = {gate: sum(bool(r[gate]) for r in picked) for gate in
                                   ("first_goal_success_at_phase_boundary", "retreated_clear", "contact_reacquired", "primary_repositioned", "terminal_success")}
        summaries.append(entry)
    return summaries


def load_checkpoints(core_path, preload_path):
    checkpoints, provenance, skipped = {}, {}, []
    for path, supports in ((core_path, ("passive", "world_anchor")), (preload_path, ("preloaded_anchor",))):
        if not path.exists():
            skipped.append({"supports": list(supports), "reason": f"Missing frozen model artifact: {path}"})
            continue
        artifact = torch.load(path, map_location="cpu")
        for support in supports:
            model_mode = "world_anchor" if support == "preloaded_anchor" else support
            key = f"{model_mode}/coarse_body_small/seed0"
            checkpoints[support] = artifact["checkpoints"][key]
        provenance[str(path)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "model_training_provenance": artifact.get("provenance", {})}
    return checkpoints, provenance, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("data/novelty/abstraction_models.pt"))
    parser.add_argument("--preload-models", type=Path, default=Path("data/novelty/abstraction_models_preload.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/novelty/abstraction_transfer"))
    parser.add_argument("--frictions", type=float, nargs="+", default=[.75, 1.5, 2.25])
    parser.add_argument("--substeps", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--supports", choices=SUPPORTS, nargs="+", default=list(SUPPORTS))
    parser.add_argument("--skip-composition", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    checkpoints, model_provenance, skipped = load_checkpoints(args.models, args.preload_models)
    supports = [s for s in args.supports if s in checkpoints]
    if not supports:
        parser.error("No requested support has a frozen model")
    groups = []
    for friction, substeps in itertools.product(args.frictions, args.substeps):
        for composed in ([False] if args.skip_composition else [False, True]):
            group = run_group(friction=friction, substeps=substeps, checkpoints=checkpoints,
                              supports=supports, composed=composed)
            groups.append(group)
    rows = [row for group in groups for row in group["rows"]]
    sources = [Path(__file__), Path(__file__).with_name("physical_abstraction.py"),
               Path(__file__).with_name("abstraction_models.py"), Path(__file__).with_name("scenario.py")]
    report = {
        "schema_version": 1, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "frozen_protocol": {"macro_steps": MACRO_STEPS, "candidate_count": 10, "angle_tolerance_rad": ANGLE_TOLERANCE,
                            "translation_tolerance_units": TRANSLATION_TOLERANCE, "model_seed": 0,
                            "primary_feedback_gains": {"angle": 8., "angular_velocity": .8},
                            "rotation_goals_rad": [.04, .08, .12], "rotation_budget_steps": 60,
                            "composition": {"first_goal_rad": .04, "final_goal_rad": .08,
                                            "rotate_steps_each": 60, "retreat_steps": 10, "reacquire_steps": 20}},
        "model_artifacts": model_provenance, "skipped": skipped,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "elapsed_seconds": time.perf_counter() - started, "episode_count": len(rows),
        "summary": summarize(rows), "rows": rows,
        "limitations": [
            "These are newly defined local angular microtasks with a bar-translation tolerance, not the original full PushT task.",
            "Controllers and seed0 body-frame models are frozen; no results were used for controller/model tuning. Episode data are evaluation only.",
            "The helper has zero learned parameters and no goal/planner input; body-frame geometric priors and additional feedback sensing are explicit.",
            "Reset-only model training does not cover most continuation or repositioned states. This intentionally tests that unsupported transfer boundary.",
            "Both primary planners update every five steps. Analytic feedback uses angular velocity, unlike the chosen coarse model input; there is no claim of equal information or computation.",
            "zero_push applies no primary force during rotation; successes in that control can be helper-driven and are not evidence of primary action authority. It shares explicit physical reposition phases in composition.",
            "Passive support means zero helper force during rotation; the composition's explicit retreat/reacquire phases move that same body with bounded PD actions.",
            "Composition completion requires first-goal boundary success, physical retreat, contact return, primary reposition, and final-goal success; terminal angle alone is not composition success.",
            "All lengths are uncalibrated scene units. Integrated force, squared control effort and absolute actuator work are separate proxies, not a certified energy measure.",
        ],
    }
    for path, entry in model_provenance.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != entry["sha256"]:
            raise AssertionError("Model artifact changed while transfer was running")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".pt")
    torch.save({"groups": groups, "model_artifacts": model_provenance}, raw_path)
    torch.load(raw_path, map_location="cpu")
    report["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episode_count": len(rows), "elapsed_seconds": report["elapsed_seconds"],
                      "skipped": skipped, "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
