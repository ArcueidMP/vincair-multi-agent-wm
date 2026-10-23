"""Numerical refinement audit of finite-force physical-support primitives.

The timestep, physics, initial complete state, primary programs and feedback
controller are fixed while only the integration substeps change. Two separate
comparisons are made: the same feedback law in every integrator, and exact replay
of the finest integrator's *joint* action sequence. The latter removes controller
response differences. No response model is queried, so a learned ``substeps``
input cannot confound this physics audit.

Tolerances are chosen before running: maximum XY discrepancy .001 scene units,
maximum wrapped angle discrepancy .003 rad, and at least 95% agreement on fixed
local pose-goal flags. Passing is a finite-grid observation, never a statement
that VMAS's friction/contact model matches real static friction.
"""
from dataclasses import dataclass
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.boundary import _at_wall, _finite, _rotate, surface_gaps
from vmas_pusht.physical_abstraction import helper_action, physical_state, setup_scene
from vmas_pusht.scenario import make, wrap
from vmas_pusht.snapshot import SimulationSnapshot, _configuration, capture


@dataclass(frozen=True)
class Tolerances:
    max_xy: float = .001
    max_angle: float = .003
    goal_agreement: float = .95
    goal_xy: float = .05
    goal_angle: float = .015


DEFAULT_MODES = ("passive", "world_anchor", "preloaded_anchor")


def _without_resolution(physics):
    """Remove exactly the three documented resolution-dependent fields."""
    result = list(physics)
    result[1] = tuple((key, value) for key, value in physics[1] if key != "substeps")
    result[2] = tuple((key, value) for key, value in physics[2]
                      if key not in ("_substeps", "_sub_dt"))
    return tuple(result)


def _validate_resolution(physics):
    scenario, world = dict(physics[1]), dict(physics[2])
    count = world["_substeps"]
    if count < 1 or scenario["substeps"] != count or not math.isclose(
            world["_sub_dt"], world["_dt"] / count, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Inconsistent substep count, physical dt, or integration dt")


def restore_across_substeps(env, snapshot: SimulationSnapshot):
    """Copy full physical and bookkeeping state after an explicit config audit.

    Unlike ordinary snapshot.restore, this audit-only operation allows a change
    in resolution. It does not rewrite the snapshot's physics signature. Every
    other configuration value, entity geometry, joint and task must match. Batch
    companions/order are retained, so batch-dependent collision branches cannot
    masquerade as a resolution effect.
    """
    target = _configuration(env)
    _validate_resolution(snapshot.physics)
    _validate_resolution(target)
    if _without_resolution(target) != _without_resolution(snapshot.physics):
        raise ValueError("Cross-resolution initialization permits only substeps/sub_dt to differ")
    if env.num_envs != snapshot.batch_size:
        raise ValueError("Cross-resolution audit requires identical batch size and row order")
    if [e.name for e in env.world.entities] != [e.name for e in snapshot.entities]:
        raise ValueError("Cross-resolution initialization requires identical entity order")
    # All validation precedes mutation. Clone all fields, including the independent
    # stem's state; compact-state reconstruction would invalidate this experiment.
    for entity, saved in zip(env.world.entities, snapshot.entities):
        for field, value in saved.state.items():
            if value is None:
                setattr(entity.state, "_" + field, None)
            else:
                setattr(entity.state, field, value.to(env.device).clone())
        if saved.action is not None:
            for field, value in saved.action.items():
                if value is None:
                    setattr(entity.action, "_" + field, None)
                else:
                    setattr(entity.action, field, value.to(env.device).clone())
    for field, value in snapshot.scenario.items():
        if value is None:
            if hasattr(env.scenario, field):
                delattr(env.scenario, field)
        else:
            setattr(env.scenario, field, value.to(env.device).clone())
    env.steps = snapshot.steps.to(env.device).clone()
    checked = capture(env)
    for original, copied in zip(snapshot.entities, checked.entities):
        for name, value in original.state.items():
            if value is not None and not torch.equal(value.cpu(), copied.state[name].cpu()):
                raise AssertionError("Complete state copy was not exact")


def default_programs():
    result = [dict(direction_deg=d, strength=s, pulse_steps=20)
              for d, s in itertools.product((-35., -20., 20., 35.), (.3, .6, .9))]
    result.append(dict(direction_deg=0., strength=0., pulse_steps=20))
    return result


def default_nuisances():
    return ((0., 0., 0.), (.03, 0., .2), (-.03, 0., -.2),
            (0., .03, 0.), (0., -.03, 0.))


def make_rows(angles=(0., 30.), nuisances=None, programs=None, modes=DEFAULT_MODES):
    nuisances = default_nuisances() if nuisances is None else nuisances
    programs = default_programs() if programs is None else programs
    return [dict(angle_deg=angle, hidden_id=hidden_id, vx=v[0], vy=v[1], omega=v[2],
                 program_id=program_id, mode=mode, **program)
            for angle, (hidden_id, v), (program_id, program), mode in itertools.product(
                angles, enumerate(nuisances), enumerate(programs), modes)]


def _weld_residual(trajectory):
    n, t, _ = trajectory.shape
    bar_pos, bar_rot = trajectory[..., 12:14], trajectory[..., 16]
    stem_pos, stem_rot = trajectory[..., 18:20], trajectory[..., 22]
    bar_local = torch.tensor([0., -.05]).expand(n * t, 2)
    stem_local = torch.tensor([0., .15]).expand(n * t, 2)
    bar_joint = bar_pos + _rotate(bar_local, bar_rot.flatten()).reshape(n, t, 2)
    stem_joint = stem_pos + _rotate(stem_local, stem_rot.flatten()).reshape(n, t, 2)
    return (bar_joint - stem_joint).norm(dim=-1), wrap(bar_rot - stem_rot).abs()


@torch.no_grad()
def rollout(env, snapshot, rows, anchor, *, horizon=60, exact_actions=None):
    """Record at environment times 0,dt,...,H*dt (not integration substeps)."""
    restore_across_substeps(env, snapshot)
    n = env.num_envs
    if len(rows) != n or horizon < 1:
        raise ValueError("Rows and horizon do not match environment")
    if exact_actions is not None and exact_actions.shape != (n, horizon, 2, 2):
        raise ValueError("Joint replay actions must have shape (batch,horizon,2,2)")
    angles = torch.tensor([math.radians(row["angle_deg"]) for row in rows])
    directions = angles + torch.tensor([math.radians(row["direction_deg"]) for row in rows])
    strengths = torch.tensor([row["strength"] for row in rows])
    primary = torch.stack((directions.sin(), -directions.cos()), -1) * strengths[:, None]
    pulse = torch.tensor([row["pulse_steps"] for row in rows])
    indices = {mode: torch.tensor([i for i, row in enumerate(rows) if row["mode"] == mode])
               for mode in dict.fromkeys(row["mode"] for row in rows)}
    trajectories, commands = [physical_state(env).clone()], []
    gaps = [surface_gaps(env).clone()]
    from vmas_pusht.rebrace import helper_contact_forces
    normal_forces = [helper_contact_forces(env).norm(dim=-1).clone()]
    finite, walls = _finite(env), _at_wall(env)
    for step in range(horizon):
        if exact_actions is None:
            helper = torch.empty(n, 2)
            for mode, ix in indices.items():
                helper[ix] = helper_action(env, mode, anchor, reference_angle=angles)[ix]
            action = torch.stack((primary * (step < pulse)[:, None], helper), dim=1)
        else:
            action = exact_actions[:, step].clone()
        if not bool(torch.isfinite(action).all()) or float(action.norm(dim=-1).max()) > 1.000001:
            raise AssertionError("Commands must be finite and norm bounded")
        env.step([action[:, 0], action[:, 1]])
        commands.append(action.clone())
        trajectories.append(physical_state(env).clone())
        gaps.append(surface_gaps(env).clone())
        normal_forces.append(helper_contact_forces(env).norm(dim=-1).clone())
        finite &= _finite(env)
        walls |= _at_wall(env)
    if not bool(finite.all()):
        raise AssertionError("Nonfinite physical trajectory")
    trajectory = torch.stack(trajectories, 1)
    gap = torch.stack(gaps, 1)
    weld_xy, weld_angle = _weld_residual(trajectory)
    actions = torch.stack(commands, 1)
    contact = gap <= .005
    return dict(trajectory=trajectory, actions=actions, gaps=gap,
                contact=contact, geometric_proximity=contact,
                helper_normal_force_norms=torch.stack(normal_forces, 1),
                finite=finite, touched_wall=walls,
                weld_xy=weld_xy, weld_angle=weld_angle,
                time=torch.arange(horizon + 1) * env.world.dt,
                dt=float(env.world.dt), substeps=env.scenario.substeps)


def goal_flags(trajectory, *, tolerance=Tolerances(), goals=(-.1, -.06, -.03, .03, .06, .1)):
    """Fixed independent local rotation goals, relative to the common start."""
    if trajectory.ndim != 3 or trajectory.shape[-1] != 24:
        raise ValueError("Expected complete trajectories of shape (B,T,24)")
    translation = (trajectory[..., 12:14] - trajectory[:, :1, 12:14]).norm(dim=-1)
    angle = wrap(trajectory[..., 16] - trajectory[:, :1, 16])
    target = torch.tensor(goals, dtype=trajectory.dtype, device=trajectory.device)
    return ((translation < tolerance.goal_xy)[..., None]
            & (wrap(angle[..., None] - target).abs() < tolerance.goal_angle))


def compare_trajectories(candidate, reference, *, tolerance=Tolerances(), horizon=None):
    """All-time errors, wrapped orientation and meaningful fixed goal agreement."""
    if candidate["trajectory"].shape != reference["trajectory"].shape:
        raise ValueError("Trajectory shapes must match; no silent trimming")
    if not torch.equal(candidate["time"], reference["time"]):
        raise ValueError("Trajectory timestamps must match exactly")
    if not torch.equal(candidate["trajectory"][:, 0], reference["trajectory"][:, 0]):
        raise ValueError("Initial complete physical states must be exactly identical")
    end = candidate["trajectory"].shape[1] if horizon is None else horizon + 1
    if end < 2 or end > candidate["trajectory"].shape[1]:
        raise ValueError("Comparison horizon is outside saved timestamps")
    c, r = candidate["trajectory"][:, :end], reference["trajectory"][:, :end]
    xy = (c[..., 12:14] - r[..., 12:14]).norm(dim=-1)
    angle = wrap(c[..., 16] - r[..., 16]).abs()
    action = (candidate["actions"][:, :end-1] - reference["actions"][:, :end-1]).norm(dim=-1)
    cg, rg = goal_flags(c, tolerance=tolerance), goal_flags(r, tolerance=tolerance)
    goal_agreement = float((cg[:, 1:] == rg[:, 1:]).float().mean())
    either_goal = cg[:, 1:] | rg[:, 1:]
    union_goal_cases = int(either_goal.sum())
    active_goal_agreement = (float((cg[:, 1:][either_goal] == rg[:, 1:][either_goal]).float().mean())
                             if union_goal_cases else None)
    contact_agreement = float((candidate["contact"][:, :end] == reference["contact"][:, :end]).float().mean())
    reference_angle = wrap(r[..., 16] - r[:, :1, 16]).abs()
    output = dict(horizon_steps=end-1, physical_seconds=float(reference["time"][end-1]),
                  candidate_substeps=candidate["substeps"], reference_substeps=reference["substeps"],
                  max_xy_units=float(xy.max()), rms_xy_units=float(xy.square().mean().sqrt()),
                  terminal_rms_xy_units=float(xy[:, -1].square().mean().sqrt()),
                  max_angle_rad=float(angle.max()), rms_angle_rad=float(angle.square().mean().sqrt()),
                  terminal_rms_angle_rad=float(angle[:, -1].square().mean().sqrt()),
                  max_joint_action_difference_norm=float(action.max()),
                  rms_joint_action_difference_norm=float(action.square().mean().sqrt()),
                  geometric_proximity_flag_agreement=contact_agreement, goal_flag_agreement=goal_agreement,
                  goal_union_count=union_goal_cases, goal_agreement_on_union=active_goal_agreement,
                  terminal_goal_flag_agreement=float((cg[:, -1] == rg[:, -1]).float().mean()),
                  reference_max_absolute_rotation_rad=float(reference_angle.max()),
                  reference_mean_peak_rotation_rad=float(reference_angle.max(dim=1).values.mean()),
                  reference_cases_rotation_over_goal_tolerance=int((reference_angle.max(dim=1).values > tolerance.goal_angle).sum()),
                  cases=len(c),
                  max_xy_error_over_tolerance=float(xy.max()) / tolerance.max_xy,
                  max_angle_error_over_tolerance=float(angle.max()) / tolerance.max_angle)
    output["passes_preregistered_tolerances"] = (output["max_xy_units"] <= tolerance.max_xy
            and output["max_angle_rad"] <= tolerance.max_angle
            and goal_agreement >= tolerance.goal_agreement)
    return output


def rollout_summary(run, rows):
    trajectory, action = run["trajectory"], run["actions"]
    angle = wrap(trajectory[..., 16] - trajectory[:, :1, 16])
    displacement = (trajectory[..., 12:14] - trajectory[:, :1, 12:14]).norm(dim=-1)
    groups = []
    for mode in dict.fromkeys(row["mode"] for row in rows):
        ix = torch.tensor([i for i, row in enumerate(rows) if row["mode"] == mode])
        contacts = run["contact"][ix, 1:, 1]
        # The zero-action branch has the same angle/nuisance/helper controller.
        zero = {(row["angle_deg"], row["hidden_id"]): i for i, row in enumerate(rows)
                if row["mode"] == mode and row["strength"] == 0.}
        moving = [i for i in ix.tolist() if rows[i]["strength"] > 0]
        authority = None
        if zero and moving:
            zi = torch.tensor([zero[(rows[i]["angle_deg"], rows[i]["hidden_id"])] for i in moving])
            signed = wrap(trajectory[moving, :, 16] - trajectory[zi, :, 16])
            authority = dict(mean_peak_abs_rotation_vs_zero_rad=float(signed.abs().max(1).values.mean()),
                             max_abs_rotation_vs_zero_rad=float(signed.abs().max()),
                             moving_cases=len(moving))
        groups.append(dict(mode=mode, cases=len(ix),
                           mean_peak_abs_rotation_rad=float(angle[ix].abs().max(1).values.mean()),
                           max_abs_rotation_rad=float(angle[ix].abs().max()),
                           max_translation_units=float(displacement[ix].max()),
                           helper_any_geometric_proximity_fraction=float(contacts.any(-1).float().mean()),
                           helper_double_geometric_proximity_fraction=float(contacts.all(-1).float().mean()),
                           helper_dual_normal_force_fraction=(
                               float((run["helper_normal_force_norms"][ix, 1:] >= .001).all(-1).float().mean())
                               if "helper_normal_force_norms" in run else None),
                           max_weld_xy_units=float(run["weld_xy"][ix].max()),
                           max_weld_angle_rad=float(run["weld_angle"][ix].max()),
                           max_action_norm=float(action[ix].norm(dim=-1).max()),
                           mean_norm_impulse_by_agent=(action[ix].norm(dim=-1).sum(1).mean(0) * run["dt"]).tolist(),
                           primary_authority_vs_matched_zero=authority,
                           wall_cases=int(run["touched_wall"][ix].sum())))
    return dict(substeps=run["substeps"], helper_modes=groups)


def run_group(*, friction=1.5, substeps=(10, 20, 40, 80, 160), rows=None, horizon=60,
              compare_horizons=(30, 60), tolerance=Tolerances()):
    steps = sorted(set(substeps))
    if not steps or steps[0] < 1:
        raise ValueError("Substeps must be positive")
    rows = make_rows() if rows is None else rows
    n, finest = len(rows), steps[-1]
    config = dict(num_envs=n, friction_ratio=friction, device="cpu", seed=42)
    source = make(substeps=finest, **config)
    angles = torch.tensor([math.radians(row["angle_deg"]) for row in rows])
    anchor = setup_scene(source, angle=angles, hidden_v=[[row["vx"], row["vy"]] for row in rows],
                         hidden_omega=[row["omega"] for row in rows])
    snapshot = capture(source)
    feedback = {}
    feedback[finest] = rollout(source, snapshot, rows, anchor, horizon=horizon)
    replay = {finest: feedback[finest]}
    for ss in steps[:-1]:
        env = make(substeps=ss, **config)
        feedback[ss] = rollout(env, snapshot, rows, anchor, horizon=horizon)
        replay[ss] = rollout(env, snapshot, rows, anchor, horizon=horizon,
                             exact_actions=feedback[finest]["actions"])
    comparisons = []
    for kind, runs in (("fixed_feedback_law", feedback), ("identical_joint_action_replay", replay)):
        for left, right in zip(steps[:-1], steps[1:]):
            for end in compare_horizons:
                item = compare_trajectories(runs[left], runs[right], horizon=end, tolerance=tolerance)
                comparisons.append(dict(kind=kind, pair="adjacent", **item))
        for ss in steps[:-1]:
            for end in compare_horizons:
                item = compare_trajectories(runs[ss], runs[finest], horizon=end, tolerance=tolerance)
                comparisons.append(dict(kind=kind, pair="vs_finest", **item))
    summary = dict(friction_ratio=friction, cases_per_integrator=n, horizon=horizon,
                   substeps=steps, finest_joint_action_source=finest, initial_state_identical=True,
                   dt=float(source.world.dt), comparisons=comparisons,
                   feedback_runs=[rollout_summary(feedback[ss], rows) for ss in steps],
                   replay_runs=[rollout_summary(replay[ss], rows) for ss in steps])
    raw = dict(rows=rows, snapshot=snapshot, anchor=anchor, feedback=feedback, replay=replay)
    return summary, raw


def select_representative_rows(raw_group, low=80, high=160):
    """Select each helper's worst XY/angle errors and highest active response.

    This is explicitly a diagnostic followup selected on an earlier audit,
    not an independent holdout or evidence that the untested full grid converges.
    Exact matched no-primary-action controls are included for every chosen case.
    The selected batch is rerun at *both* resolutions so changed companions do
    not get compared against the original larger batch.
    """
    rows = raw_group["rows"]
    c, r = (raw_group["feedback"][ss]["trajectory"] for ss in (low, high))
    xy = (c[..., 12:14] - r[..., 12:14]).norm(dim=-1).max(1).values
    angle = wrap(c[..., 16] - r[..., 16]).abs().max(1).values
    activity = wrap(r[..., 16] - r[:, :1, 16]).abs().max(1).values
    selected, reasons = set(), []
    for mode in dict.fromkeys(row["mode"] for row in rows):
        indices = torch.tensor([i for i, row in enumerate(rows) if row["mode"] == mode])
        for name, metric in (("worst_xy_error", xy), ("worst_angle_error", angle),
                             ("largest_finest_absolute_response", activity)):
            index = int(indices[metric[indices].argmax()])
            selected.add(index)
            reasons.append(dict(source_index=index, mode=mode, criterion=name,
                                value=float(metric[index])))
    for index in list(selected):
        row = rows[index]
        zero = next(i for i, candidate in enumerate(rows)
                    if candidate["mode"] == row["mode"] and candidate["angle_deg"] == row["angle_deg"]
                    and candidate["hidden_id"] == row["hidden_id"] and candidate["strength"] == 0.)
        selected.add(zero)
    selected = sorted(selected)
    return [rows[i] for i in selected], dict(source_low_substeps=low, source_high_substeps=high,
                    selected_source_indices=selected, selection_reasons=reasons,
                    all_selected_rows_rerun_at_all_requested_substeps=True)


@torch.no_grad()
def recovery_rollout(env, snapshot, rows, *, max_recovery=80, post_steps=20):
    """Frozen recovery, then 20 steps of fixed zero/.3 primary intervention.

    Exactly the same policy/phase guards run at each numerical resolution.
    Success does not stop simulation: every trace covers max_recovery+post_steps
    environment steps. A failed case receives bounded primary holding and zero
    helper force after timeout. Readiness is never inferred from a past touch.
    """
    from vmas_pusht.rebrace import (RebraceController, helper_contact_forces,
        main_holding_action, readiness_metrics, support_action)
    restore_across_substeps(env, snapshot)
    n, horizon = len(rows), max_recovery + post_steps
    supports = [row["mode"] for row in rows]
    controller = RebraceController(env, supports, max_steps=max_recovery)
    first_ready = torch.full((n,), -1, dtype=torch.long)
    post_count = torch.zeros(n, dtype=torch.long)
    post_force_count = torch.zeros(n, dtype=torch.long)
    launch_angle = env.scenario.bar.state.rot[:, 0].clone()
    handoff_force = torch.tensor([row["handoff_force"] for row in rows])
    poses = [physical_state(env).clone()]
    gaps = [surface_gaps(env).clone()]
    normals = [helper_contact_forces(env).norm(dim=-1).clone()]
    commands, metrics, readiness, phases, anchors = [], [], [], [], []
    finite, walls = _finite(env), _at_wall(env)
    for step in range(horizon):
        recovering = (first_ready < 0) & ~controller.failed
        post = (first_ready >= 0) & (post_count < post_steps)
        command = controller.step()
        launched_force = torch.stack((launch_angle.sin(), -launch_angle.cos()), -1) * handoff_force[:, None]
        command[post, 0] = launched_force[post]
        command[~recovering, 1] = support_action(env, controller.anchor, supports)[~recovering]
        command[~recovering & ~post, 0] = main_holding_action(env)[~recovering & ~post]
        command[controller.failed, 1] = 0.
        if command.norm(dim=-1).max() > 1.000001:
            raise AssertionError("Recovery force exceeds declared unit norm")
        env.step([command[:, 0], command[:, 1]])
        status = controller.update(active_mask=recovering)
        now = readiness_metrics(env, controller.anchor, supports)
        newly = recovering & status["ready"]
        first_ready[newly] = step + 1
        launch_angle[newly] = env.scenario.bar.state.rot[newly, 0]
        post_count[post] += 1
        force_contact = (now["helper_normal_forces"] >= controller.config.minimum_contact_force).all(-1)
        post_force_count[post] += force_contact[post]
        commands.append(command.clone()); poses.append(physical_state(env).clone())
        gaps.append(surface_gaps(env).clone()); normals.append(now["helper_normal_forces"].clone())
        metrics.append({key: value.clone() for key, value in now.items()})
        readiness.append(now["instantaneous_ready"].clone())
        phases.append(controller.phase.clone()); anchors.append(controller.anchor.clone())
        finite &= _finite(env); walls |= _at_wall(env)
    if not finite.all():
        raise AssertionError("Recovery produced nonfinite physics")
    trajectory = torch.stack(poses, 1)
    gap = torch.stack(gaps, 1)
    weld_xy, weld_angle = _weld_residual(trajectory)
    return dict(trajectory=trajectory, actions=torch.stack(commands, 1), gaps=gap,
                contact=gap <= .005, geometric_proximity=gap <= .005,
                helper_normal_force_norms=torch.stack(normals, 1),
                finite=finite, touched_wall=walls, weld_xy=weld_xy, weld_angle=weld_angle,
                time=torch.arange(horizon + 1) * env.world.dt, dt=float(env.world.dt),
                substeps=env.scenario.substeps, first_ready_step=first_ready,
                recovery_failed=controller.failed.clone(), post_handoff_steps=post_count,
                post_handoff_dual_force_steps=post_force_count,
                full_20_step_dual_force=(post_count == post_steps) & (post_force_count == post_steps),
                instantaneous_ready=torch.stack(readiness, 1), phase=torch.stack(phases, 1),
                anchor=torch.stack(anchors, 1), controller_configuration=controller.configuration(),
                metrics={key: torch.stack([value[key] for value in metrics], 1) for key in metrics[0]})


def recovery_summary(run, rows):
    groups = []
    for mode in dict.fromkeys(row["mode"] for row in rows):
        ix = torch.tensor([i for i, row in enumerate(rows) if row["mode"] == mode])
        first = run["first_ready_step"][ix]
        ready = first >= 0
        groups.append(dict(mode=mode, cases=len(ix), ready_cases=int(ready.sum()),
                           first_ready_steps=first.tolist(),
                           mean_first_ready_step=float(first[ready].float().mean()) if ready.any() else None,
                           timeout_cases=int(run["recovery_failed"][ix].sum()),
                           stable_dual_contact_20_cases=int(run["full_20_step_dual_force"][ix].sum()),
                           post_handoff_steps=run["post_handoff_steps"][ix].tolist(),
                           post_handoff_dual_force_steps=run["post_handoff_dual_force_steps"][ix].tolist(),
                           max_action_norm=float(run["actions"][ix].norm(dim=-1).max()),
                           wall_cases=int(run["touched_wall"][ix].sum())))
    return dict(substeps=run["substeps"], helper_modes=groups)


def recovery_group(*, friction=1.5, substeps=(20, 40, 80, 160), compare_horizons=(80, 100),
                   tolerance=Tolerances(), history_substeps=160):
    """One common, legally evolved post-retreat full state per paired trial."""
    from vmas_pusht.rebrace import main_holding_action
    steps = sorted(set(substeps))
    base_rows = make_rows(angles=(0., 30.), nuisances=((0., 0., 0.), (.03, 0., .2)),
                         programs=[dict(direction_deg=20., strength=.6, pulse_steps=20)])
    rows = [dict(row, handoff_force=force) for row in base_rows for force in (0., .3)]
    n, finest, horizon = len(rows), steps[-1], 100
    config = dict(num_envs=n, friction_ratio=friction, device="cpu", seed=42)
    source = make(substeps=history_substeps, **config)
    angles = torch.tensor([math.radians(row["angle_deg"]) for row in rows])
    anchor = setup_scene(source, angle=angles, hidden_v=[[row["vx"], row["vy"]] for row in rows],
                         hidden_omega=[row["omega"] for row in rows])
    initial_snapshot = capture(source)
    warm_direction = angles + math.radians(20.)
    primary = .6 * torch.stack((warm_direction.sin(), -warm_direction.cos()), -1)
    history, history_actions = [physical_state(source).clone()], []
    indices = {mode: torch.tensor([i for i, row in enumerate(rows) if row["mode"] == mode])
               for mode in DEFAULT_MODES}
    for step in range(20):
        helper = torch.empty(n, 2)
        for mode, ix in indices.items():
            helper[ix] = helper_action(source, mode, anchor, reference_angle=angles)[ix]
        command = torch.stack((primary, helper), 1)
        source.step([command[:, 0], command[:, 1]])
        history.append(physical_state(source).clone()); history_actions.append(command.clone())
    retreat_target = source.world.agents[1].state.pos.clone() + _rotate(
        torch.tensor([.15, -.15]).expand(n, 2), source.scenario.bar.state.rot[:, 0])
    from vmas_pusht.physical_abstraction import norm_bound
    for step in range(20):
        helper = source.world.agents[1]
        retreat = norm_bound(30. * (retreat_target - helper.state.pos) - 4. * helper.state.vel)
        command = torch.stack((main_holding_action(source), retreat), 1)
        source.step([command[:, 0], command[:, 1]])
        history.append(physical_state(source).clone()); history_actions.append(command.clone())
    snapshot = capture(source)
    initially_clear = surface_gaps(source)[:, 1].amin(-1) > .01
    feedback, replay = {}, {}
    for ss in reversed(steps):
        env = make(substeps=ss, **config)
        feedback[ss] = recovery_rollout(env, snapshot, rows)
        if ss == finest:
            replay[ss] = feedback[ss]
        else:
            replay[ss] = rollout(env, snapshot, rows, anchor, horizon=horizon,
                                 exact_actions=feedback[finest]["actions"])
    comparisons = []
    for kind, runs in (("fixed_recovery_and_handoff_policy", feedback),
                       ("identical_joint_action_replay", replay)):
        for low, high in zip(steps[:-1], steps[1:]):
            for end in compare_horizons:
                item = compare_trajectories(runs[low], runs[high], horizon=end, tolerance=tolerance)
                if kind == "fixed_recovery_and_handoff_policy":
                    lf, hf = feedback[low]["first_ready_step"], feedback[high]["first_ready_step"]
                    both = (lf >= 0) & (hf >= 0)
                    item.update(readiness_outcome_agreement=float(((lf >= 0) == (hf >= 0)).float().mean()),
                                stable_dual_contact20_outcome_agreement=float((feedback[low]["full_20_step_dual_force"]
                                    == feedback[high]["full_20_step_dual_force"]).float().mean()),
                                common_ready_cases=int(both.sum()),
                                max_first_ready_step_difference=int((lf[both]-hf[both]).abs().max()) if both.any() else None)
                comparisons.append(dict(kind=kind, pair="adjacent", **item))
    summary = dict(friction_ratio=friction, cases_per_integrator=n, horizon=horizon,
                   substeps=steps, history_substeps=history_substeps,
                   history="20 fixed primary push steps, then20 finite-force retreat steps with primary holding",
                   initial_state_identical=True, initial_helper_clear_cases=int(initially_clear.sum()),
                   dt=float(source.world.dt), controller_configuration=feedback[finest]["controller_configuration"],
                   comparisons=comparisons, feedback_runs=[recovery_summary(feedback[ss], rows) for ss in steps])
    raw = dict(rows=rows, initial_snapshot=initial_snapshot, snapshot=snapshot,
               history_trajectory=torch.stack(history, 1), history_actions=torch.stack(history_actions, 1),
               initially_clear=initially_clear, feedback=feedback, replay=replay)
    return summary, raw


@torch.no_grad()
def audit_saved_contacts(report_path, out_path):
    """Evaluate exact VMAS normal forces on saved poses, without integrating.

    This is a read-only metric recalculation. It is not a physical continuation
    from the 24D states, and doesn't alter any original numerical-audit artifact.
    """
    from vmas_pusht.rebrace import helper_contact_forces
    report_path, out_path = Path(report_path), Path(out_path)
    report = json.loads(report_path.read_text())
    raw_path = Path(report["raw_file"])
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != report["raw_sha256"]:
        raise ValueError("Original raw checksum mismatch")
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    output, raw_output = [], []
    for summary, group in zip(report["groups"], raw["groups"]):
        rows, runs, raw_runs = group["rows"], [], {}
        for kind in ("feedback", "replay"):
            for ss, run in group[kind].items():
                trajectory = run["trajectory"]
                n, t, _ = trajectory.shape
                env = make(num_envs=n*t, device="cpu", friction_ratio=summary["friction_ratio"], substeps=ss)
                # Only pose/velocity fields used by this geometry diagnostic are
                # assigned. No simulation is executed and goals cannot matter.
                flat = trajectory.reshape(n*t, 24)
                for index, entity in enumerate((*env.world.agents, env.scenario.bar, env.scenario.stem)):
                    state = flat[:, 6*index:6*index+6]
                    entity.set_pos(state[:, :2].clone(), batch_index=None)
                    entity.set_vel(state[:, 2:4].clone(), batch_index=None)
                    entity.set_rot(state[:, 4:5].clone(), batch_index=None)
                    entity.set_ang_vel(state[:, 5:6].clone(), batch_index=None)
                forces = helper_contact_forces(env).norm(dim=-1).reshape(n, t, 2)
                raw_runs[f"{kind}/{ss}"] = forces
                for mode in DEFAULT_MODES:
                    ix = torch.tensor([i for i, row in enumerate(rows) if row["mode"] == mode])
                    if not len(ix):
                        continue
                    prox = run["gaps"][ix, 1:, 1] <= .005
                    actual = forces[ix, 1:] >= .001
                    runs.append(dict(kind=kind, substeps=ss, mode=mode, cases=len(ix),
                                     helper_geometric_any_proximity_fraction=float(prox.any(-1).float().mean()),
                                     helper_geometric_dual_proximity_fraction=float(prox.all(-1).float().mean()),
                                     helper_any_normal_force_fraction=float(actual.any(-1).float().mean()),
                                     helper_dual_normal_force_fraction=float(actual.all(-1).float().mean())))
        output.append(dict(friction_ratio=summary["friction_ratio"], runs=runs))
        raw_output.append(raw_runs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_out = out_path.with_suffix(".pt")
    torch.save(dict(groups=raw_output), raw_out)
    result = dict(schema_version=1, source_report=str(report_path),
                  source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                  source_raw_sha256=report["raw_sha256"], raw_file=str(raw_out),
                  raw_sha256=hashlib.sha256(raw_out.read_bytes()).hexdigest(),
                  force_threshold=.001, geometric_proximity_threshold=.005,
                  explanation="VMAS LINE_MIN_DIST shell carries finite force at some gaps above .005; proximity is not physical contact.",
                  controller_source_sha256=hashlib.sha256(Path(__file__).with_name("rebrace.py").read_bytes()).hexdigest(),
                  groups=output)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/novelty/abstraction_convergence.json"))
    parser.add_argument("--substeps", nargs="+", type=int, default=[10, 20, 40, 80, 160])
    parser.add_argument("--frictions", nargs="+", type=float, default=[.75, 1.5, 2.25])
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--compare-horizons", nargs="+", type=int, default=[30, 60])
    parser.add_argument("--smoke", action="store_true", help="Small explicitly labeled diagnostic grid")
    parser.add_argument("--recovery", action="store_true", help="Frozen recovery from common post-retreat states")
    parser.add_argument("--representative-from", type=Path,
                        help="Prior JSON: select diagnostic worst/high-activity cases plus zero controls")
    args = parser.parse_args()
    if any(h > args.horizon or h < 1 for h in args.compare_horizons):
        parser.error("comparison horizons must be within the recorded trajectory")
    if args.recovery and (args.representative_from or args.smoke):
        parser.error("recovery uses its own declared common-history grid")
    torch.set_num_threads(1)
    rows = (make_rows(angles=(0.,), nuisances=default_nuisances()[:2],
                      programs=[default_programs()[4], default_programs()[-1]]) if args.smoke else make_rows())
    summaries, groups = [], []
    selection = None
    selected_groups = {}
    if args.representative_from:
        if args.smoke:
            parser.error("representative selection and smoke are distinct protocols")
        previous_report = json.loads(args.representative_from.read_text())
        previous_raw_path = Path(previous_report["raw_file"])
        if hashlib.sha256(previous_raw_path.read_bytes()).hexdigest() != previous_report["raw_sha256"]:
            raise ValueError("Prior raw data checksum mismatch")
        previous_raw = torch.load(previous_raw_path, map_location="cpu", weights_only=False)
        selection = dict(source_report=str(args.representative_from),
                         source_report_sha256=hashlib.sha256(args.representative_from.read_bytes()).hexdigest(),
                         source_raw_sha256=previous_report["raw_sha256"], groups=[])
        for summary, raw in zip(previous_report["groups"], previous_raw["groups"]):
            chosen, specification = select_representative_rows(raw)
            selected_groups[summary["friction_ratio"]] = chosen
            selection["groups"].append(dict(friction_ratio=summary["friction_ratio"], **specification))
    start = time.monotonic()
    for friction in args.frictions:
        if selected_groups:
            if friction not in selected_groups:
                raise ValueError("Requested friction missing from representative source")
            rows = selected_groups[friction]
        if args.recovery:
            summary, raw = recovery_group(friction=friction, substeps=args.substeps,
                                          compare_horizons=args.compare_horizons)
        else:
            summary, raw = run_group(friction=friction, substeps=args.substeps, rows=rows,
                                     horizon=args.horizon, compare_horizons=args.compare_horizons)
        summaries.append(summary); groups.append(raw)
        print(json.dumps(dict(friction=friction, cases=summary["cases_per_integrator"], elapsed_seconds=time.monotonic()-start,
                              finest_comparison=[v for v in summary["comparisons"] if v["pair"] == "adjacent"
                                  and v["reference_substeps"] == max(args.substeps)])), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.out.with_suffix(".pt")
    torch.save(dict(groups=groups), raw_path)
    paths = [Path(__file__), *(Path(__file__).with_name(name) for name in
             ("physical_abstraction.py", "scenario.py", "snapshot.py", "boundary.py", "rebrace.py"))]
    tolerance = Tolerances()
    report = dict(schema_version=1, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  representative_selection=selection,
                  preregistered_tolerances=vars(tolerance), physical_timepoints="all environment steps, including t=0",
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                  versions={name: importlib.metadata.version(name) for name in ("torch", "vmas", "numpy")},
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  raw_file=str(raw_path), raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-start, groups=summaries,
                  limitations=["No response model is queried; learned substeps conditioning is not a numerical convergence test.",
                      "The full grid is a finite set of local rotation programs, not an arbitrary-trajectory error bound.",
                      "Geometric proximity flags use .005; helper normal-force norms separately use the exact VMAS collision law at environment timepoints.",
                      "VMAS velocity-dependent floor friction has no true balancing static friction at zero velocity.",
                      "Goal flags use fixed local pose targets; this audit does not plan or claim full manipulation success."])
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("->", args.out, flush=True)


if __name__ == "__main__":
    main()
