"""Bounded-contact pilot for cooperation as a physical abstraction.

This is a mechanism audit, not a trained-helper result. A fixed family of local
primary pushes is paired with cheap, goal-independent helper controllers. All
helper modes start at identical physical states, and unobserved rigid-body
velocities/friction produce exact aliases in a geometry-only representation.

Initial velocities are explicitly constructed, physically admissible initial
conditions. They are NOT claimed to have been reached by previous trajectories.
All actors remain ordinary VMAS dynamic bodies throughout each rollout: no
teleports, clamps, infinite-mass fixtures, or action norms above one are used.
"""

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch

from vmas_pusht.boundary import _at_wall, _finite, _plain, _rotate, surface_gaps, t_com
from vmas_pusht.scenario import make, wrap


MODES = ("passive", "world_anchor", "track_contact", "damping")
PRELOAD_MODES = ("preloaded_anchor", "preloaded_track")
PRIMARY_LOCAL = (-0.13, 0.101)
HELPER_LOCAL = (0.101, -0.101)
COARSE_FEATURES = ["A_x", "A_y", "B_x", "B_y", "bar_x", "bar_y", "bar_sin", "bar_cos",
                   "stem_x", "stem_y", "stem_sin", "stem_cos", "main_u_x", "main_u_y",
                   "duration_seconds", "substeps", "helper_anchor_x", "helper_anchor_y"]
FULL_EXTRA_FEATURES = [f"{name}_{field}" for name in ("A", "B", "bar", "stem")
                       for field in ("vx", "vy", "omega")] + ["friction_ratio", "rot_friction_ratio"]


def norm_bound(action, bound=1.0):
    """Euclidean norm bound, unlike VMAS's default per-coordinate bound."""
    if bound <= 0:
        raise ValueError("bound must be positive")
    return action / (action.norm(dim=-1, keepdim=True) / bound).clamp_min(1.0)


def physical_state(env):
    """All four movable entities: x,y,vx,vy,angle,angular_velocity (24 floats)."""
    return torch.cat([torch.cat((e.state.pos, e.state.vel, e.state.rot, e.state.ang_vel), 1)
                      for e in (*env.world.agents, env.scenario.bar, env.scenario.stem)], 1)


def coarse_state(env):
    """Geometry only (12D); velocity and physical parameters are deliberately omitted."""
    values = [a.state.pos for a in env.world.agents]
    for part in (env.scenario.bar, env.scenario.stem):
        values += [part.state.pos, part.state.rot.sin(), part.state.rot.cos()]
    return torch.cat(values, 1)


def setup_scene(env, *, angle=0.0, hidden_v=(0.0, 0.0), hidden_omega=0.0,
                helper_error=(0.0, 0.0), primary_local=PRIMARY_LOCAL,
                helper_local=HELPER_LOCAL):
    """Construct a legal touching-near-contact scene, returning the helper anchor.

    ``angle`` and ``hidden_omega`` accept scalar or (B,) values; ``hidden_v`` and
    ``helper_error`` accept (2,) or (B,2). Velocities are in world coordinates.
    ``helper_error`` is a local contact-position perturbation. The bar and stem
    receive rigid-body-consistent velocities. No warmup is used: it would destroy
    exact coarse-state aliasing across nuisance conditions.
    """
    if len(env.world.agents) != 2:
        raise ValueError("This pilot requires two agents")
    B, dev = env.num_envs, env.device
    a = torch.as_tensor(angle, dtype=torch.float32, device=dev).expand(B)
    velocity = torch.as_tensor(hidden_v, dtype=torch.float32, device=dev).expand(B, 2)
    omega = torch.as_tensor(hidden_omega, dtype=torch.float32, device=dev).reshape(-1, 1).expand(B, 1)
    error = torch.as_tensor(helper_error, dtype=torch.float32, device=dev).expand(B, 2)
    pose = torch.cat((torch.zeros(B, 2, device=dev), a[:, None]), 1)
    sc = env.scenario
    sc.place_t(pose)
    sc.place_t(pose, goal=True)
    sc.goal_pose[:] = pose  # Diagnostics do not use reward/success or a task target.
    displacement = sc.stem.state.pos - sc.bar.state.pos
    sc.bar.set_vel(velocity.clone(), batch_index=None)
    sc.stem.set_vel(velocity + omega * torch.stack((-displacement[:, 1], displacement[:, 0]), 1),
                    batch_index=None)
    for part in (sc.bar, sc.stem):
        part.set_ang_vel(omega.clone(), batch_index=None)
    for i, local in enumerate((primary_local, helper_local)):
        loc = torch.as_tensor(local, dtype=torch.float32, device=dev).expand(B, 2)
        if i == 1:
            loc = loc + error
        ag = env.world.agents[i]
        ag.set_pos(_rotate(loc, a), batch_index=None)
        ag.set_vel(torch.zeros(B, 2, device=dev), batch_index=None)
        ag.set_rot(torch.zeros(B, 1, device=dev), batch_index=None)
        ag.set_ang_vel(torch.zeros(B, 1, device=dev), batch_index=None)
        ag.action.u = torch.zeros(B, 2, device=dev)
    sc.prev_err[:] = sc.pose_err()
    return env.world.agents[1].state.pos.clone()


def helper_action(env, mode, anchor, *, kp=30.0, kd=4.0, damping_gain=8.0,
                  helper_local=HELPER_LOCAL, reference_angle=0.0):
    """A finite-force, goal-independent helper command with no primary-action input.

    world_anchor: compliant fixed-location PD baseline, tracking only own state.
    track_contact: PD tracking a body-relative contact point and its velocity.
    damping: own-velocity and T translational-velocity feedback, no target pose.
    passive: zero actuator force; the same dynamic collidable body remains.
    preloaded_anchor / preloaded_track: supplemental controls whose finite PD
    target is inset 0.01 scene units along both corner normals. For preloaded_anchor, pass
    the ORIGINAL initial helper position as anchor and its initial bar angle as
    reference_angle (radians, scalar/B). No body is initially moved inward.

    All controllers have zero learned parameters. Gains are fixed before the
    heldout tests; no task reward or model-prediction loss is used to select them.
    """
    if mode not in MODES + PRELOAD_MODES:
        raise ValueError(f"Unknown helper mode {mode}")
    if kp < 0 or kd < 0 or damping_gain < 0:
        raise ValueError("Feedback gains must be nonnegative")
    ag, bar = env.world.agents[1], env.scenario.bar
    if mode == "preloaded_anchor":
        angle = torch.as_tensor(reference_angle, dtype=ag.state.pos.dtype, device=env.device).expand(env.num_envs)
        anchor = anchor + _rotate(torch.tensor([-.01, .01], device=env.device).expand(env.num_envs, 2), angle)
        mode = "world_anchor"
    elif mode == "preloaded_track":
        helper_local = (helper_local[0] - .01, helper_local[1] + .01)
        mode = "track_contact"
    if mode == "passive":
        return torch.zeros_like(ag.state.pos)
    if mode == "world_anchor":
        force = kp * (anchor - ag.state.pos) - kd * ag.state.vel
    elif mode == "track_contact":
        local = torch.as_tensor(helper_local, dtype=ag.state.pos.dtype, device=env.device)
        offset = _rotate(local.expand(env.num_envs, 2), bar.state.rot[:, 0])
        desired = bar.state.pos + offset
        target_v = bar.state.vel + bar.state.ang_vel * torch.stack((-offset[:, 1], offset[:, 0]), 1)
        force = kp * (desired - ag.state.pos) + kd * (target_v - ag.state.vel)
    else:
        force = -kd * ag.state.vel - damping_gain * bar.state.vel
    return norm_bound(force)


def model_features(env, primary, duration, anchor):
    """Shared response-model interface, including the helper's fixed anchor memory.

    During these reset-based trials the anchor equals initial helper position,
    so it is redundant but explicit. A continuation query must pass the original
    anchor; reinitializing it would be a different helper controller. Continuation
    states are not part of the reset-based model's training distribution.
    """
    n = env.num_envs
    duration = torch.as_tensor(duration, dtype=torch.float32, device=env.device).reshape(-1, 1).expand(n, 1)
    coarse = torch.cat((coarse_state(env), primary, duration * env.world.dt,
                        torch.full((n, 1), float(env.scenario.substeps), device=env.device), anchor), 1)
    velocities = torch.cat([torch.cat((e.state.vel, e.state.ang_vel), 1)
                             for e in (*env.world.agents, env.scenario.bar, env.scenario.stem)], 1)
    full = torch.cat((coarse, velocities, torch.tensor([env.scenario.friction_ratio,
                     env.scenario.rot_friction_ratio], device=env.device).expand(n, 2)), 1)
    return coarse, full


def action_grid():
    """Predetermined local training/validation and intervention holdout programs."""
    result = []
    grids = (("train", (-20, 0, 20), (.3, .6, .9), (5, 10, 20)),
             ("val", (-10, 10), (.45, .75), (10, 20)),
             ("test_action", (-35, 35), (.45, .75), (15, 30)))
    for split, directions, strengths, durations in grids:
        for direction, strength, duration in itertools.product(directions, strengths, durations):
            result.append(dict(direction_deg=direction, strength=strength, duration=duration,
                               action_split=split))
    # No-push controls provide action authority at every evaluated duration.
    for duration in (5, 10, 15, 20, 30):
        result.append(dict(direction_deg=0, strength=0.0, duration=duration,
                           action_split="train" if duration in (5, 10, 20) else "test_action"))
    for i, row in enumerate(result):
        row["action_id"] = i
    return result


def nuisance_grid():
    """Explicit legal initial conditions, with two larger heldout combinations."""
    result = [(0., 0., 0.), (.03, 0., 0.), (-.03, 0., 0.),
              (0., .03, 0.), (0., -.03, 0.), (0., 0., .2), (0., 0., -.2),
              (.045, 0., .3), (-.045, 0., -.3)]
    return [dict(hidden_id=i, vx=vx, vy=vy, omega=w, nuisance_is_train=i < 7)
            for i, (vx, vy, w) in enumerate(result)]


def _split(angle, action, nuisance):
    if angle != 0:
        return "test_context"
    if action["action_split"] == "test_action":
        return "test_action"
    if not nuisance["nuisance_is_train"]:
        return "test_nuisance"
    return action["action_split"]


def audit_group(*, friction_ratio, substeps, angles_deg=(0., 30.), actions=None,
                nuisances=None, modes=MODES, kp=30., kd=4., damping_gain=8.,
                preload=0.0, helper_error=(0.0, 0.0), save_trajectories=True, seed=0):
    """Run a vectorized factorial group with exact matched starts across modes."""
    from vmas_pusht.snapshot import capture

    if friction_ratio < 0 or substeps < 1 or preload < 0:
        raise ValueError("Friction/preload must be nonnegative and substeps positive")
    actions = action_grid() if actions is None else actions
    nuisances = nuisance_grid() if nuisances is None else nuisances
    rows = []
    for angle, action, nuisance, mode in itertools.product(angles_deg, actions, nuisances, modes):
        rows.append(dict(mode=mode, angle_deg=angle, **action, **nuisance,
                         friction_ratio=friction_ratio, substeps=substeps,
                         preload=preload, helper_error=list(helper_error),
                         context_id=f"angle{angle}:fr{friction_ratio}:ss{substeps}:n{nuisance['hidden_id']}",
                         alias_group_id=f"angle{angle}:ss{substeps}:action{action['action_id']}",
                         matched_key=f"angle{angle}:fr{friction_ratio}:ss{substeps}:n{nuisance['hidden_id']}:a{action['action_id']}",
                         action_is_train=action["action_split"] == "train",
                         context_is_train=angle == 0 and nuisance["nuisance_is_train"],
                         split=_split(angle, action, nuisance)))
    n = len(rows)
    env = make(num_envs=n, device="cpu", seed=seed, friction_ratio=friction_ratio, substeps=substeps)
    angles = torch.tensor([math.radians(r["angle_deg"]) for r in rows])
    anchor = setup_scene(env, angle=angles, hidden_v=[[r["vx"], r["vy"]] for r in rows],
                         hidden_omega=torch.tensor([r["omega"] for r in rows])[:, None],
                         helper_error=helper_error)
    # A prespecified optional finite preload probes whether the first pilot's
    # failure to maintain contact was simply lack of a compressive preload.
    # This changes PD reference points only, never physical placement/forces.
    anchor = anchor + _rotate(torch.tensor([-preload, preload]).expand(n, 2), angles)
    tracking_local = (HELPER_LOCAL[0] - preload, HELPER_LOCAL[1] + preload)
    initial = physical_state(env).clone()
    physics_configuration = capture(env).physics
    if not torch.equal(initial.reshape(-1, len(modes), 24),
                       initial.reshape(-1, len(modes), 24)[:, :1].expand(-1, len(modes), -1)):
        raise AssertionError("Helper modes did not start at exactly matching physical states")
    gaps0 = surface_gaps(env)
    if gaps0.amin() < -1e-6:
        raise AssertionError("Initial bodies overlap")
    durations = torch.tensor([r["duration"] for r in rows])
    direction = torch.tensor([math.radians(r["direction_deg"]) for r in rows]) + angles
    strength = torch.tensor([r["strength"] for r in rows])
    primary = torch.stack((direction.sin(), -direction.cos()), 1) * strength[:, None]
    dt = env.world.dt
    coarse, full = model_features(env, primary, durations, anchor)
    initial_bar = env.scenario.bar.state.pos.clone()
    initial_rot = env.scenario.bar.state.rot[:, 0].clone()
    initial_com = t_com(env).clone()
    Y, com_delta = torch.empty(n, 3), torch.empty(n, 2)
    finite, wall = _finite(env), _at_wall(env)
    maximum_norm, norm_impulse, effort, work, travel, contacts = [torch.zeros(n, 2) for _ in range(6)]
    double_contacts, helper_saturation = torch.zeros(n), torch.zeros(n)
    max_anchor_drift, max_joint_position_residual, max_joint_angle_residual = [torch.zeros(n) for _ in range(3)]
    min_gap = gaps0.amin(-1)
    saved_stats = {}
    trajectories = [initial] if save_trajectories else []
    commands = []
    mode_indices = {m: torch.tensor([i for i, r in enumerate(rows) if r["mode"] == m]) for m in modes}
    # Every branch is advanced for the largest duration, but statistics and Y
    # are captured only at that branch's requested macro horizon.
    for t in range(int(durations.max())):
        helper = torch.empty(n, 2)
        for m, indices in mode_indices.items():
            helper[indices] = helper_action(env, m, anchor, kp=kp, kd=kd, damping_gain=damping_gain,
                                            helper_local=tracking_local)[indices]
        action = torch.stack((primary, helper), 1)
        action *= (t < durations)[:, None, None]
        old_positions = torch.stack([a.state.pos.clone() for a in env.world.agents], 1)
        old_velocities = torch.stack([a.state.vel.clone() for a in env.world.agents], 1)
        env.step([action[:, 0], action[:, 1]])
        new_positions = torch.stack([a.state.pos for a in env.world.agents], 1)
        new_velocities = torch.stack([a.state.vel for a in env.world.agents], 1)
        norms = action.norm(dim=-1)
        maximum_norm = torch.maximum(maximum_norm, norms)
        norm_impulse += norms * env.scenario.agent_force * dt
        effort += norms.square() * dt  # Actuator effort proxy, not physical energy.
        work += (action * env.scenario.agent_force * (old_velocities + new_velocities) / 2).sum(-1).abs() * dt
        travel += (new_positions - old_positions).norm(dim=-1)
        part_gaps = surface_gaps(env)
        gap = part_gaps.amin(-1)
        contacts += (gap <= .005)
        double_contacts += (part_gaps[:, 1] <= .005).all(-1)
        helper_saturation += norms[:, 1] >= 1 - 1e-6
        max_anchor_drift = torch.maximum(max_anchor_drift, (new_positions[:, 1] - anchor).norm(dim=-1))
        bar, stem = env.scenario.bar, env.scenario.stem
        bar_joint = bar.state.pos + _rotate(torch.tensor([0., -.05]).expand(n, 2), bar.state.rot[:, 0])
        stem_joint = stem.state.pos + _rotate(torch.tensor([0., .15]).expand(n, 2), stem.state.rot[:, 0])
        max_joint_position_residual = torch.maximum(max_joint_position_residual, (bar_joint - stem_joint).norm(dim=-1))
        max_joint_angle_residual = torch.maximum(max_joint_angle_residual, wrap(bar.state.rot[:, 0] - stem.state.rot[:, 0]).abs())
        min_gap = torch.minimum(min_gap, gap)
        finite &= _finite(env)
        wall |= _at_wall(env)
        at_horizon = durations == t + 1
        result = torch.cat((env.scenario.bar.state.pos - initial_bar,
                            wrap(env.scenario.bar.state.rot[:, 0] - initial_rot)[:, None]), 1)
        Y[at_horizon] = result[at_horizon]
        com_delta[at_horizon] = (t_com(env) - initial_com)[at_horizon]
        for name, value in dict(maximum_action_norm_by_agent=maximum_norm,
                                norm_impulse_by_agent=norm_impulse,
                                squared_control_effort_by_agent=effort,
                                absolute_actuator_work_proxy_by_agent=work,
                                agent_travel=travel, contact_fraction_by_agent=contacts / (t + 1),
                                helper_double_contact_fraction=double_contacts / (t + 1),
                                helper_saturation_fraction=helper_saturation / (t + 1),
                                max_helper_anchor_drift=max_anchor_drift,
                                max_joint_position_residual=max_joint_position_residual,
                                max_joint_angle_residual=max_joint_angle_residual,
                                minimum_gap_by_agent=min_gap, finite_all_steps=finite,
                                touched_arena_boundary=wall).items():
            if name not in saved_stats:
                saved_stats[name] = torch.empty_like(value)
            saved_stats[name][at_horizon] = value[at_horizon]
        if save_trajectories:
            trajectories.append(physical_state(env).clone())
            commands.append(action.clone())
    for i, row in enumerate(rows):
        row.update({key: _plain(value[i]) for key, value in saved_stats.items()})
        row["com_delta"] = _plain(com_delta[i])
        row["bar_delta"] = _plain(Y[i])
        row["initial_minimum_gap"] = float(gaps0[i].min())
    result = dict(X_coarse=coarse, X_full=full, Y=Y, rows=rows, initial_physical_state=initial,
                  com_delta=com_delta, exact_matched_starts=True,
                  physics_configuration=physics_configuration)
    if save_trajectories:
        result["trajectories"] = torch.stack(trajectories, 1)
        result["commands"] = torch.stack(commands, 1)
    return result


def summarize(data):
    """Nuisance dispersion and primary authority, kept separate from success."""
    import numpy as np

    rows, y = data["rows"], data["Y"].numpy()
    groups = {}
    for i, r in enumerate(rows):
        # Friction varies inside each alias group, alongside hidden velocities.
        key = (r["mode"], r["angle_deg"], r["substeps"], r["action_id"])
        groups.setdefault(key, []).append(i)
    per_alias = []
    means = {}
    for key, ix in groups.items():
        sample = rows[ix[0]]
        values = y[ix]
        mean, var = values.mean(0), values.var(0)
        means[key] = mean
        per_alias.append(dict(mode=key[0], angle_deg=key[1], substeps=key[2], action_id=key[3],
                              action_split=sample["action_split"], duration=sample["duration"],
                              strength=sample["strength"], direction_deg=sample["direction_deg"],
                              samples=len(ix), mean_response=mean.tolist(), variance=var.tolist(),
                              theta_std=float(np.sqrt(var[2])),
                              metric_std=float(np.sqrt(var[0] + var[1] + .2**2 * var[2]))))
    by_condition = []
    for mode, angle, substeps in sorted({(r["mode"], r["angle_deg"], r["substeps"]) for r in rows}):
        for duration in sorted({r["duration"] for r in rows}):
            selected = [r for r in per_alias if (r["mode"], r["angle_deg"], r["substeps"], r["duration"]) ==
                         (mode, angle, substeps, duration)]
            driven = [r for r in selected if r["strength"] > 0]
            zero = [r for r in selected if r["strength"] == 0]
            if not driven or not zero:
                continue
            baseline = np.asarray(zero[0]["mean_response"])
            authority = np.asarray([r["mean_response"] for r in driven]) - baseline
            by_condition.append(dict(mode=mode, angle_deg=angle, substeps=substeps, duration=duration,
                driven_actions=len(driven), mean_theta_nuisance_std=float(np.mean([r["theta_std"] for r in driven])),
                mean_metric_nuisance_std=float(np.mean([r["metric_std"] for r in driven])),
                rms_primary_theta_authority=float(np.sqrt(np.mean(authority[:, 2]**2))),
                maximum_primary_theta_authority=float(np.max(np.abs(authority[:, 2]))),
                rms_primary_metric_authority=float(np.sqrt(np.mean(authority[:, 0]**2 + authority[:, 1]**2 + .2**2 * authority[:, 2]**2)))))
    return dict(rows=len(rows), finite_rows=sum(r["finite_all_steps"] for r in rows),
                wall_rows=sum(r["touched_arena_boundary"] for r in rows),
                max_action_norm=max(max(r["maximum_action_norm_by_agent"]) for r in rows),
                by_condition=by_condition, per_alias=per_alias)


def matched_progress(data, thresholds=(.01, .02)):
    """Compare translation dispersion at equal positive rotational progress.

    Only matched initial-state/action trials that reach the threshold under
    EVERY helper mode are retained, and coverage is reported. The rotation
    variance at its own stopping threshold is tautologically zero, so this
    diagnostic measures bar-position variance, elapsed time, effort and travel.
    No claim is made about unreached trials or whole-task success.
    """
    import numpy as np

    if "trajectories" not in data:
        return []
    rows = data["rows"]
    trajectory = data["trajectories"].numpy()
    commands = data["commands"].numpy()
    modes = tuple(dict.fromkeys(r["mode"] for r in rows))
    dt = data.get("metadata", {}).get("dt", .1)
    results = []
    for threshold in thresholds:
        crossed, records = {}, {}
        for i, r in enumerate(rows):
            if r["strength"] <= 0:
                continue
            path = trajectory[i, :r["duration"] + 1]
            theta = (path[:, 16] - path[0, 16] + math.pi) % (2 * math.pi) - math.pi
            ix = np.flatnonzero(theta >= threshold)
            if not len(ix):
                continue
            t = int(ix[0])
            fraction = float((threshold - theta[t-1]) / (theta[t] - theta[t-1]))
            position = path[t-1, 12:14] + fraction * (path[t, 12:14] - path[t-1, 12:14]) - path[0, 12:14]
            command = commands[i, :t]
            weights = np.ones(t) * dt
            weights[-1] *= fraction
            impulse = float(np.sum(np.linalg.norm(command, axis=-1) * weights[:, None]))
            effort = float(np.sum((command**2).sum(-1) * weights[:, None]))
            step_travel = np.stack([np.linalg.norm(np.diff(path[:t+1, j:j+2], axis=0), axis=1) for j in (0, 6)], 1)
            step_travel[-1] *= fraction
            record = dict(position=position, time=(t-1+fraction)*dt, total_norm_impulse=impulse,
                          total_effort=effort, total_travel=float(step_travel.sum()))
            records[i] = record
            crossed.setdefault(r["matched_key"], {})[r["mode"]] = i
        common = {key: mapping for key, mapping in crossed.items() if len(mapping) == len(modes)}
        for ss in sorted({r["substeps"] for r in rows}):
            for angle in sorted({r["angle_deg"] for r in rows}):
                eligible = [i for i, r in enumerate(rows) if r["substeps"] == ss and
                            r["angle_deg"] == angle and r["strength"] > 0]
                if not eligible:
                    continue
                for mode in modes:
                    own = [i for i in eligible if rows[i]["mode"] == mode]
                    retained = [i for i in own if rows[i]["matched_key"] in common]
                    groups = {}
                    for i in retained:
                        groups.setdefault(rows[i]["alias_group_id"], []).append(records[i]["position"])
                    variances = [float(np.asarray(v).var(0).sum()) for v in groups.values() if len(v) > 1]
                    def mean(field):
                        return float(np.mean([records[i][field] for i in retained])) if retained else None
                    results.append(dict(mode=mode, angle_deg=angle, substeps=ss, target_angle=threshold,
                        eligible_rows=len(own), reached_rows=sum(i in records for i in own),
                        common_reached_rows=len(retained), common_reached_fraction=len(retained)/len(own),
                        alias_groups_with_replication=len(variances),
                        mean_bar_position_std_at_progress=float(np.mean(np.sqrt(variances))) if variances else None,
                        mean_time=mean("time"), mean_total_norm_impulse=mean("total_norm_impulse"),
                        mean_total_effort=mean("total_effort"), mean_total_travel=mean("total_travel")))
    return results


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def provenance(args, groups):
    """Plain-string runtime versions and exact source/physics provenance."""
    import vmas.simulator.core

    root = Path(__file__).resolve().parents[1]
    paths = [Path("vmas_pusht") / name for name in
             ("physical_abstraction.py", "scenario.py", "boundary.py", "snapshot.py")]
    hashes = {str(path): file_sha256(root / path) for path in paths}
    hashes["installed_vmas/simulator/core.py"] = file_sha256(vmas.simulator.core.__file__)

    def git(*arguments):
        result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return dict(command_argv=list(sys.argv), full_args=vars(args).copy(),
                versions={name: str(version(name)) for name in ("torch", "vmas", "numpy")},
                python_version=str(platform.python_version()), platform=str(platform.platform()),
                git_head=git("rev-parse", "HEAD"), git_branch=git("branch", "--show-current"),
                source_sha256=hashes,
                physics_configurations=[dict(friction_ratio=g["rows"][0]["friction_ratio"],
                                              substeps=g["rows"][0]["substeps"],
                                              configuration=_plain(g["physics_configuration"])) for g in groups])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/novelty/abstraction_physics.pt")
    p.add_argument("--frictions", type=float, nargs="+", default=[.75, 1.5, 2.25])
    p.add_argument("--substeps", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--angles", type=float, nargs="+", default=[0., 30.])
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--kp", type=float, default=30.)
    p.add_argument("--kd", type=float, default=4.)
    p.add_argument("--damping-gain", type=float, default=8.)
    p.add_argument("--preload", type=float, default=0., help="Finite PD target inset along both corner normals")
    p.add_argument("--helper-error", type=float, nargs=2, default=[0., 0.], help="Legal local placement perturbation")
    p.add_argument("--no-trajectories", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Small debugging grid; not the registered pilot")
    args = p.parse_args()
    torch.set_num_threads(1)
    started = time.time()
    groups = []
    for friction, substeps in itertools.product(args.frictions, args.substeps):
        group = audit_group(friction_ratio=friction, substeps=substeps, angles_deg=args.angles,
                            modes=args.modes, kp=args.kp, kd=args.kd, damping_gain=args.damping_gain,
                            preload=args.preload, helper_error=args.helper_error,
                            save_trajectories=not args.no_trajectories,
                            actions=[action_grid()[13], action_grid()[-2]] if args.smoke else None,
                            nuisances=nuisance_grid()[:7] if args.smoke else None)
        groups.append(group)
        print(json.dumps(dict(friction=friction, substeps=substeps, rows=len(group["rows"]),
                              seconds=round(time.time() - started, 2),
                              mean_abs_angle=float(group["Y"][:, 2].abs().mean()))), flush=True)
    data = {k: torch.cat([g[k] for g in groups], 0) for k in groups[0] if torch.is_tensor(groups[0][k])}
    data["rows"] = [r for g in groups for r in g["rows"]]
    data["exact_matched_starts"] = all(g["exact_matched_starts"] for g in groups)
    data["metadata"] = dict(schema_version=1, coarse_features=COARSE_FEATURES,
                            full_features=COARSE_FEATURES + FULL_EXTRA_FEATURES,
                            Y_features=["bar_dx", "bar_dy", "bar_dtheta_wrapped"],
                            physical_state_features=[f"{e}_{f}" for e in ("A", "B", "bar", "stem")
                                                      for f in ("x", "y", "vx", "vy", "theta", "omega")],
                            gains=dict(kp=args.kp, kd=args.kd, damping_gain=args.damping_gain),
                            preload=args.preload, helper_error=args.helper_error,
                            primary_local=PRIMARY_LOCAL, helper_local=HELPER_LOCAL,
                            dt=.1, learned_helper_parameters=0,
                            length_units="uncalibrated VMAS scene units", time_units="VMAS simulation time units",
                            initial_conditions="Constructed rigid-body-consistent velocities; not generated histories",
                            trajectory_validity="For row i only times 0..rows[i]['duration'] are part of its macro trial",
                            wall_seconds=time.time()-started)
    data["metadata"].update(provenance(args, groups))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, target)
    summary = summarize(data)
    target.with_suffix(".json").write_text(json.dumps(_plain(dict(metadata=data["metadata"], summary=summary,
                                                                dataset_sha256=file_sha256(target),
                                                                matched_progress=matched_progress(data))), indent=2, allow_nan=False))
    print(json.dumps(dict(output=str(target), rows=len(data["rows"]), finite=summary["finite_rows"],
                          walls=summary["wall_rows"], max_action_norm=summary["max_action_norm"],
                          seconds=round(time.time()-started, 2))), flush=True)


if __name__ == "__main__":
    main()
