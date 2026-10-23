"""Matched empirical solo/joint Push-T audit, not a proof of solo impossibility.

Run with ``python -m vmas_pusht.boundary --output data/novelty/boundary.json``.
Each intervention starts from the same complete simulator snapshot. An inactive
agent remains a passive, collidable body with zero commanded action. Controls
have a fixed Euclidean norm, avoiding the box-action diagonal force advantage.
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

from vmas_pusht.scenario import AGENT_R, BAR, STEM, STEM_OFFSET, make, wrap


BRANCHES = ((0, 0), (1, 0), (0, 1), (1, 1))
BRANCH_NAMES = ("00", "10", "01", "11")
LAYOUTS = ("wide", "narrow", "side", "centered")


def _rotate(v, angle):
    c, s = angle.cos(), angle.sin()
    return torch.stack((c * v[..., 0] - s * v[..., 1],
                        s * v[..., 0] + c * v[..., 1]), -1)


def place_contact(env, angles=None, layout="wide", goal_distance=0.2, gap=0.001):
    """Place two resting agents next to a resting T; return unit push direction.

    ``angles`` is a scalar or tensor of shape (B,) in radians. ``layout`` may be
    one name or a length-B sequence of names. Wide/narrow put both agents above
    the bar, at x=+/-0.13 or +/-0.065, pushing toward local -y. Side puts one
    agent left of each T part, pushing toward local +x. Centered puts A on the
    bar's centerline and B at x=0.13, providing a torque-free solo A control.
    All configurations have
    nonoverlapping agents when gap >= 0. Reset before calling for a fresh trial;
    this function changes physical placement and goals, not simulator counters.
    """
    if len(env.world.agents) != 2:
        raise ValueError("Contact audit requires exactly two agents")
    if gap < 0:
        raise ValueError("gap must be nonnegative")
    B, dev = env.world.batch_dim, env.world.device
    angles = torch.as_tensor(0.0 if angles is None else angles,
                             dtype=torch.float32, device=dev).expand(B)
    layouts = [layout] * B if isinstance(layout, str) else list(layout)
    if len(layouts) != B or any(x not in LAYOUTS for x in layouts):
        raise ValueError(f"layout must contain {B} entries from {LAYOUTS}")
    positions = torch.empty(B, 2, 2, device=dev)
    direction = torch.zeros(B, 2, device=dev)
    for name in LAYOUTS:
        mask = torch.tensor([x == name for x in layouts], device=dev)
        if name in ("wide", "narrow", "centered"):
            x = 0.13 if name == "wide" else 0.065
            left, right = (0.0, 0.13) if name == "centered" else (-x, x)
            positions[mask] = torch.tensor([[left, BAR[1] / 2 + AGENT_R + gap],
                                            [right, BAR[1] / 2 + AGENT_R + gap]], device=dev)
            direction[mask, 1] = -1.0
        else:
            positions[mask] = torch.tensor([[-BAR[0] / 2 - AGENT_R - gap, 0.0],
                                            [-STEM[0] / 2 - AGENT_R - gap, STEM_OFFSET]], device=dev)
            direction[mask, 0] = 1.0
    direction = _rotate(direction, angles)
    positions = _rotate(positions, angles[:, None])
    sc = env.scenario
    pose = torch.cat((torch.zeros(B, 2, device=dev), angles[:, None]), 1)
    sc.place_t(pose)
    goal = pose.clone()
    goal[:, :2] += float(goal_distance) * direction
    sc.place_t(goal, goal=True)
    sc.goal_pose[:] = goal
    for i, agent in enumerate(env.world.agents):
        agent.set_pos(positions[:, i].contiguous(), batch_index=None)
        agent.set_vel(torch.zeros(B, 2, device=dev), batch_index=None)
        agent.set_rot(torch.zeros(B, 1, device=dev), batch_index=None)
        agent.set_ang_vel(torch.zeros(B, 1, device=dev), batch_index=None)
        agent.action.u = torch.zeros(B, 2, device=dev)
    sc.prev_err[:] = sc.pose_err()
    return direction


def t_com(env):
    """Mass-weighted COM of both simulated T parts (not just bar position)."""
    bar, stem = env.scenario.bar, env.scenario.stem
    return (bar.mass * bar.state.pos + stem.mass * stem.state.pos) / (bar.mass + stem.mass)


def surface_gaps(env):
    """Exact geometric sphere-to-box signed gaps, shape (B, agent, T part)."""
    distances = []
    for agent in env.world.agents:
        part_distances = []
        for part in (env.scenario.bar, env.scenario.stem):
            local = _rotate(agent.state.pos - part.state.pos, -part.state.rot[:, 0])
            half_extent = local.new_tensor([part.shape.length, part.shape.width]) / 2
            q = local.abs() - half_extent
            signed = q.clamp_min(0).norm(dim=-1) + q.max(dim=-1).values.clamp_max(0)
            part_distances.append(signed - agent.shape.radius)
        distances.append(torch.stack(part_distances, -1))
    return torch.stack(distances, 1)


def intervention_actions(direction, strength, delay, branch, step):
    """Match A/B controls across 00/10/01/11, including B's onset delay."""
    active = branch.to(direction.dtype).clone()
    active[:, 1] *= (step >= delay)
    return direction[:, None, :] * strength[:, None, None] * active[:, :, None]


def interaction_contrast(values):
    """Last dimension must be ordered 00, 10, 01, 11."""
    if values.shape[-1] != 4:
        raise ValueError("Expected four matched branches")
    return values[..., 3] - values[..., 1] - values[..., 2] + values[..., 0]


def _finite(env):
    tensors = [x for entity in env.world.entities for x in
               (entity.state.pos, entity.state.vel, entity.state.rot, entity.state.ang_vel)]
    return torch.cat(tensors, -1).isfinite().all(-1)


def _at_wall(env):
    """Geometric contact with the arena's axis-aligned position limits."""
    touching = torch.zeros(env.num_envs, dtype=torch.bool, device=env.world.device)
    lim = torch.tensor([env.world.x_semidim, env.world.y_semidim], device=env.world.device)
    for entity in (*env.world.agents, env.scenario.bar, env.scenario.stem):
        if entity in env.world.agents:
            ext = entity.shape.radius
        else:
            r = entity.state.rot[:, 0]
            c, s = r.cos().abs(), r.sin().abs()
            l, w = entity.shape.length / 2, entity.shape.width / 2
            ext = torch.stack((c * l + s * w, s * l + c * w), -1)
        touching |= ((entity.state.pos.abs() + ext) >= lim - 1e-6).any(-1)
    return touching


def _plain(value):
    """Emit standards-compliant JSON; nonfinite measurements become null."""
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def audit_group(*, friction_ratio, substeps, angles_deg=(0, 45),
                layouts=("wide", "side", "centered"), strengths=(0.5, 1.0),
                delays=(0, 10, 25), horizons=(5, 25, 100, 200),
                prelude=10, seed=0, goal_distance=0.2, contact_tolerance=0.005):
    """Evaluate a deterministic matched grid under one simulator configuration."""
    from vmas_pusht.snapshot import capture, restore

    conditions = [dict(angle_deg=a, layout=l, strength=f, delay=d)
                  for a, l, f, d in itertools.product(angles_deg, layouts, strengths, delays)]
    source = make(num_envs=len(conditions), device="cpu", seed=seed,
                  friction_ratio=friction_ratio, substeps=substeps)
    directions = place_contact(source,
        angles=torch.tensor([math.radians(x["angle_deg"]) for x in conditions]),
        layout=[x["layout"] for x in conditions], goal_distance=goal_distance)
    initial_placement_gaps = surface_gaps(source)
    for _ in range(prelude):
        source.step([torch.zeros(len(conditions), 2) for _ in range(2)])
    snap = capture(source)
    env = make(num_envs=4 * len(conditions), device="cpu", seed=seed,
               friction_ratio=friction_ratio, substeps=substeps)
    indices = torch.arange(len(conditions)).repeat_interleave(4)
    restore(env, snap, indices=indices)
    directions = directions.repeat_interleave(4, 0)
    initial_com = t_com(env).clone()
    initial_bar = env.scenario.bar.state.pos.clone()
    initial_rot = env.scenario.bar.state.rot[:, 0].clone()
    strength = torch.tensor([x["strength"] for x in conditions]).repeat_interleave(4)
    delay = torch.tensor([x["delay"] for x in conditions]).repeat_interleave(4)
    branch = torch.tensor(BRANCHES).repeat(len(conditions), 1)
    # Assert equality of all physical fields, including the separately simulated stem.
    matched = all(torch.equal(x.reshape(-1, 4, x.shape[-1]),
                             x.reshape(-1, 4, x.shape[-1])[:, :1].expand(-1, 4, -1))
                  for entity in env.world.entities for x in
                  (entity.state.pos, entity.state.vel, entity.state.rot, entity.state.ang_vel))
    if not matched:
        raise AssertionError("Matched intervention snapshot restoration changed physical states")
    start = dict(com=initial_com[::4], bar=initial_bar[::4],
                 stem=env.scenario.stem.state.pos[::4].clone(),
                 agents=torch.stack([a.state.pos[::4].clone() for a in env.world.agents], 1),
                 goal=env.scenario.goal_pose[::4].clone(),
                 placement_min_gap=initial_placement_gaps.amin((1, 2)),
                 prelude_min_gap=surface_gaps(env)[::4].amin((1, 2)))
    n = env.num_envs
    finite = _finite(env)
    ever_success = env.scenario.success().clone()
    first_success = torch.where(ever_success, 0, -1)
    at_wall = _at_wall(env)
    first_boundary = torch.where(at_wall, 0, -1)
    min_gap = surface_gaps(env).amin(-1)
    contact_steps = torch.zeros(n, 2)
    max_norm = torch.zeros(n, 2)
    rows = []
    for step in range(max(horizons)):
        actions = intervention_actions(directions, strength, delay, branch, step)
        max_norm = torch.maximum(max_norm, actions.norm(dim=-1))
        env.step([actions[:, 0], actions[:, 1]])
        finite &= _finite(env)
        success = env.scenario.success()
        first_success = torch.where(success & ~ever_success, step + 1, first_success)
        ever_success |= success
        touches = _at_wall(env)
        first_boundary = torch.where(touches & ~at_wall, step + 1, first_boundary)
        at_wall |= touches
        gaps = surface_gaps(env).amin(-1)
        min_gap = torch.minimum(min_gap, gaps)
        contact_steps += gaps <= contact_tolerance
        horizon = step + 1
        if horizon not in horizons:
            continue
        com_delta = t_com(env) - initial_com
        bar_delta = env.scenario.bar.state.pos - initial_bar
        signed = (com_delta * directions).sum(-1)
        contrast = interaction_contrast(signed.reshape(-1, 4))
        angle = wrap(env.scenario.bar.state.rot[:, 0] - initial_rot)
        goal_err = (env.scenario.bar.state.pos - env.scenario.goal_pose[:, :2]).norm(dim=-1)
        for i, condition in enumerate(conditions):
            interventions = {}
            for j, name in enumerate(BRANCH_NAMES):
                k = 4 * i + j
                interventions[name] = dict(
                    directed_com_displacement=signed[k], com_displacement=com_delta[k],
                    com_distance=com_delta[k].norm(), bar_displacement=bar_delta[k],
                    directed_bar_displacement=(bar_delta[k] * directions[k]).sum(),
                    bar_distance=bar_delta[k].norm(), signed_angle_change_rad=angle[k],
                    goal_position_error=goal_err[k], goal_angle_error_rad=wrap(
                        env.scenario.bar.state.rot[k, 0] - env.scenario.goal_pose[k, 2]).abs(),
                    success_ever=ever_success[k], success_final=success[k],
                    first_success_step=first_success[k], finite_all_steps=finite[k],
                    touched_arena_boundary=at_wall[k], first_boundary_step=first_boundary[k],
                    minimum_surface_gap_by_agent=min_gap[k],
                    contact_fraction_by_agent=contact_steps[k] / horizon,
                    maximum_action_norm_by_agent=max_norm[k])
            rows.append(_plain(dict(**condition, horizon=horizon, friction_ratio=friction_ratio,
                             substeps=substeps, initial={k: v[i] for k, v in start.items()},
                             interventions=interventions, directed_com_interaction=contrast[i])))
    return _plain(dict(snapshot_matched_exactly=matched, physics_configuration=snap.physics, rows=rows))


def summarize(rows):
    """Conservative aggregate counts; raw trajectories remain the primary result."""
    by_physics = []
    for fr, ss in sorted({(r["friction_ratio"], r["substeps"]) for r in rows}):
        selected = [r for r in rows if r["friction_ratio"] == fr and r["substeps"] == ss]
        for h in sorted({r["horizon"] for r in selected}):
            records = [r for r in selected if r["horizon"] == h]
            valid = [r for r in records if all(b["finite_all_steps"] for b in r["interventions"].values())]
            solo_moves = sum(any(r["interventions"][b]["directed_com_displacement"] > 0.001
                                 for b in ("10", "01")) for r in valid)
            solo_success = sum(any(r["interventions"][b]["success_ever"] for b in ("10", "01")) for r in valid)
            joint_success = sum(r["interventions"]["11"]["success_ever"] for r in valid)
            joint_only = sum(r["interventions"]["11"]["success_ever"] and not any(
                r["interventions"][b]["success_ever"] for b in ("10", "01")) for r in valid)

            def succeeded_before_wall(branch):
                return branch["success_ever"] and (branch["first_boundary_step"] < 0 or
                    branch["first_success_step"] < branch["first_boundary_step"])

            by_physics.append(dict(friction_ratio=fr, substeps=ss, horizon=h,
                conditions=len(records), finite_conditions=len(valid),
                any_solo_directed_displacement_over_0p001_units=solo_moves,
                any_solo_success_ever=solo_success, joint_success_ever=joint_success,
                any_solo_success_before_boundary=sum(any(succeeded_before_wall(r["interventions"][b])
                    for b in ("10", "01")) for r in valid),
                joint_success_before_boundary=sum(succeeded_before_wall(r["interventions"]["11"]) for r in valid),
                joint_only_success_in_tested_controls=joint_only,
                any_branch_touched_arena=sum(any(b["touched_arena_boundary"] for b in r["interventions"].values()) for r in valid)))
    return by_physics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/novelty/boundary.json"))
    parser.add_argument("--frictions", type=float, nargs="+", default=[0, 0.75, 1.5, 2.25])
    parser.add_argument("--substeps", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 45])
    parser.add_argument("--layouts", choices=LAYOUTS, nargs="+", default=["wide", "side", "centered"])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 10, 25])
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 25, 100, 200])
    parser.add_argument("--prelude", type=int, default=10)
    parser.add_argument("--goal-distance", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if any(x <= 0 for x in args.substeps + args.horizons) or args.prelude < 0 or min(args.delays) < 0:
        parser.error("substeps/horizons must be positive; prelude/delays must be nonnegative")
    if any(x < 0 or x > 1 for x in args.strengths):
        parser.error("strengths must lie in [0, 1] for unit-norm force comparisons")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started = time.time()
    try:
        git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision, git_status = None, None
    config = vars(args).copy()
    config["output"] = str(args.output)
    source_paths = [Path(__file__), Path(__file__).with_name("scenario.py"), Path(__file__).with_name("snapshot.py")]
    result = dict(schema_version=1, config=config, command=[sys.executable, "-m", "vmas_pusht.boundary", *sys.argv[1:]],
        versions=dict(python=platform.python_version(), torch=torch.__version__, vmas=version("vmas")),
        git_revision=git_revision, git_status=git_status,
        source_sha256={str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else str(p):
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        interpretation=[
            "Matched fixed-direction force family, not a proof that all solo policies fail.",
            "Inactive agents are passive collidable bodies, not removed or position-clamped.",
            "VMAS kinetic friction is zero at zero speed and has componentwise timestep clipping.",
            "Success is bar position within 0.05 and angle within 20 degrees of a translated goal.",
            "Success-ever counts transient passage; success-final can be false after overshoot.",
            "No independent repetitions: deterministic condition grid, no confidence intervals.",
            "All distances are simulator scene length units; no SI calibration is assumed.",
            "Long-horizon outcomes may encounter arena boundaries; per-case flags are reported."],
        rows=[], groups=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for friction, substeps in itertools.product(args.frictions, args.substeps):
        try:
            group = audit_group(friction_ratio=friction, substeps=substeps,
                angles_deg=args.angles, layouts=args.layouts, strengths=args.strengths,
                delays=args.delays, horizons=args.horizons, prelude=args.prelude,
                seed=args.seed, goal_distance=args.goal_distance)
            result["rows"].extend(group.pop("rows"))
            group["status"] = "completed"
        except AssertionError as error:
            if substeps != 1 or str(error) != "For joints, world substeps needs to be more than 1":
                raise
            group = dict(status="unsupported", error_type=type(error).__name__, reason=str(error))
        result["groups"].append(dict(friction_ratio=friction, substeps=substeps, **group))
        result["elapsed_seconds"] = time.time() - started
        result["summary"] = summarize(result["rows"])
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
        print(json.dumps(dict(friction_ratio=friction, substeps=substeps,
                              status=group["status"],
                              elapsed_seconds=round(result["elapsed_seconds"], 1))), flush=True)
    print(f"Saved {len(result['rows'])} matched conditions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
