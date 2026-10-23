"""Read-only, same-action refinement of completed recovery MPC executions.

Replay each saved joint action at 160 and 320 integration substeps, from the
identical complete handoff snapshot and unchanged batch/order. This queries no
MPC, world model, helper policy, or route controller. Evaluation uses each
source episode's original stop time; it is NOT a closed-loop 320-substep MPC
experiment. Source-active states and post-stop zero-action padding are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.abstraction_convergence import _weld_residual, restore_across_substeps
from vmas_pusht.abstraction_transfer import ANGLE_TOLERANCE, TRANSLATION_TOLERANCE
from vmas_pusht.boundary import _at_wall, _finite
from vmas_pusht.physical_abstraction import physical_state
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.scenario import wrap
from vmas_pusht.stable_composition import snapshot_from_record


MAX_XY = .001
MAX_ANGLE = .003
DWELL = 5


def validate_source(source):
    trajectory, actions, active = (source[k] for k in ("trajectory", "actions", "active"))
    n, horizon = actions.shape[:2]
    if actions.shape != (n, horizon, 2, 2) or trajectory.shape != (n, horizon+1, 24):
        raise ValueError("Complete source states and actions are misaligned")
    if active.shape != (n, horizon) or len(source["rows"]) != n:
        raise ValueError("Source rows and activity must retain the original batch")
    if (active[:, 1:] & ~active[:, :-1]).any() or not active[:, 0].all():
        raise ValueError("Each episode must have a nonempty monotone active prefix")
    if not torch.isfinite(actions).all() or (actions.norm(dim=-1) > 1.000001).any():
        raise ValueError("Source forces must be finite and Euclidean norm bounded")
    if actions[~active].abs().gt(0).any():
        raise ValueError("Expected zero commands in post-stop padding")
    stops = torch.tensor([row["continuation_steps"] for row in source["rows"]])
    if not torch.equal(stops, active.sum(1)):
        raise ValueError("Recorded stop times do not match the active prefix")
    record = source["start"]
    if len(record["rows"]) != n or record["snapshot"]["batch_size"] != n:
        raise ValueError("Full handoff snapshot batch is required")
    keys = lambda rows: [(r["source_group"], r["source_row"]) for r in rows]
    if keys(record["rows"]) != keys(source["rows"]):
        raise ValueError("Handoff snapshot and source trajectory row order differ")
    return snapshot_from_record(record)


@torch.no_grad()
def replay(source, substeps):
    snapshot = validate_source(source)
    env = make_profile("dry_light", num_envs=len(source["rows"]), substeps=substeps, seed=101)
    restore_across_substeps(env, snapshot)
    if not torch.equal(physical_state(env), source["trajectory"][:, 0]):
        raise AssertionError("Full initial handoff physical state must match exactly")
    states = [physical_state(env).clone()]
    finite, walls = [_finite(env).clone()], [_at_wall(env).clone()]
    for commands in source["actions"].unbind(1):
        env.step([commands[:, 0].clone(), commands[:, 1].clone()])
        states.append(physical_state(env).clone())
        finite.append(_finite(env).clone())
        walls.append(_at_wall(env).clone())
    trajectory = torch.stack(states, 1)
    weld_xy, weld_angle = _weld_residual(trajectory)
    return dict(substeps=substeps, trajectory=trajectory, finite=torch.stack(finite, 1),
        touched_wall=torch.stack(walls, 1), weld_xy=weld_xy, weld_angle=weld_angle,
        time=torch.arange(trajectory.shape[1])*env.world.dt, dt=float(env.world.dt))


def score_original_stops(source, run):
    """Score the final five active post-step states at each original stop.

    ``ever_five_step_dwell`` is also retained to distinguish an early excursion
    into the target region from a terminal dwell. We do not stop or change saved
    forces when the finer simulator would hit a target at a different time.
    """
    trajectory, active = run["trajectory"], source["active"]
    n, horizon = active.shape
    if trajectory.shape != (n, horizon+1, 24):
        raise ValueError("Task scoring requires the same physical-time trajectory")
    initial = torch.tensor([row["initial_pose"] for row in source["rows"]], dtype=trajectory.dtype)
    goals = torch.tensor([row["relative_goal_rad"] for row in source["rows"]], dtype=trajectory.dtype)
    angle = wrap(initial[:, None, 2]+goals[:, None]-trajectory[:, 1:, 16]).abs()
    translation = (trajectory[:, 1:, 12:14]-initial[:, None, :2]).norm(dim=-1)
    goal_ok = (angle < ANGLE_TOLERANCE) & (translation < TRANSLATION_TOLERANCE)
    dwell = torch.zeros(n, dtype=torch.long)
    maximum = torch.zeros_like(dwell)
    last_dwell = torch.zeros_like(dwell)
    for step in range(horizon):
        dwell = torch.where(active[:, step] & goal_ok[:, step], dwell+1, 0)
        maximum = torch.maximum(maximum, dwell)
        last_dwell[active[:, step]] = dwell[active[:, step]]
    stops = active.sum(1)
    idx = torch.arange(n)
    stop_index = stops-1
    terminal_angle, terminal_shift = angle[idx, stop_index], translation[idx, stop_index]
    terminal_ok = goal_ok[idx, stop_index]
    state_mask = torch.cat((torch.ones(n, 1, dtype=torch.bool), active), 1)
    finite = (run["finite"] | ~state_mask).all(1)
    wall = (run["touched_wall"] & state_mask).any(1)
    final_dwell = last_dwell >= DWELL
    continuation = final_dwell & terminal_ok & finite & ~wall
    first = torch.tensor([r["first_goal_pass"] for r in source["rows"]])
    # Use only observed history through handoff. An original continuation's
    # future wall contact must not contaminate the score of a replacement plan.
    prefix = torch.tensor([r["original_prefix_finite"] and not r["original_prefix_wall"] for r in source["rows"]])
    return dict(stop_steps=stops, final_angle_error_rad=terminal_angle,
        final_translation_units=terminal_shift, terminal_goal_success=terminal_ok,
        last_consecutive_goal_steps=last_dwell, maximum_consecutive_goal_steps=maximum,
        terminal_five_step_dwell=final_dwell, ever_five_step_dwell=maximum >= DWELL,
        finite=finite, wall=wall, continuation_goal_success=continuation,
        ordered_goal_success=first & prefix & continuation,
        terminal_pose=torch.cat((trajectory[idx, stops, 12:14], trajectory[idx, stops, 16:17]), 1))


def _max_location(values, mask, source):
    if not mask.any():
        return None
    flat = int(torch.where(mask, values, -torch.inf).flatten().argmax())
    row, step = divmod(flat, values.shape[1])
    item = source["rows"][row]
    return dict(row=row, state_step=step, seconds_after_handoff=step*.1,
        source_group=item["source_group"], source_row=item["source_row"],
        support=item["support"], original_planner=item["planner"],
        route_mode=None if step == 0 else int(source["route_mode"][row, step-1]),
        route_phase=None if step == 0 else int(source["route_phase"][row, step-1]),
        value=float(values[row, step]))


def errors(a, b, mask, source):
    xy = (a[..., 12:14]-b[..., 12:14]).norm(dim=-1)
    angle = wrap(a[..., 16]-b[..., 16]).abs()
    if not mask.any():
        return dict(samples=0, passes=None)
    return dict(samples=int(mask.sum()), cases=int(mask.any(1).sum()),
        max_xy_units=float(xy[mask].max()), rms_xy_units=float(xy[mask].square().mean().sqrt()),
        max_angle_rad=float(angle[mask].max()), rms_angle_rad=float(angle[mask].square().mean().sqrt()),
        max_xy_location=_max_location(xy, mask, source), max_angle_location=_max_location(angle, mask, source),
        passes=bool((xy[mask] <= MAX_XY).all() and (angle[mask] <= MAX_ANGLE).all()))


def _plain(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def summarize(source, coarse, fine):
    if not torch.equal(coarse["time"], fine["time"]):
        raise ValueError("Numerical comparison requires a common physical clock")
    if not torch.equal(coarse["trajectory"][:, 0], fine["trajectory"][:, 0]):
        raise ValueError("Numerical comparison requires identical complete starts")
    n = len(source["rows"])
    mask = torch.cat((torch.ones(n, 1, dtype=torch.bool), source["active"]), 1)
    a, b = coarse["trajectory"], fine["trajectory"]
    score_a, score_b = score_original_stops(source, coarse), score_original_stops(source, fine)
    per_row = []
    for i, original in enumerate(source["rows"]):
        one = dict(source_group=original["source_group"], source_row=original["source_row"],
            support=original["support"], planner=original["planner"], first_goal_pass=original["first_goal_pass"],
            original_stop_step=int(score_a["stop_steps"][i]),
            source_continuation_goal_success=original["continuation_goal_success"],
            source_ordered_goal_success=original["ordered_goal_success"],
            replay_source={k: _plain(v[i]) for k, v in score_a.items()},
            replay_fine={k: _plain(v[i]) for k, v in score_b.items()})
        one["terminal_xy_refinement_difference"] = float((score_a["terminal_pose"][i, :2]-score_b["terminal_pose"][i, :2]).norm())
        one["terminal_angle_refinement_difference"] = float(wrap(score_a["terminal_pose"][i, 2]-score_b["terminal_pose"][i, 2]).abs())
        per_row.append(one)
    original_continuation = torch.tensor([r["continuation_goal_success"] for r in source["rows"]])
    original_ordered = torch.tensor([r["ordered_goal_success"] for r in source["rows"]])
    expected_terminal = torch.tensor([r["terminal_goal_success"] for r in source["rows"]])
    exact = bool(torch.equal(source["trajectory"], a))
    labels_match = bool(torch.equal(original_continuation, score_a["continuation_goal_success"])
        and torch.equal(original_ordered, score_a["ordered_goal_success"])
        and torch.equal(expected_terminal, score_a["terminal_goal_success"]))
    transitions = {}
    for key in ("terminal_goal_success", "terminal_five_step_dwell", "ever_five_step_dwell",
                "continuation_goal_success", "ordered_goal_success"):
        left, right = score_a[key], score_b[key]
        transitions[key] = dict(source_success=int(left.sum()), fine_success_at_same_stop=int(right.sum()),
            retained=int((left & right).sum()), lost=int((left & ~right).sum()), gained=int((~left & right).sum()),
            agreement=float((left == right).float().mean()),
            lost_rows=(left & ~right).nonzero().flatten().tolist(), gained_rows=(~left & right).nonzero().flatten().tolist())
    refinement = dict(active_prefix=errors(a, b, mask, source),
        whole_saved_prefix=errors(a, b, torch.ones_like(mask), source),
        post_stop_padding=errors(a, b, ~mask, source))
    return dict(library=source["library"], source_substeps=coarse["substeps"], fine_substeps=fine["substeps"],
        cases=n, recorded_horizon_steps=source["actions"].shape[1], dt=coarse["dt"],
        exact_source_replay=exact, source_scoring_reproduced=labels_match,
        refinement_and_stop_scoring_valid=exact and labels_match,
        source_reproduction=errors(source["trajectory"], a, torch.ones_like(mask), source),
        identical_full_start=True, full_batch_and_order_preserved=True, decision_queries=0,
        is_closed_loop_fine_mpc=False, scoring="Same original stopping time; last five active post-step states and terminal pose.",
        maximum_action_norm=float(source["actions"].norm(dim=-1).max()),
        nonzero_primary_command_samples=int((source["actions"][:, :, 0].norm(dim=-1) > 1e-8).sum()),
        tolerances=dict(max_xy_units=MAX_XY, max_angle_rad=MAX_ANGLE), refinement=refinement,
        stop_time_outcomes=transitions, per_row=per_row,
        physical_diagnostics={str(run["substeps"]): dict(all_active_finite=bool((run["finite"] | ~mask).all()),
            active_wall_cases=int((run["touched_wall"] & mask).any(1).sum()),
            max_active_weld_xy_units=float(run["weld_xy"][mask].max()),
            max_active_weld_angle_rad=float(run["weld_angle"][mask].max())) for run in (coarse, fine)})


def provenance():
    paths = [Path(__file__), Path("vmas_pusht/physics_profiles.py"),
        Path("vmas_pusht/snapshot.py"), Path("vmas_pusht/abstraction_convergence.py"),
        Path("vmas_pusht/stable_composition.py"), Path("vmas_pusht/recovery_mpc.py")]
    return dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        versions={key: importlib.metadata.version(key) for key in ("vmas", "torch", "numpy")},
        torch_threads=torch.get_num_threads(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", default=[Path("data/novelty/recovery_mpc_bank13.pt"), Path("data/novelty/recovery_mpc_expanded.pt")])
    parser.add_argument("--fine-substeps", type=int, default=320)
    parser.add_argument("--out", type=Path, default=Path("data/novelty/recovery_mpc_numerics"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = dict(provenance=provenance(), results=[],
        interpretation="Same-action refinement at source stopping times; not a finer closed-loop MPC rerun.")
    raw = []
    started = time.time()
    for path in args.sources:
        source = torch.load(path, weights_only=False, map_location="cpu")
        nominal = int(source["substeps"])
        if args.fine_substeps <= nominal:
            raise ValueError("Fine resolution must exceed source resolution")
        a, b = replay(source, nominal), replay(source, args.fine_substeps)
        result = summarize(source, a, b)
        result.update(source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report["results"].append(result)
        raw.append(dict(source_path=str(path), source_actions=source["actions"], source_active=source["active"],
            source_rows=source["rows"], start=source["start"], replay_source=a, replay_fine=b))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        raw_path = args.out.with_suffix(".pt")
        torch.save(dict(results=raw), raw_path)
        report.update(raw_path=str(raw_path), raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                      wall_seconds=time.time()-started)
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(dict(library=result["library"], exact=result["exact_source_replay"],
            scoring_match=result["source_scoring_reproduced"], active_errors=result["refinement"]["active_prefix"],
            continuation=result["stop_time_outcomes"]["continuation_goal_success"],
            ordered=result["stop_time_outcomes"]["ordered_goal_success"], seconds=report["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
