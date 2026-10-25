"""Same-action refinement of complete atomic-interface composition executions.

Every saved group is replayed in its original batch/order from the complete
post-prelude snapshot. No planner, world model, helper policy, or option state
machine is queried. First and final goals are rescored at their original phase
end times, including the final five post-step states in each rotation stage.
The original prelude is checked separately and is not rerun at finer resolution.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.abstraction_transfer import ANGLE_TOLERANCE, TRANSLATION_TOLERANCE
from vmas_pusht.boundary import _plain
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.scenario import wrap
from vmas_pusht.stable_task_numerics import replay_group


MAX_XY = .001
MAX_ANGLE = .003
DWELL = 5
PHASES = ("rotate_first", "retreat", "recovery", "rotate_final")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recorded_wall(states, env):
    """Original geometric wall predicate evaluated on stored full states."""
    mask = torch.zeros(states.shape[:-1], dtype=torch.bool)
    limits = states.new_tensor([env.world.x_semidim, env.world.y_semidim])
    for i, entity in enumerate((*env.world.agents, env.scenario.bar, env.scenario.stem)):
        position = states[..., i*6:i*6+2]
        if i < 2:
            extent = entity.shape.radius
        else:
            angle = states[..., i*6+4]
            c, s = angle.cos().abs(), angle.sin().abs()
            l, w = entity.shape.length/2, entity.shape.width/2
            extent = torch.stack((c*l+s*w, s*l+c*w), -1)
        mask |= (position.abs()+extent >= limits-1e-6).any(-1)
    return mask


def prelude_validity(group):
    prelude = group["prelude_trajectory"]
    if prelude.ndim != 3 or prelude.shape[0] != len(group["rows"]) or prelude.shape[2] != 24:
        raise ValueError("Expected each original full 24D prelude trajectory")
    env = make_profile(group["physics_profile"], num_envs=1,
        substeps=int(group["rows"][0]["substeps"]), seed=101)
    return dict(finite=torch.isfinite(prelude).all(-1).all(-1),
        wall=_recorded_wall(prelude, env).any(-1), reused_source_prelude=True)


def validate_clock(group):
    active, phases = group["active"], group["phases"]
    if active.shape != phases.shape:
        raise ValueError("Recorded physical phases and activity must align")
    n, horizon = active.shape
    if (active[:, 1:] & ~active[:, :-1]).any() or not active[:, 0].all():
        raise ValueError("Expected a nonempty monotone active prefix")
    ends = torch.tensor([[row["phase_end_steps"][phase] for phase in PHASES] for row in group["rows"]])
    if ends.shape != (n, 4) or (ends < 1).any() or (ends[:, 1:] <= ends[:, :-1]).any() or (ends > horizon).any():
        raise ValueError("All original phase endpoints must be ordered physical-step indices")
    if not torch.equal(ends[:, 3], active.sum(1)):
        raise ValueError("Final phase endpoint must match each original active stop")
    for i in range(n):
        start = 0
        for j in range(4):
            end = int(ends[i, j])
            if not phases[i, start:end].eq(j).all():
                raise ValueError("Phase clock differs from its recorded endpoint")
            start = end
    return ends


def score_original_endpoints(group, run):
    """Score original first/final stopping times, never a later favorable state."""
    ends = validate_clock(group)
    trajectory = run["trajectory"]
    n, horizon = group["active"].shape
    if trajectory.shape != (n, horizon+1, 24):
        raise ValueError("Replay requires the same full physical-time trajectory")
    initial = group["initial_pose"].to(trajectory)
    prelude = prelude_validity(group)
    state_mask = torch.cat((torch.ones(n, 1, dtype=torch.bool), group["active"]), 1)
    finite = (run["finite"] | ~state_mask).all(1) & prelude["finite"]
    wall = (run["touched_wall"] & state_mask).any(1) | prelude["wall"]
    translation = (trajectory[:, :, 12:14]-initial[:, None, :2]).norm(dim=-1)
    result = dict(finite=finite, wall=wall, prelude_finite=prelude["finite"], prelude_wall=prelude["wall"])
    for name, phase, goal_key in (("first", 0, "first_goal_rad"), ("final", 3, "relative_goal_rad")):
        goals = trajectory.new_tensor([row[goal_key] for row in group["rows"]])
        error = wrap(initial[:, None, 2]+goals[:, None]-trajectory[:, :, 16]).abs()
        good = (error < ANGLE_TOLERANCE) & (translation < TRANSLATION_TOLERANCE)
        endpoint = ends[:, phase]
        phase_start = torch.zeros(n, dtype=torch.long) if phase == 0 else ends[:, phase-1]
        elapsed = endpoint-phase_start
        last_dwell = torch.zeros(n, dtype=torch.long)
        maximum_dwell = torch.zeros_like(last_dwell)
        for i in range(n):
            counter = 0
            for step in range(int(phase_start[i])+1, int(endpoint[i])+1):
                counter = counter+1 if bool(good[i, step]) else 0
                maximum_dwell[i] = max(int(maximum_dwell[i]), counter)
            last_dwell[i] = counter
        idx = torch.arange(n)
        result.update({name+"_stop_steps": endpoint, name+"_phase_steps": elapsed,
            name+"_goal_dwell_success": (last_dwell >= DWELL) & (elapsed >= DWELL),
            name+"_last_consecutive_goal_steps": last_dwell,
            name+"_maximum_consecutive_goal_steps": maximum_dwell,
            name+"_terminal_goal_success": good[idx, endpoint],
            name+"_angle_error_rad": error[idx, endpoint],
            name+"_translation_units": translation[idx, endpoint],
            name+"_pose": torch.cat((trajectory[idx, endpoint, 12:14], trajectory[idx, endpoint, 16:17]), 1)})
    result["ordered_goal_success"] = result["first_goal_dwell_success"] & result["final_goal_dwell_success"]
    result["ordered_goal_success"] &= result["final_terminal_goal_success"] & finite & ~wall
    return result


def _location(values, mask, group):
    if not mask.any():
        return None
    index = int(torch.where(mask, values, -torch.inf).flatten().argmax())
    row_index, step = divmod(index, values.shape[1])
    row = group["rows"][row_index]
    return dict(row=row_index, state_step=step, task_seconds=step*.1,
        support=row["support"], planner=row["planner"], initial_angle_deg=row.get("initial_angle_deg"),
        nuisance_id=row.get("nuisance_id"), value=float(values[row_index, step]))


def discrepancies(coarse, fine, mask, group):
    if not mask.any():
        return dict(samples=0, cases=0, passes=None)
    xy = (coarse[..., 12:14]-fine[..., 12:14]).norm(dim=-1)
    angle = wrap(coarse[..., 16]-fine[..., 16]).abs()
    return dict(samples=int(mask.sum()), cases=int(mask.any(1).sum()),
        max_xy_units=float(xy[mask].max()), rms_xy_units=float(xy[mask].square().mean().sqrt()),
        max_angle_rad=float(angle[mask].max()), rms_angle_rad=float(angle[mask].square().mean().sqrt()),
        max_xy_location=_location(xy, mask, group), max_angle_location=_location(angle, mask, group),
        passes=bool((xy[mask] <= MAX_XY).all() and (angle[mask] <= MAX_ANGLE).all()))


def summarize(group, coarse, fine):
    n = len(group["rows"])
    ends = validate_clock(group)
    if not torch.equal(coarse["time"], fine["time"]) or not torch.equal(coarse["trajectory"][:, 0], fine["trajectory"][:, 0]):
        raise ValueError("Refinement requires identical complete starts and a common physical clock")
    a, b = coarse["trajectory"], fine["trajectory"]
    masks = dict(active_prefix=torch.cat((torch.ones(n, 1, dtype=torch.bool), group["active"]), 1))
    masks["whole_saved_prefix"] = torch.ones_like(masks["active_prefix"])
    masks["post_stop_padding"] = ~masks["active_prefix"]
    for j, name in enumerate(PHASES):
        mask = torch.zeros_like(masks["active_prefix"])
        for i in range(n):
            start = 0 if j == 0 else int(ends[i, j-1])
            mask[i, start:int(ends[i, j])+1] = True
        masks[name] = mask
    left, right = score_original_endpoints(group, coarse), score_original_endpoints(group, fine)
    original = {
        "first_goal_dwell_success": torch.tensor([row["first_goal_success_at_phase_boundary"] for row in group["rows"]]),
        "final_goal_dwell_success": torch.tensor([row["final_goal_dwell_success"] for row in group["rows"]]),
        "final_terminal_goal_success": torch.tensor([row["terminal_success"] for row in group["rows"]]),
        "ordered_goal_success": torch.tensor([row["ordered_goal_success"] for row in group["rows"]]),
    }
    exact = bool(torch.equal(group["trajectory"], a))
    labels_match = all(torch.equal(values, left[key]) for key, values in original.items())
    outcomes = {}
    for key in original:
        x, y = left[key], right[key]
        outcomes[key] = dict(source_success=int(x.sum()), fine_success_at_same_stop=int(y.sum()),
            retained=int((x & y).sum()), lost=int((x & ~y).sum()), gained=int((~x & y).sum()),
            lost_rows=(x & ~y).nonzero().flatten().tolist(), gained_rows=(~x & y).nonzero().flatten().tolist())
    rows = []
    active_xy = (a[..., 12:14]-b[..., 12:14]).norm(dim=-1)
    active_angle = wrap(a[..., 16]-b[..., 16]).abs()
    for i, source in enumerate(group["rows"]):
        row_mask = masks["active_prefix"][i]
        row = dict(row=i, support=source["support"], planner=source["planner"],
            model_seed=source.get("model_seed"),
            initial_angle_deg=source.get("initial_angle_deg"), nuisance_id=source.get("nuisance_id"),
            first_goal_rad=source["first_goal_rad"], relative_goal_rad=source["relative_goal_rad"],
            max_active_xy_discrepancy_units=float(active_xy[i, row_mask].max()),
            max_active_angle_discrepancy_rad=float(active_angle[i, row_mask].max()),
            active_prefix_passes=bool((active_xy[i, row_mask] <= MAX_XY).all()
                and (active_angle[i, row_mask] <= MAX_ANGLE).all()),
            replay_source={k: _plain(v[i]) for k, v in left.items()},
            replay_fine={k: _plain(v[i]) for k, v in right.items()})
        rows.append(row)
    return dict(cases=n, source_substeps=coarse["substeps"], fine_substeps=fine["substeps"],
        source_replay_bitwise=exact, source_scoring_reproduced=bool(labels_match),
        full_batch_and_order_preserved=True, identical_complete_start=True,
        planner_queries=0, world_model_queries=0, is_fine_closed_loop_experiment=False,
        prelude_handling="Original recorded prelude checked for finite states/walls; finer replay starts at identical full post-prelude snapshot.",
        scoring="Original first and final phase endpoints, last five real post-step states, original terminal time, finite and no wall through active prefix.",
        tolerances=dict(max_xy_units=MAX_XY, max_angle_rad=MAX_ANGLE),
        refinement={name: discrepancies(a, b, mask, group) for name, mask in masks.items()},
        stop_time_outcomes=outcomes, rows=rows,
        physical_diagnostics={str(run["substeps"]): dict(
            all_active_finite=bool((run["finite"] | ~masks["active_prefix"]).all()),
            active_wall_cases=int((run["touched_wall"] & masks["active_prefix"]).any(1).sum()),
            max_active_weld_xy_units=float(run["weld_xy"][masks["active_prefix"]].max()),
            max_active_weld_angle_rad=float(run["weld_angle"][masks["active_prefix"]].max())) for run in (coarse, fine)})


def audit_file(path, expected_hash, fine_substeps, output):
    """One independent source file; each contained physical batch stays intact."""
    torch.set_num_threads(1)
    path, output = Path(path), Path(output)
    if sha(path) != expected_hash:
        raise AssertionError("Source artifact changed before loading: "+str(path))
    artifact = torch.load(path, map_location="cpu")
    if "groups" not in artifact or not artifact["groups"]:
        raise ValueError("Expected a saved experiment containing nonempty groups")
    results, raw = [], []
    started = time.perf_counter()
    for group_index, group in enumerate(artifact["groups"]):
        nominal = int(group["rows"][0]["substeps"])
        if fine_substeps <= nominal:
            raise ValueError("Fine resolution must exceed the original source resolution")
        a, b = replay_group(group, nominal), replay_group(group, fine_substeps)
        result = summarize(group, a, b)
        result.update(source_path=str(path), source_sha256=expected_hash, group_index=group_index)
        results.append(result)
        raw.append(dict(source_path=str(path), group_index=group_index,
            source_rows=group["rows"], source_actions=group["actions"], source_active=group["active"],
            source_phases=group["phases"], initial_pose=group["initial_pose"],
            start=group["counterfactual_snapshots"][0], replay_source=a, replay_fine=b))
    if sha(path) != expected_hash:
        raise AssertionError("Source artifact changed during replay: "+str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path, report_path = output.with_suffix(".pt"), output.with_suffix(".json")
    torch.save(dict(results=raw), raw_path)
    report = dict(source_path=str(path), input_sha256=expected_hash, results=results,
        raw_path=str(raw_path), raw_sha256=sha(raw_path), wall_seconds=time.perf_counter()-started)
    report_path.write_text(json.dumps(_plain(report), indent=2, allow_nan=False)+"\n")
    return str(report_path)


def aggregate_results(results):
    grouped, changes = {}, []
    for result in results:
        for row in result["rows"]:
            key = (row["planner"], row["model_seed"])
            if key not in grouped:
                grouped[key] = dict(planner=key[0], model_seed=key[1], cases=0,
                    active_prefix_passes=0, source_ordered=0, fine_ordered=0,
                    source_first_dwell=0, fine_first_dwell=0,
                    source_final_dwell=0, fine_final_dwell=0, ordered_lost=0, ordered_gained=0)
            group = grouped[key]
            group["cases"] += 1
            group["active_prefix_passes"] += int(row["active_prefix_passes"])
            for label, metric in (("ordered", "ordered_goal_success"),
                ("first_dwell", "first_goal_dwell_success"), ("final_dwell", "final_goal_dwell_success")):
                group["source_"+label] += int(row["replay_source"][metric])
                group["fine_"+label] += int(row["replay_fine"][metric])
            old, new = row["replay_source"]["ordered_goal_success"], row["replay_fine"]["ordered_goal_success"]
            group["ordered_lost"] += int(old and not new)
            group["ordered_gained"] += int(new and not old)
            for metric in ("first_goal_dwell_success", "final_goal_dwell_success", "final_terminal_goal_success", "ordered_goal_success"):
                old, new = row["replay_source"][metric], row["replay_fine"][metric]
                if old != new:
                    changes.append(dict(source_path=result["source_path"], group_index=result["group_index"],
                        **{k: row[k] for k in ("row", "planner", "model_seed", "support", "initial_angle_deg", "nuisance_id", "first_goal_rad", "relative_goal_rad")},
                        metric=metric, change="gained" if new else "lost",
                        replay_source=row["replay_source"], replay_fine=row["replay_fine"]))
    return [grouped[k] for k in sorted(grouped, key=lambda x: (x[0], str(x[1])))], changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sources", nargs="+", type=Path)
    inputs.add_argument("--manifest", type=Path, help="Completed experiment JSON containing groups[*].raw_path/raw_sha256")
    parser.add_argument("--fine-substeps", type=int, default=320)
    parser.add_argument("--workers", type=int, default=1, help="Independent source-file workers, from1 through8")
    parser.add_argument("--out", type=Path, default=Path("data/novelty/interface_numerics"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.workers < 1 or args.workers > 8:
        raise ValueError("Use between one and eight independent workers")
    if args.out.with_suffix(".pt").exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError("Use a fresh output prefix to preserve numerical audit artifacts")
    paths = [Path(__file__)] + [Path("vmas_pusht")/(name+".py") for name in
        ("stable_task_numerics", "abstraction_convergence", "physics_profiles", "scenario",
         "boundary", "snapshot", "stable_composition", "abstraction_transfer")]
    sources = {str(path): sha(path) for path in paths}
    manifest_digest = None
    if args.manifest is not None:
        manifest_digest = sha(args.manifest)
        manifest = json.loads(args.manifest.read_text())
        args.sources = [Path(item["raw_path"]) for item in manifest["groups"]]
        for item in manifest["groups"]:
            if sha(item["raw_path"]) != item["raw_sha256"]:
                raise AssertionError("Experiment manifest input hash does not match: "+item["raw_path"])
    if not args.sources or len({p.resolve() for p in args.sources}) != len(args.sources):
        raise ValueError("Source files must be nonempty and unique")
    source_hashes = {str(path): sha(path) for path in args.sources}
    report = dict(source_sha256=sources, input_sha256=source_hashes, results=[],
        manifest_path=None if args.manifest is None else str(args.manifest), manifest_sha256=manifest_digest,
        worker_count=args.workers, expected_source_files=len(args.sources), completed_source_files=0,
        complete=False, raw_parts=[],
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        versions={key: importlib.metadata.version(key) for key in ("vmas", "torch", "numpy")},
        interpretation="Fixed-action refinement with original phase stopping times; not a fine-resolution closed-loop planner rerun.")
    started = time.perf_counter()
    completed = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def consume(index, report_path):
        part = json.loads(Path(report_path).read_text())
        part["report_path"] = report_path
        part["report_sha256"] = sha(report_path)
        completed[index] = part
        report["results"] = [item for i in sorted(completed) for item in completed[i]["results"]]
        report["raw_parts"] = [{k: completed[i][k] for k in ("source_path", "raw_path", "raw_sha256", "report_path", "report_sha256")} for i in sorted(completed)]
        report["completed_source_files"] = len(completed)
        report["wall_seconds"] = time.perf_counter()-started
        report["summary"], report["changed_cases"] = aggregate_results(report["results"])
        args.out.with_suffix(".json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False)+"\n")
        for result in part["results"]:
            print(json.dumps(dict(source=result["source_path"], group=result["group_index"], cases=result["cases"],
                completed_sources=len(completed), total_sources=len(args.sources),
                exact=result["source_replay_bitwise"], labels_match=result["source_scoring_reproduced"],
                errors=result["refinement"]["active_prefix"], outcomes=result["stop_time_outcomes"])), flush=True)
    jobs = [(i, path, args.out.with_name(args.out.name+f"_part{i:03d}")) for i, path in enumerate(args.sources)]
    if args.workers == 1:
        for i, path, output in jobs:
            consume(i, audit_file(path, source_hashes[str(path)], args.fine_substeps, output))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = {pool.submit(audit_file, path, source_hashes[str(path)], args.fine_substeps, output): i for i, path, output in jobs}
            for future in as_completed(pending):
                consume(pending[future], future.result())
    raw = [item for i in sorted(completed) for item in torch.load(completed[i]["raw_path"], map_location="cpu")["results"]]
    raw_path = args.out.with_suffix(".pt")
    torch.save(dict(results=raw), raw_path)
    report.update(raw_path=str(raw_path), raw_sha256=sha(raw_path))
    for path, digest in {**sources, **source_hashes}.items():
        if sha(path) != digest:
            raise AssertionError("Frozen source/input artifact changed during numerical replay: "+path)
    report["verified_hashes_at_completion"] = True
    if args.manifest is not None and sha(args.manifest) != manifest_digest:
        raise AssertionError("Input experiment manifest changed during numerical replay")
    report["source_replay_valid"] = all(item["source_replay_bitwise"] and item["source_scoring_reproduced"] for item in report["results"])
    report["complete"] = True
    report["wall_seconds"] = time.perf_counter()-started
    args.out.with_suffix(".json").write_text(json.dumps(_plain(report), indent=2, allow_nan=False)+"\n")
    if not report["source_replay_valid"]:
        raise AssertionError("Source replay/scoring failed; all artifacts retained for diagnosis")


if __name__ == "__main__":
    main()
