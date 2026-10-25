"""Complete frozen predictor by interface matrix and matched-budget controls.

All six unrestricted cells and all three predetermined 39-candidate allocations
are regenerated from initial scenes. No model is fitted or selected here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from functools import partial
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.boundary import _plain
from vmas_pusht.interface_composition import run_group, summarize
from vmas_pusht.interface_experiment import GOALS, aggregate
from vmas_pusht.interface_factorial_planner import BASE_PREDICTORS, POLICIES, allocated_table, select_options
from vmas_pusht.option_models import load_model_pack, sha


def arm_name(predictor, policy):
    if predictor not in BASE_PREDICTORS or policy not in POLICIES:
        raise ValueError("Unknown factorial arm")
    return predictor if policy == "expanded" else predictor+"_"+policy


def jobs(predictors=BASE_PREDICTORS, policies=POLICIES, model_seeds=(0, 1, 2)):
    if any(not values or len(values) != len(set(values)) for values in (predictors, policies, model_seeds)):
        raise ValueError("Factor values must be nonempty and unique")
    if set(predictors)-set(BASE_PREDICTORS) or set(policies)-set(POLICIES):
        raise ValueError("Unknown factorial factor")
    return [(predictor, policy, seed, goal)
        for predictor in sorted(predictors, key=lambda x: x != "true_option")
        for policy in policies
        for seed in ((0,) if predictor == "true_option" else model_seeds)
        for goal in GOALS]


def relabel(group, predictor, policy):
    """Distinguish audit arms without changing which frozen model was queried."""
    name = arm_name(predictor, policy)
    for row in group["rows"]:
        row.update(base_planner=predictor, planner=name, candidate_policy=policy)
    for item in group["option_records"]:
        item.update(base_planner=predictor, planner=name, candidate_policy=policy)
    for snapshot in group["counterfactual_snapshots"]:
        for row in snapshot["rows"]:
            row.update(base_planner=predictor, planner=name, candidate_policy=policy)
    group["factorial_arm"] = dict(predictor=predictor, policy=policy, name=name)
    return group


def worker(model_path, model_digest, output, predictor, policy, seed, goal, substeps, source_hashes):
    torch.set_num_threads(1)
    name = arm_name(predictor, policy)
    prefix = Path(f"{output}_{name}_seed{seed}_{goal}")
    if any(prefix.with_suffix(suffix).exists() for suffix in (".pt", ".json", ".log")):
        raise FileExistsError("Preserve existing factorial artifacts: "+str(prefix))
    for path, digest in {str(model_path): model_digest, **source_hashes}.items():
        if sha(path) != digest:
            raise AssertionError("Registered input changed before evaluation: "+path)
    pack = None if predictor == "true_option" else load_model_pack(model_path)
    started = time.perf_counter()
    with prefix.with_suffix(".log").open("w") as stream, redirect_stdout(stream):
        group = run_group(substeps=substeps, model_pack=pack, planners=[predictor],
            model_seed=seed, goal_pair=GOALS[goal], observe=True,
            option_selector=partial(select_options, policy=policy),
            expanded_final_interface=policy != "direct")
    group = relabel(group, predictor, policy)
    if sha(model_path) != model_digest:
        raise AssertionError("Frozen model artifact changed during execution")
    raw_path = prefix.with_suffix(".pt")
    torch.save(dict(groups=[group], source_sha256=source_hashes,
        model_path=str(model_path), model_sha256=model_digest), raw_path)
    report = dict(planner=name, base_planner=predictor, candidate_policy=policy,
        model_seed=seed, goal=goal, substeps=substeps, rows=group["rows"],
        summary=summarize(group["rows"]), raw_path=str(raw_path), raw_sha256=sha(raw_path),
        elapsed_seconds=time.perf_counter()-started, model_sha256=model_digest,
        option_count=len(group["option_records"]))
    path = prefix.with_suffix(".json")
    path.write_text(json.dumps(_plain(report), indent=2, allow_nan=False)+"\n")
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("data/novelty/option_interface_local_models.pt"))
    parser.add_argument("--previous", type=Path, default=Path("data/novelty/interface_complete_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/novelty/interface_factorial_v1"))
    parser.add_argument("--predictors", nargs="+", choices=BASE_PREDICTORS, default=list(BASE_PREDICTORS))
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--substeps", type=int, default=160)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8 or args.substeps < 1:
        raise ValueError("Use 1 to 8 workers and positive substeps")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schedule = jobs(tuple(args.predictors), tuple(args.policies), tuple(args.model_seeds))
    model_digest = sha(args.models)
    previous_digest = sha(args.previous)
    previous = json.loads(args.previous.read_text())
    if previous["model_sha256"] != model_digest:
        raise AssertionError("The factorial must use the previously frozen model pack")
    paths = [Path(__file__)] + [Path("vmas_pusht")/(name+".py") for name in
        ("interface_factorial_planner", "interface_composition", "interface_experiment", "option_planner",
         "option_models", "action_interface", "recovery_primitives", "recovery_attribution",
         "rebrace", "snapshot", "scenario", "physics_profiles", "physical_abstraction",
         "abstraction_transfer", "stable_composition")]
    hashes = {str(path): sha(path) for path in paths}
    old_hashes = previous["old_model_sha256"]
    declaration = dict(model_path=str(args.models), model_sha256=model_digest,
        previous_path=str(args.previous), previous_sha256=previous_digest,
        source_sha256=hashes, old_model_sha256=old_hashes, job_schedule=schedule,
        total_cases=18*len(schedule), batch_size_per_group=18, substeps=args.substeps,
        physical_cases_per_seed=36, goals=GOALS, supports=["passive", "world_anchor", "preloaded_anchor"],
        angles_degrees=[0.,30.], nuisances=[[0.,0.,0.],[.045,0.,.3],[-.045,0.,-.3]],
        prelude_steps=10, stage_budgets=[60,20,80,60], goal_dwell=5,
        force_norm_bound=1., angle_tolerance=.015, translation_tolerance=.05,
        predictor_parameters=dict(wm_full=1949, wm_large=22149, true_option=None),
        no_training_or_task_based_model_selection=True,
        task="Complete rotation-retreat-rebrace-rotation from the original initial scene",
        score="Unchanged one-option endpoint objective and original ordered five-step dwell scoring",
        first_stage="Original direct39 legal action bank for every arm; unrestricted mixed-phase batches may evaluate117 with routes masked, preserving the historical selector",
        budget39=dict(allocations={str(s): allocated_table(s).tolist() for s in range(3)},
            assignment="(duration_index + zero_based_nonzero_force_index + shift) modulo 3; all zero pushes direct",
            routes_per_table=[15,12,12], exactly_preserved="All 39 direction/force/duration programs",
            fallback_to_direct39="First stage, acquired reverse face, or immediately unavailable route",
            budget_scope="39 distinct candidate slots per decision, not equal episode compute or oracle simulation steps",
            training_coverage="Every allocated atomic option was already in the frozen local training bank"),
        pairing="Within predictor/fit seed; audit physical prefixes before final rotation",
        interpretation="Finite action bank, one-option lookahead, fixed endpoint objective; no claim of global unreachability or IID population success.")
    registration = args.output.with_name(args.output.name+"_registration.json")
    if registration.exists():
        if json.loads(registration.read_text()) != json.loads(json.dumps(declaration)):
            raise ValueError("Registration changed: use a fresh output prefix")
    else:
        registration.write_text(json.dumps(declaration, indent=2)+"\n")
    if args.register_only:
        print(str(registration)); return
    if args.output.with_suffix(".json").exists():
        raise FileExistsError("Preserve completed factorial results")
    started = time.perf_counter()
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(worker, args.models, model_digest, args.output, p, policy, seed, goal,
                               args.substeps, hashes) for p, policy, seed, goal in schedule]
        for future in as_completed(futures):
            path = Path(future.result())
            result = json.loads(path.read_text())
            completed.append(dict(report_path=str(path), report_sha256=sha(path), **result))
            print(json.dumps(dict(planner=result["planner"], seed=result["model_seed"], goal=result["goal"],
                first=sum(r["first_goal_success_at_phase_boundary"] for r in result["rows"]),
                ordered=sum(r["ordered_goal_success"] for r in result["rows"]),
                completed_groups=len(completed), total_groups=len(schedule),
                wall_seconds=time.perf_counter()-started)), flush=True)
    completed.sort(key=lambda x: (x["planner"], x["model_seed"], x["goal"]))
    rows = [row for group in completed for row in group["rows"]]
    keys = [(r["planner"],r["model_seed"],r["support"],r["initial_angle_deg"],r["nuisance_id"],r["first_goal_rad"]) for r in rows]
    if len(rows) != declaration["total_cases"] or len(set(keys)) != len(rows):
        raise AssertionError("Incomplete or duplicated factorial matrix")
    for path, digest in {str(args.models):model_digest,str(args.previous):previous_digest,**hashes,**old_hashes}.items():
        if sha(path) != digest:
            raise AssertionError("Frozen input changed during factorial evaluation: "+path)
    report = dict(declaration, registration_path=str(registration), registration_sha256=sha(registration),
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        rows=rows, summary=aggregate(rows), detailed_summary=summarize(rows),
        groups=[{k:g[k] for k in ("planner","model_seed","goal","raw_path","raw_sha256","report_path","report_sha256")} for g in completed],
        verified_frozen_inputs=True, wall_seconds=time.perf_counter()-started)
    args.output.with_suffix(".json").write_text(json.dumps(_plain(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output),cases=len(rows),summary=report["summary"])),flush=True)


if __name__ == "__main__":
    main()
