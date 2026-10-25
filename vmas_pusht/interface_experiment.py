"""Prespecified full-task matrix for the completed contact-option interface.

The learned models are frozen before any task execution. Independent planner,
seed and goal groups run in separate CPU processes; both initial conditions and
the complete four-stage task are regenerated, rather than replacing only a
saved handoff continuation. Every raw group retains its actual batch/order.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.boundary import _plain
from vmas_pusht.interface_composition import run_group, summarize
from vmas_pusht.option_planner import MODEL_NAMES, PLANNERS, protocol
from vmas_pusht.option_models import load_model_pack, sha


GOALS = dict(standard=(.04, .08), holdout=(.06, .12))


def jobs(planners=PLANNERS, model_seeds=(0, 1, 2)):
    """Run physical baselines once; repeat learned policies over fit seeds."""
    if len(set(planners)) != len(planners) or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("Duplicate planners/seeds would double-count cases")
    if not planners or not model_seeds or set(planners)-set(PLANNERS):
        raise ValueError("Expected declared planners and nonempty model seeds")
    # Start expensive simulator controls early while learned groups use the
    # remaining workers. Scheduling does not change either group's controls.
    order = sorted(planners, key=lambda p: (p != "true_option", p in MODEL_NAMES, p))
    return [(p, seed, goal) for p in order
            for seed in (model_seeds if p in MODEL_NAMES else (0,)) for goal in GOALS]


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["planner"], row["model_seed"])].append(row)
    return [dict(planner=key[0], model_seed=key[1], cases=len(selected),
        first_goal_success=sum(r["first_goal_success_at_phase_boundary"] for r in selected),
        final_goal_success=sum(r["final_goal_dwell_success"] for r in selected),
        ordered_success=sum(r["ordered_goal_success"] for r in selected),
        interface_success=sum(r["interface_completion_success"] for r in selected),
        mean_total_steps=sum(r["total_steps"] for r in selected)/len(selected),
        route_episodes=sum(r["route_options_started"] > 0 for r in selected),
        route_acquisitions=sum(r.get("route_contacts_acquired", 0) for r in selected),
        failed_options=sum(r["failed_options"] for r in selected),
        finite_all=all(r["finite"] for r in selected), wall_cases=sum(r["wall"] for r in selected))
        for key, selected in sorted(groups.items())]


def worker(model_path, model_digest, output, planner, seed, goal, substeps, source_hashes):
    torch.set_num_threads(1)
    prefix = Path(f"{output}_{planner}_seed{seed}_{goal}")
    if prefix.with_suffix(".pt").exists() or prefix.with_suffix(".json").exists():
        raise FileExistsError(f"Preserve existing evaluation: {prefix}")
    if sha(model_path) != model_digest:
        raise AssertionError("Frozen model artifact changed before worker loading")
    pack = load_model_pack(model_path) if planner in MODEL_NAMES else None
    if sha(model_path) != model_digest:
        raise AssertionError("Frozen model artifact changed while loading")
    started = time.perf_counter()
    with prefix.with_suffix(".log").open("w") as stream, redirect_stdout(stream):
        group = run_group(substeps=substeps, model_pack=pack, planners=[planner],
            model_seed=seed, goal_pair=GOALS[goal], observe=True)
    if sha(model_path) != model_digest:
        raise AssertionError("Frozen model artifact changed during task execution")
    raw = prefix.with_suffix(".pt")
    torch.save(dict(groups=[group], source_sha256=source_hashes,
        model_path=str(model_path), model_sha256=model_digest), raw)
    report = dict(planner=planner, model_seed=seed, goal=goal, substeps=substeps,
        rows=group["rows"], summary=summarize(group["rows"]), raw_path=str(raw), raw_sha256=sha(raw),
        elapsed_seconds=time.perf_counter()-started,
        option_count=len(group["option_records"]), model_path=str(model_path), model_sha256=model_digest)
    report_path = prefix.with_suffix(".json")
    report_path.write_text(json.dumps(_plain(report), indent=2, allow_nan=False)+"\n")
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("data/novelty/option_interface_local_models.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/novelty/interface_complete_v1"))
    parser.add_argument("--planners", nargs="+", choices=PLANNERS, default=list(PLANNERS))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--substeps", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.substeps < 1:
        raise ValueError("Positive worker count and substeps required")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schedule = jobs(args.planners, tuple(args.model_seeds))
    declaration = dict(protocol=protocol(), job_schedule=schedule,
        model_path=str(args.models), substeps=args.substeps, batch_size_per_group=18,
        total_cases=len(schedule)*18, full_task_from_initial_scene=True,
        angle_degrees=[0., 30.], nuisance_conditions=[[0.,0.,0.],[.045,0.,.3],[-.045,0.,-.3]],
        supports=["passive", "world_anchor", "preloaded_anchor"], goal_pairs=GOALS,
        prelude_steps=10, stage_budgets=[60,20,80,60], goal_dwell=5,
        interpretation="Model seeds measure fit variation on repeated physical cases; no population confidence interval.")
    registration = args.output.with_name(args.output.name+"_registration.json")
    if not registration.exists():
        registration.write_text(json.dumps(declaration, indent=2)+"\n")
    elif json.dumps(json.loads(registration.read_text()), sort_keys=True) != json.dumps(declaration, sort_keys=True):
        raise ValueError("Registration changed: use a fresh output prefix")
    if args.register_only:
        print(str(registration)); return
    if args.output.with_suffix(".json").exists():
        raise FileExistsError("Preserve the completed evaluation with a fresh output prefix")
    paths = [Path(__file__)] + [Path("vmas_pusht")/(name+".py") for name in
        ("interface_composition", "option_planner", "option_models", "action_interface",
         "recovery_primitives", "recovery_attribution", "rebrace", "snapshot", "scenario",
         "physics_profiles", "physical_abstraction", "abstraction_transfer", "stable_composition")]
    hashes = {str(p):sha(p) for p in paths}
    model_digest = sha(args.models)
    old_models = [Path("data/novelty")/(name+".pt") for name in
        ("stable_local_original_models", "stable_local_selected_models")]
    old_hashes = {str(p):sha(p) for p in old_models}
    started = time.perf_counter()
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(worker,args.models,model_digest,args.output,p,seed,goal,args.substeps,hashes)
                   for p,seed,goal in schedule]
        for future in as_completed(futures):
            path = Path(future.result())
            result = json.loads(path.read_text())
            completed.append(dict(report_path=str(path),report_sha256=sha(path),**result))
            print(json.dumps(dict(planner=result["planner"],seed=result["model_seed"],goal=result["goal"],
                cases=len(result["rows"]),first=sum(r["first_goal_success_at_phase_boundary"] for r in result["rows"]),
                ordered=sum(r["ordered_goal_success"] for r in result["rows"]),
                finished_groups=len(completed),total_groups=len(schedule),wall_seconds=time.perf_counter()-started)),flush=True)
    completed.sort(key=lambda g:(g["planner"],g["model_seed"],g["goal"]))
    rows=[r for g in completed for r in g["rows"]]
    keys=[(r["planner"],r["model_seed"],r["support"],r["initial_angle_deg"],r["nuisance_id"],r["first_goal_rad"]) for r in rows]
    if len(rows)!=declaration["total_cases"] or len(set(keys))!=len(rows):
        raise AssertionError("Incomplete or duplicated experiment matrix")
    for path,digest in {str(args.models):model_digest,**hashes,**old_hashes}.items():
        if sha(path)!=digest:raise AssertionError("Frozen input changed: "+path)
    report=dict(declaration,git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_sha256=hashes,model_sha256=model_digest,old_model_sha256=old_hashes,
        registration_path=str(registration),registration_sha256=sha(registration),
        rows=rows,summary=aggregate(rows),detailed_summary=summarize(rows),
        groups=[{k:g[k] for k in ("planner","model_seed","goal","raw_path","raw_sha256","report_path","report_sha256")} for g in completed],
        wall_seconds=time.perf_counter()-started)
    args.output.with_suffix(".json").write_text(json.dumps(_plain(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output),summary=report["summary"],verified_frozen_sources=True)),flush=True)


if __name__=="__main__":
    main()
