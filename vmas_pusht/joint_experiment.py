"""Frozen fixed/planned helper x contact-interface experiment from full starts."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import torch

from vmas_pusht.boundary import _plain
from vmas_pusht.interface_composition import run_group
from vmas_pusht.interface_experiment import GOALS
from vmas_pusht.joint_interface import HELPERS, OFFSETS, JointActionInterface
from vmas_pusht.joint_models import COUNTS, FrozenJointPredictor
from vmas_pusht.joint_planner import select_options
from vmas_pusht.option_models import sha


ARMS = tuple((planned, expanded, shift)
             for planned in (False, True)
             for expanded in (False, True)
             for shift in ((0, 1, 2) if expanded else (0,)))
SOURCE_FILES = ("joint_interface", "joint_models", "joint_planner", "joint_experiment",
    "interface_composition", "action_interface", "recovery_primitives", "rebrace",
    "option_models", "option_planner", "stable_composition", "stable_local_wm",
    "physical_abstraction", "snapshot", "scenario", "physics_profiles", "abstraction_models",
    "abstraction_transfer", "recovery_attribution", "boundary", "abstraction_composition",
    "interface_experiment")


def arm_name(planned, expanded, shift):
    return ("joint" if planned else "fixed") + (f"_expanded_s{shift}" if expanded else "_direct")


def schedule():
    return [(predictor, seed, goal, *arm) for predictor in ("oracle", "learned")
            for seed in ((0,) if predictor == "oracle" else (0, 1, 2))
            for goal in GOALS for arm in ARMS]


def protocol():
    return dict(version=1, question="Does contact-interface coverage still limit a planner choosing BOTH agents' options?",
        helpers=list(HELPERS), helper_anchor_offsets=list(OFFSETS),
        helper_semantics="Concurrent bounded physical control during both rotation phases; common scripted prelude, retreat and recovery retained.",
        primary_programs=39, fixed_candidate_budget=39, joint_candidate_budget=273,
        route_assignment="(duration_index + nonzero_force_index + helper_id + allocation) % 3; zero pushes direct",
        allocations=[0, 1, 2], candidate_match="Direct/expanded exactly preserve the primary force/duration/helper multiset; no claim of equal fixed-vs-joint compute.",
        fallback="Full direct bank on current face if routes unavailable; same slot count. Five-step fixed-support zero primary if no feasible prediction.",
        physics=dict(profile="dry_light", substeps=160, verification_substeps=320, force_norm_bound_per_agent=1.),
        local_data=dict(context_counts_per_support=COUNTS, context_seed=92617, supports=["passive", "world_anchor", "preloaded_anchor"],
            helper_continuation="Every second prepared source executes an indexed goal-free five-step helper/primary option before collection.",
            full_task_labels=False, goals_in_inputs=False, helper_ids="all seven in every partition; tests hold out contexts or primary actions"),
        training=dict(width=128, steps=2000, batch_size=512, lr=.003, seeds=[0, 1, 2],
            shared_across_supports=True, normalization="train only", checkpoint="fixed final update", duration_margin="validation q95 absolute residual"),
        tasks=dict(goal_pairs=GOALS, supports=["passive", "world_anchor", "preloaded_anchor"], angles=[0, 30],
            nuisances=[[0, 0, 0], [.045, 0, .3], [-.045, 0, -.3]], stage_budgets=[60, 20, 80, 60],
            prelude_steps=10, angle_tolerance=.015, translation_tolerance=.05, consecutive_goal_states=5),
        primary_outcome="Complete ordered-goal success from the initial scene, retaining all failed first stages",
        numerical_rule="Replay identical recorded actions from the full post-prelude snapshot at 160/320, original dwell windows and stop times; report all label changes.",
        limits="Handwritten low-level options; centralized high-level joint planning; one primary side change; not learned helper discovery, LeWM, or general coalition necessity.",
        job_schedule=schedule(), cases=1152)


def source_hashes():
    return {f"vmas_pusht/{name}.py": sha(f"vmas_pusht/{name}.py") for name in SOURCE_FILES}


def verify_inputs(frozen):
    for path, digest in frozen["source_sha256"].items():
        if sha(path) != digest:
            raise AssertionError("Registered source changed: "+path)
    if sha(frozen["model_path"]) != frozen["model_sha256"]:
        raise AssertionError("Frozen model changed")


def worker(root, frozen, job):
    torch.set_num_threads(1)
    verify_inputs(frozen)
    predictor, seed, goal, planned, expanded, shift = job
    arm = arm_name(planned, expanded, shift)
    name = f"{predictor}_{arm}"
    prefix = Path(root)/"tasks"/f"{name}_seed{seed}_{goal}"
    if prefix.with_suffix(".pt").exists() or prefix.with_suffix(".json").exists():
        raise FileExistsError("Preserve completed tasks: "+str(prefix))
    model = None
    if predictor == "learned":
        pack = torch.load(frozen["model_path"], weights_only=False)
        model = FrozenJointPredictor(pack["checkpoints"][seed])
    selector = partial(select_options, predictor=model, planned_helper=planned,
                       expanded=expanded, allocation=shift)
    started = time.perf_counter()
    group = run_group(planners=["true_option"], model_seed=seed, goal_pair=GOALS[goal],
        observe=True, option_selector=selector, expanded_final_interface=expanded,
        interface_factory=JointActionInterface)
    for i, row in enumerate(group["rows"]):
        own = [r for r in group["option_records"] if r["row"] == i]
        row.update(planner=name, predictor=predictor, planned_helper=planned,
            candidate_policy=arm, allocation=shift, model_kind="joint_local_endpoint" if model else "true_option",
            candidate_slots=273 if planned else 39,
            helper_choices=dict(Counter(HELPERS[int(r["option"][4])] for r in own)),
            helper_distinct_options=len({int(r["option"][4]) for r in own}),
            helper_switches=sum(a["option"][4] != b["option"][4] for a, b in zip(own, own[1:])),
            helper_nonfixed_options=sum(r["option"][4] != 0 for r in own))
    for item in group["option_records"]:
        item["planner"] = name
    group["resource_accounting"].update(helper_is_planned=planned,
        helper_option_count=len(HELPERS) if planned else 1,
        joint_predictor_parameters=sum(p.numel() for p in model.model.parameters()) if model else 0,
        post_handoff_helper_memory_floats=4, post_handoff_helper_memory_integers=1,
        fixed_retreat_recovery=True, same_candidate_budget_across_interfaces=True)
    verify_inputs(frozen)
    torch.save(dict(groups=[group], model_sha256=frozen["model_sha256"], source_sha256=frozen["source_sha256"]), prefix.with_suffix(".pt"))
    result = dict(job=job, name=name, rows=group["rows"],
        path=str(prefix.with_suffix(".pt")), sha256=sha(prefix.with_suffix(".pt")),
        seconds=time.perf_counter()-started,
        ordered_success=sum(r["ordered_goal_success"] for r in group["rows"]), cases=len(group["rows"]))
    prefix.with_suffix(".json").write_text(json.dumps(_plain(result), indent=2, allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    registration = root/"registration.json"
    if args.register_only:
        if registration.exists():
            raise FileExistsError("Preserve preregistration")
        registration.write_text(json.dumps(protocol(), indent=2)+"\n")
        print(registration)
        return
    if json.loads(registration.read_text()) != json.loads(json.dumps(protocol())):
        raise AssertionError("Declared protocol changed")
    models = root/"local"/"models.pt"
    frozen = dict(model_path=str(models), model_sha256=sha(models), source_sha256=source_hashes(),
        registration_sha256=sha(registration), git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        versions={name: importlib.metadata.version(name) for name in ("torch", "vmas", "numpy")})
    freeze_path = root/"frozen.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text())
        if previous != frozen:
            raise AssertionError("Cannot resume changed evaluation inputs")
    else:
        freeze_path.write_text(json.dumps(frozen, indent=2)+"\n")
    (root/"tasks").mkdir(exist_ok=True)
    results, pending = [], []
    for job in schedule():
        predictor, seed, goal, planned, expanded, shift = job
        path = root/"tasks"/f"{predictor}_{arm_name(planned, expanded, shift)}_seed{seed}_{goal}.json"
        if path.exists():
            value = json.loads(path.read_text())
            if sha(value["path"]) != value["sha256"]:
                raise AssertionError("Completed task hash changed")
            results.append(value)
        else:
            pending.append(job)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(worker, root, frozen, job) for job in pending]
        for future in as_completed(futures):
            value = future.result(); results.append(value)
            print(json.dumps({key: value[key] for key in ("job", "ordered_success", "cases", "seconds")}), flush=True)
    verify_inputs(frozen)
    results.sort(key=lambda r: str(r["job"]))
    manifest = dict(frozen=frozen, protocol=protocol(), groups=results,
                    cases=sum(r["cases"] for r in results), wall_seconds=time.perf_counter()-started)
    (root/"manifest.json").write_text(json.dumps(_plain(manifest), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
