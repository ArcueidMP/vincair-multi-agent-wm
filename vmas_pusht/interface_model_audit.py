"""Common-state frozen-model audit on oracle-selected complete local options.

This is read-only post hoc diagnosis. It replays saved actions in each original
batch to reconstruct current observations and genuine history, then scores all
architectures/seeds on the SAME true-simulator counterfactual option outcomes.
An interrupted real option may still supply a complete label because the
recorded true_option prediction came from a separate complete simulator branch.
No model is fit and no new task or candidate rollout is generated here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch

from vmas_pusht.action_interface import model_inputs
from vmas_pusht.abstraction_transfer import ANGLE_TOLERANCE, TRANSLATION_TOLERANCE
from vmas_pusht.boundary import _plain
from vmas_pusht.option_models import MODELS, load_model_pack
from vmas_pusht.physical_abstraction import coarse_state, physical_state
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.scenario import wrap
from vmas_pusht.snapshot import restore
from vmas_pusht.stable_composition import history_inputs, snapshot_from_record


TENSOR_FIELDS = ("coarse", "full", "history", "truth", "query_pose", "initial_pose", "goals")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_from_physical(states):
    """Exact documented coarse geometry from all four entities' saved states."""
    if states.shape[-1] != 24:
        raise ValueError("Expected full24D physical states")
    return torch.cat((states[..., 0:2], states[..., 6:8], states[..., 12:14],
        states[..., 16:17].sin(), states[..., 16:17].cos(), states[..., 18:20],
        states[..., 22:23].sin(), states[..., 22:23].cos()), -1)


def _memory_at(group, step, start):
    if step == 0:
        return start.get("interface_memory")
    if "controller_memory" in group:
        return group["controller_memory"][step-1]["interface"]
    return None


@torch.no_grad()
def extract_oracle_queries(group, *, source_id="oracle", reference_substeps=160):
    """Replay the intact original batch and collect only source oracle labels."""
    n = len(group["rows"])
    if any(r["planner"] != "true_option" for r in group["rows"]):
        raise ValueError("Common labels must come from true_option, not WM guesses")
    trajectory, actions = group["trajectory"], group["actions"]
    if trajectory.shape != (n, actions.shape[1]+1, 24) or actions.shape[2:] != (2, 2):
        raise ValueError("Saved full-state trajectory and actions do not align")
    starts = [r for r in group["counterfactual_snapshots"]
              if all(row["snapshot_stage"] == "post_prelude" for row in r["rows"])]
    if len(starts) != 1 or starts[0]["snapshot"]["batch_size"] != n:
        raise ValueError("An intact original post-prelude snapshot is required")
    start = starts[0]
    if [r["composition_row"] for r in start["rows"]] != list(range(n)):
        raise ValueError("The original batch row order must be retained")
    env = make_profile(group["physics_profile"], num_envs=n,
                       substeps=int(group["rows"][0]["substeps"]), seed=101)
    restore(env, snapshot_from_record(start))
    if not torch.equal(physical_state(env), trajectory[:, 0]):
        raise AssertionError("Original initial snapshot does not reproduce full state")
    prelude = group["prelude_trajectory"]
    if not torch.equal(prelude[:, -1], trajectory[:, 0]):
        raise ValueError("Real prelude does not join the task trajectory")
    geometry = list(geometry_from_physical(prelude).unbind(1))
    history_actions = [a.clone() for a in group["prelude_actions"].unbind(1)]
    if not torch.equal(history_inputs(geometry, history_actions), start["history"]):
        raise AssertionError("Reconstructed prelude history differs from the saved real history")
    if not torch.equal(geometry[-1], coarse_state(env)):
        raise AssertionError("Recorded physical-to-geometry mapping differs from live geometry")
    by_step = defaultdict(list)
    for record_index, record in enumerate(group["option_records"]):
        if record["planner"] != "true_option":
            raise ValueError("Mixed planner labels in oracle option records")
        by_step[int(record["start_step"])].append((record_index, record))
    query_by_step = {int(q["step"]):q for q in group["queries"]}
    if len(query_by_step) != len(group["queries"]):
        raise ValueError("Repeated query-step diagnostics")
    tensors, rows, excluded, complete_replay_errors = defaultdict(list), [], [], []
    checked_steps, checked_memory, checked_snapshots = 0, 0, 0
    for step in range(actions.shape[1]):
        if step in by_step:
            records = by_step[step]
            ids = [int(r["row"]) for _, r in records]
            if len(ids) != len(set(ids)) or step not in query_by_step:
                raise ValueError("Missing query diagnostics or repeated row query at a step")
            q = query_by_step[step]
            memory = _memory_at(group, step, start)
            anchor = group["helper_anchor"][:, step].clone()
            modes = torch.full((n,), 3, dtype=torch.long)
            options = torch.zeros(n, 4); options[:, 3] = 5
            for _, record in records:
                i = int(record["row"])
                if not bool(q["query_mask"][i]):
                    raise ValueError("Option record was not queried")
                if q["diagnostics"]["source"][i] != "true_simulator":
                    raise ValueError("Stored prediction is not a simulator counterfactual")
                if float(env.steps[i]) != record["start_env_step"]:
                    raise AssertionError("Query clock differs from original actual clock")
                if not torch.equal(env.scenario.t_pose()[i], torch.tensor(record["start_pose"])):
                    raise AssertionError("Query pose differs from saved option start")
                modes[i] = int(record["start_contact_mode"])
                options[i] = torch.tensor(record["option"])
                if not torch.equal(options[i], q["options"][i]):
                    raise ValueError("Option record differs from selected query option")
                if memory is not None:
                    if int(memory["primitive"]["mode"][i]) != int(modes[i]):
                        raise AssertionError("Recorded pre-query contact mode differs from actual memory")
                    if not torch.equal(memory["anchor"][i], anchor[i]):
                        raise AssertionError("Recorded pre-query helper anchor differs from actual memory")
                    checked_memory += 1
            coarse, full = model_inputs(env, anchor, modes, options,
                                        reference_substeps=reference_substeps)
            past = history_inputs(geometry, history_actions)
            for record_index, record in records:
                i = int(record["row"])
                source_row = group["rows"][i]
                metadata = dict(source_id=source_id, record_index=record_index,
                    episode_row=i, episode_id=f"{source_id}/{i}",
                    support=source_row["support"], task_phase=record["task_phase"],
                    route_id=int(record["option"][0]), start_contact_mode=int(record["start_contact_mode"]),
                    initial_angle_deg=source_row["initial_angle_deg"], nuisance_id=source_row["nuisance_id"],
                    first_goal_rad=source_row["first_goal_rad"], final_goal_rad=source_row["relative_goal_rad"],
                    start_step=step, start_env_step=record["start_env_step"], option=record["option"],
                    real_interrupted=bool(record["interrupted"]), real_completed=bool(record["completed"]),
                    real_elapsed_steps=record["elapsed_steps"], actual_termination_reason=record["termination_reason"])
                label = record.get("predicted_outcome")
                if label is None:
                    excluded.append(dict(metadata, exclusion="fallback_without_complete_simulator_label"))
                    continue
                truth = torch.tensor(label)
                if truth.shape != (5,) or not torch.isfinite(truth).all() or truth[3] < 0:
                    raise ValueError("Invalid complete simulator option label")
                if record.get("chosen_index") is None or record["chosen_index"] < 0:
                    raise ValueError("A fallback cannot be credited with a complete simulator label")
                if not torch.equal(truth, q["diagnostics"]["selected_prediction"][i]):
                    raise AssertionError("Stored oracle label differs from raw selected simulator outcome")
                metadata.update(label_source="complete_true_simulator_option_branch",
                    full_counterfactual_seconds=float(truth[3]),
                    full_counterfactual_completed=bool(truth[4] >= .5))
                if not record["interrupted"]:
                    actual = torch.tensor([*record["actual_response"], record["elapsed_seconds"], float(record["completed"])])
                    complete_replay_errors.append(dict(source_id=source_id, record_index=record_index,
                        error=(truth-actual).tolist()))
                rows.append(metadata)
                for key, value in dict(coarse=coarse[i], full=full[i], history=past[i], truth=truth,
                        query_pose=env.scenario.t_pose()[i], initial_pose=group["initial_pose"][i],
                        goals=torch.tensor(source_row["first_goal_rad"] if record["task_phase"] == "rotate_first" else source_row["relative_goal_rad"])).items():
                    tensors[key].append(value.clone())
        command = actions[:, step]
        env.step([command[:, 0].clone(), command[:, 1].clone()])
        if not torch.equal(physical_state(env), trajectory[:, step+1]):
            error = float((physical_state(env)-trajectory[:, step+1]).abs().max())
            raise AssertionError(f"Original batch replay diverged at step{step+1}: {error}")
        geometry.append(coarse_state(env).clone()); history_actions.append(command.clone())
        checked_steps += 1
        # Independently saved recovery histories validate accumulation through
        # all intervening phases, not only the initial prelude reconstruction.
        for saved in group["counterfactual_snapshots"]:
            if saved is start or not saved["rows"] or saved["rows"][0]["snapshot_stage"] != "post_recovery":
                continue
            expected_steps = saved["snapshot"]["steps"]
            selected = torch.tensor([r["composition_row"] for r in saved["rows"]])
            if torch.equal(env.steps[selected], expected_steps):
                if not torch.equal(history_inputs(geometry, history_actions)[selected], saved["history"]):
                    raise AssertionError("Accumulated real history differs at a saved handoff")
                checked_snapshots += 1
    if len(rows)+len(excluded) != len(group["option_records"]):
        raise AssertionError("Not every saved option query was accounted for")
    raw = {key: torch.stack(values) for key, values in tensors.items()}
    raw.update(rows=rows, excluded=excluded, complete_label_vs_actual_errors=complete_replay_errors,
        replay=dict(source_id=source_id, original_batch_size=n, original_row_order_preserved=True,
            replay_steps=checked_steps, physical_coordinates=24, bitwise_all_times=True,
            complete_initial_snapshot=True, prelude_history_bitwise=True,
            checked_query_memories=checked_memory, checked_saved_handoff_histories=checked_snapshots,
            no_new_task_or_candidate_rollouts=True))
    return raw


def combine_queries(parts):
    if not parts:
        raise ValueError("No oracle source groups")
    return dict({key:torch.cat([p[key] for p in parts], 0) for key in TENSOR_FIELDS},
        rows=[r for p in parts for r in p["rows"]], excluded=[r for p in parts for r in p["excluded"]],
        replay_checks=[p["replay"] for p in parts],
        complete_label_vs_actual_errors=[r for p in parts for r in p["complete_label_vs_actual_errors"]])


def groupings(rows):
    groups = [("all", "all", list(range(len(rows))))]
    for field in ("task_phase", "support", "route_id", "start_contact_mode", "real_interrupted"):
        for value in sorted({r[field] for r in rows}, key=str):
            groups.append((field, str(value), [i for i,r in enumerate(rows) if r[field] == value]))
    for fields in (("task_phase", "support"), ("task_phase", "support", "route_id", "start_contact_mode")):
        combinations = sorted({tuple(r[k] for k in fields) for r in rows})
        for values in combinations:
            groups.append(("/".join(fields), "/".join(map(str, values)),
                           [i for i,r in enumerate(rows) if tuple(r[k] for k in fields) == values]))
    return groups


def score_predictions(data, prediction, indices, time_q95, pose_q95):
    idx = torch.tensor(indices)
    truth = data["truth"][idx]; estimate = prediction[idx]
    error = estimate-truth
    error[:, 2] = wrap(error[:, 2])
    actual_endpoint = data["query_pose"][idx]+truth[:, :3]
    predicted_endpoint = data["query_pose"][idx]+estimate[:, :3]
    reference = data["initial_pose"][idx]; goals = data["goals"][idx]
    def inside(pose):
        return (wrap(pose[:, 2]-reference[:, 2]-goals).abs() < ANGLE_TOLERANCE) & ((pose[:, :2]-reference[:, :2]).norm(dim=-1) < TRANSLATION_TOLERANCE)
    actual_inside, predicted_inside = inside(actual_endpoint), inside(predicted_endpoint)
    by_episode = defaultdict(list)
    for j,i in enumerate(indices):
        by_episode[data["rows"][i]["episode_id"]].append(j)
    balanced_angle_mse = torch.stack([error[js, 2].square().mean() for js in by_episode.values()]).mean()
    result = dict(queries=len(idx), episodes=len(by_episode),
        translation_rmse_units=float(error[:, :2].square().sum(1).mean().sqrt()),
        angular_rmse_rad=float(error[:, 2].square().mean().sqrt()),
        angular_mae_rad=float(error[:, 2].abs().mean()), angular_bias_rad=float(error[:, 2].mean()),
        angular_underprediction_fraction=float((error[:, 2] < 0).float().mean()),
        angle_within_one_third_tolerance_fraction=float((error[:, 2].abs() <= ANGLE_TOLERANCE/3).float().mean()),
        episode_balanced_angular_rmse_rad=float(balanced_angle_mse.sqrt()),
        elapsed_rmse_seconds=float(error[:, 3].square().mean().sqrt()), elapsed_mae_seconds=float(error[:, 3].abs().mean()),
        validation_time_q95_coverage=float((error[:, 3].abs() <= time_q95[idx]).float().mean()),
        validation_pose_q95_marginal_coverage=(error[:, :3].abs() <= pose_q95[idx]).float().mean(0).tolist(),
        completion_accuracy=float(((estimate[:, 4] >= .5) == (truth[:, 4] >= .5)).float().mean()),
        observed_completion_fraction=float((truth[:, 4] >= .5).float().mean()),
        true_full_endpoint_inside_count=int(actual_inside.sum()), predicted_full_endpoint_inside_count=int(predicted_inside.sum()),
        false_inside_count=int((predicted_inside & ~actual_inside).sum()), false_outside_count=int((~predicted_inside & actual_inside).sum()))
    return result


@torch.no_grad()
def audit_models(data, model_pack):
    n = len(data["rows"])
    if not n:
        raise ValueError("No complete oracle labels after exclusions")
    results, predictions = [], {}
    supports = sorted({r["support"] for r in data["rows"]})
    for model in MODELS:
        for seed in (0, 1, 2):
            predicted = torch.empty(n, 5)
            times, poses = torch.empty(n), torch.empty(n, 3)
            for support in supports:
                key = f"{support}/{model}/seed{seed}"
                entry = model_pack["checkpoints"][key]
                idx = torch.tensor([i for i,r in enumerate(data["rows"]) if r["support"] == support])
                predicted[idx] = model_pack["predictors"][key](data["coarse"][idx], full=data["full"][idx], history=data["history"][idx])
                times[idx] = entry["elapsed_time_q95_absolute_residual_seconds"]
                poses[idx] = entry["pose_q95_absolute_residual"]
            predictions[f"{model}/seed{seed}"] = predicted
            for kind, value, indices in groupings(data["rows"]):
                results.append(dict(model=model, seed=seed, group=kind, group_value=value,
                    **score_predictions(data, predicted, indices, times, poses)))
    grouped = defaultdict(list)
    for item in results:
        grouped[(item["model"],item["group"],item["group_value"])].append(item)
    summaries = []
    for (model, kind, value), items in sorted(grouped.items()):
        row = dict(model=model, group=kind, group_value=value, queries=items[0]["queries"], episodes=items[0]["episodes"], seeds=len(items))
        for field in ("translation_rmse_units", "angular_rmse_rad", "angular_bias_rad", "angular_mae_rad", "episode_balanced_angular_rmse_rad", "elapsed_mae_seconds", "validation_time_q95_coverage", "completion_accuracy"):
            values = torch.tensor([r[field] for r in items])
            row[field+"_mean"] = float(values.mean())
            row[field+"_std_over_seeds"] = float(values.std(correction=0))
        summaries.append(row)
    first_history = [r for r in results if r["model"] == "history_matched" and r["group"] == "task_phase" and r["group_value"] == "rotate_first"]
    first_history.sort(key=lambda r:r["seed"])
    seed2_worst = bool(len(first_history) == 3 and first_history[2]["angular_rmse_rad"] > max(first_history[0]["angular_rmse_rad"], first_history[1]["angular_rmse_rad"]))
    interpretation = ("History seed2 also has the largest first-stage angular RMSE on the common oracle query set. This is compatible with its poorer task result, but does not establish a causal explanation of its own action rankings or visited states."
        if seed2_worst else "History seed2 is not the worst first-stage angular predictor on the common oracle query set. These common-state errors alone do not explain its poorer first-stage task result; action ranking and its own visited-state distribution remain unmeasured here.")
    return dict(seed_metrics=results, summary=summaries,
        history_first_stage_seed_comparison=first_history, history_seed2_diagnosis=interpretation), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/novelty/interface_complete_v1.json"))
    parser.add_argument("--out", type=Path, default=Path("data/novelty/interface_complete_v1_model_audit"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    aggregate_digest = sha(args.source)
    aggregate = json.loads(args.source.read_text())
    entries = [g for g in aggregate["groups"] if g["planner"] == "true_option" and g["model_seed"] == 0]
    if {g["goal"] for g in entries} != {"standard", "holdout"} or len(entries) != 2:
        raise ValueError("Expected both original true_option seed0 goal groups")
    model_path = Path(aggregate["model_path"])
    model_digest = sha(model_path)
    if model_digest != aggregate["model_sha256"]:
        raise AssertionError("The model artifact differs from the frozen task run")
    paths = [Path(__file__)] + [Path("vmas_pusht")/(name+".py") for name in
        ("option_models","action_interface","stable_composition","snapshot","physical_abstraction","physics_profiles","scenario")]
    sources = {str(p):sha(p) for p in paths}
    parts, source_digests = [], {}
    for entry in sorted(entries,key=lambda e:e["goal"]):
        path = Path(entry["raw_path"])
        digest = sha(path)
        if digest != entry["raw_sha256"]:
            raise AssertionError("Oracle source artifact changed")
        source_digests[str(path)] = digest
        raw = torch.load(path,map_location="cpu")
        if len(raw["groups"]) != 1:
            raise ValueError("Expected original intact18-row task group")
        parts.append(extract_oracle_queries(raw["groups"][0],source_id=entry["goal"]))
    data = combine_queries(parts)
    pack = load_model_pack(model_path)
    report, predictions = audit_models(data, pack)
    model_task_rows = [r for r in aggregate["summary"] if r["planner"] == "wm_history"]
    comparisons = data["complete_label_vs_actual_errors"]
    errors = torch.tensor([r["error"] for r in comparisons])
    report.update(source_path=str(args.source),source_sha256=aggregate_digest,source_artifact_sha256=source_digests,
        model_path=str(model_path),model_sha256=model_digest,source_code_sha256=sources,
        query_count=len(data["rows"]),excluded_count=len(data["excluded"]),exclusions=data["excluded"],
        query_counts_by_phase=dict(Counter(r["task_phase"] for r in data["rows"])),
        query_counts_by_support=dict(Counter(r["support"] for r in data["rows"])),
        interrupted_real_options_with_complete_counterfactual_labels=sum(r["real_interrupted"] for r in data["rows"]),
        complete_real_option_count=len(comparisons),
        complete_counterfactual_vs_real_execution_max_absolute_error=errors.abs().amax(0).tolist() if len(errors) else None,
        replay_checks=data["replay_checks"], history_task_outcomes=model_task_rows,
        all_architectures_and_seeds_use_identical_queries_and_labels=True,
        no_fitting_or_model_selection=True,
        limitations=[
            "Labels are the source true_option planner's complete private simulator branch outcomes, not WM predictions or interrupted real outcomes.",
            "Queries and actions were selected by the oracle policy. This common distribution controls between-model visitation differences but is not an unbiased sample of every possible option.",
            "Only selected-option truth is saved. This audit does not evaluate model action rankings over the entire candidate set or identify the causal reason for task failures.",
            "Three initialization seeds measure optimizer variation, not independent physical episodes or population confidence intervals.",
            "Validation residual intervals are descriptive OOD coverage checks, not guaranteed calibrated intervals. Completion scores are clipped regression and the production training set had no failure labels.",
            "A complete option's endpoint can differ from the actual task interruption endpoint. Counterfactual and real labels are kept separate.",
            "Original-batch physical replay is bitwise checked. The source counterfactual candidate batches may differ numerically from actual completed options, and that discrepancy is reported.",
        ], elapsed_seconds=time.perf_counter()-started)
    for path,digest in {str(args.source):aggregate_digest,str(model_path):model_digest,**sources,**source_digests}.items():
        if sha(path) != digest:
            raise AssertionError("Read-only input changed during audit: "+path)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.with_suffix(".pt").exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError("Choose a new prefix to preserve an existing audit")
    torch.save(dict(data=data,predictions=predictions,model_sha256=model_digest,source_artifact_sha256=source_digests),args.out.with_suffix(".pt"))
    report["raw_sha256"] = sha(args.out.with_suffix(".pt"))
    args.out.with_suffix(".json").write_text(json.dumps(_plain(report),indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(out=str(args.out),queries=len(data["rows"]),excluded=len(data["excluded"]),
        interrupted=report["interrupted_real_options_with_complete_counterfactual_labels"],
        seconds=report["elapsed_seconds"],history_seed2_diagnosis=report["history_seed2_diagnosis"])),flush=True)


if __name__ == "__main__":
    main()
