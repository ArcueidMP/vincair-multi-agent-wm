"""Held-out local decision audit for frozen UAV recovery outcome models.

Every comparison uses saved counterfactual outcomes from a common physical
context. This audit selects local operations; complete transport missions are
evaluated separately. Training, margins, action banks and thresholds stay fixed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .local_model import FEATURE_NAMES, TARGET_NAMES, load_dataset, load_model, prediction_metrics
from .mission import MissionConfig, outcome_scores


BANKS = ("direct9", "expanded18", "allocated9_s0", "allocated9_s1")
ACTION_FEATURES = ("operation_rearrange_seconds", "operation_retension_seconds", "operation_reconfigure")
GRID = tuple(itertools.product((1., 2., 3.), (1.5, 2., 3.)))
MODEL_KEYS = tuple(f"{kind}/seed{seed}" for kind in ("small", "large") for seed in range(3))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_references(references):
    for path, expected in references.items():
        if sha(path) != expected:
            raise AssertionError("Frozen input hash changed: "+str(path))


def action_key(row):
    operation = row["recovery_config"]
    return (bool(row["reconfigure"]), float(operation["rearrange_seconds"]),
            float(operation["retension_seconds"]))


def bank_indices(rows, bank):
    """Map a declared bank to arbitrary source-row order, with stable tie order."""
    if bank not in BANKS:
        raise ValueError("Unknown candidate bank")
    lookup = {}
    for i, row in enumerate(rows):
        key = action_key(row)
        if key in lookup:
            raise ValueError("Duplicate action in a physical context")
        lookup[key] = i
    expected = {(mode, a, b) for mode in (False, True) for a, b in GRID}
    if set(lookup) != expected:
        raise ValueError("A test context must contain the complete declared eighteen-option bank")
    if bank == "expanded18":
        keys = [(mode, a, b) for mode in (False, True) for a, b in GRID]
    elif bank == "direct9":
        keys = [(False, a, b) for a, b in GRID]
    else:
        shift = int(bank[-1])
        keys = [(bool((k+shift) % 2), a, b) for k, (a, b) in enumerate(GRID)]
    return np.asarray([lookup[key] for key in keys], dtype=int)


def validate_dataset(dataset, metadata):
    """Check split identities, metadata alignment and common-state pairing."""
    if metadata["feature_names"] != list(FEATURE_NAMES) or metadata["target_names"] != list(TARGET_NAMES):
        raise ValueError("Dataset feature/target declarations differ from the frozen interface")
    context_specs = {}
    for context in metadata["contexts"]:
        cid = int(context["context_id"])
        if cid in context_specs:
            raise ValueError("Duplicate context declaration")
        context_specs[cid] = context
    split_rows, groups = {}, {}
    for split in ("train", "val", "test", "test_action"):
        rows = [row for row in metadata["rows"] if row["split"] == split]
        x, y, ids = (dataset[f"{key}_{split}"] for key in ("X", "Y", "groups"))
        if len(rows) != len(x) or ids.shape != (len(x),) or y.shape != (len(x), 8):
            raise ValueError("Metadata and arrays have different split lengths")
        if metadata["examples_by_split"][split] != len(x):
            raise ValueError("Registered split count differs from arrays")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite saved local features or outcomes")
        expected_split = "test" if split == "test_action" else split
        for i, row in enumerate(rows):
            cid = int(ids[i])
            if cid not in context_specs or row["context_id"] != cid or context_specs[cid]["split"] != expected_split:
                raise ValueError("Metadata context order differs from array order")
            labels = np.asarray([float(row["summary"][name]) for name in TARGET_NAMES])
            if not np.array_equal(labels, y[i]):
                raise ValueError("Saved outcome labels differ from row metadata")
            for name, value in zip(ACTION_FEATURES,
                    (row["recovery_config"]["rearrange_seconds"],
                     row["recovery_config"]["retension_seconds"], float(row["reconfigure"]))):
                if x[i, FEATURE_NAMES.index(name)] != value:
                    raise ValueError("Declared action differs from its model feature")
        groups[split] = set(int(x) for x in ids)
        split_rows[split] = rows
    for a, b in (("train", "val"), ("train", "test"), ("val", "test"),
                 ("train", "test_action"), ("val", "test_action")):
        if groups[a] & groups[b]:
            raise ValueError("Physical contexts overlap between development and held-out splits")
    if groups["test_action"] != groups["test"]:
        raise ValueError("Unseen-action examples must cover the same held-out contexts")
    fixed = np.ones(len(FEATURE_NAMES), dtype=bool)
    fixed[[FEATURE_NAMES.index(name) for name in ACTION_FEATURES]] = False
    blocks = []
    for cid in sorted(groups["test"]):
        indices = np.flatnonzero(dataset["groups_test"] == cid)
        rows = [split_rows["test"][i] for i in indices]
        bank_indices(rows, "expanded18")
        x = dataset["X_test"][indices]
        extra = np.flatnonzero(dataset["groups_test_action"] == cid)
        extra_x = dataset["X_test_action"][extra]
        if not np.array_equal(x[:, fixed], np.repeat(x[:1, fixed], len(x), axis=0)):
            raise ValueError("Counterfactual candidates differ in non-action state or controller memory")
        if not np.array_equal(extra_x[:, fixed], np.repeat(x[:1, fixed], len(extra_x), axis=0)):
            raise ValueError("Unseen-action rows differ from their held-out common state")
        known_pairs = {action_key(row)[1:] for row in rows}
        extra_keys = [action_key(split_rows["test_action"][i]) for i in extra]
        expected_extra = set(itertools.product((False, True), (1.5, 2.5), (1.75, 2.5)))
        if len(extra_keys) != 8 or set(extra_keys) != expected_extra:
            raise ValueError("Unseen-action context must contain the eight declared programs exactly once")
        if any(action_key(split_rows["test_action"][i])[1:] in known_pairs for i in extra):
            raise ValueError("An unseen-action program duplicates the decision bank")
        blocks.append(dict(context_id=cid, indices=indices, rows=rows))
    return blocks, dict(contexts_by_split={k: sorted(v) for k, v in groups.items()},
        test_contexts=len(blocks), decision_examples=len(dataset["X_test"]),
        unseen_action_examples=len(dataset["X_test_action"]), common_state_features=int(fixed.sum()),
        varied_features=list(ACTION_FEATURES), metadata_labels_and_array_order_match=True)


def select_predicted(predictions, config=None, margins=(0., 0.)):
    """Selection accepts predictions only, so diagnostic labels cannot alter it."""
    scores, feasible, fallback = outcome_scores(predictions, config, margins)
    return dict(index=int(np.argmin(scores)), scores=scores, feasible=feasible,
                fallback=bool(fallback))


def _ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    begin = 0
    while begin < len(values):
        end = begin+1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        ranks[order[begin:end]] = (begin+end-1)/2.
        begin = end
    return ranks


def rank_correlation(left, right):
    """Spearman correlation with average ties; constant rankings are undefined."""
    a, b = _ranks(left), _ranks(right)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def base_cost(outcomes):
    values = np.asarray(outcomes)
    return values[:, 2]+.02*values[:, 3]+.01*values[:, 4]


def score_selection(selection, predictions, actual, all_actual, config=None):
    """Reveal actual outcomes only after a bank-specific selection is fixed."""
    _, actual_feasible, _ = outcome_scores(actual, config)
    actual_scores, _, _ = outcome_scores(actual, config)
    _, global_feasible, _ = outcome_scores(all_actual, config)
    chosen = selection["index"]
    local_cost, global_cost = base_cost(actual), base_cost(all_actual)
    best_local = float(local_cost[actual_feasible].min()) if actual_feasible.any() else None
    best_global = float(global_cost[global_feasible].min()) if global_feasible.any() else None
    valid = bool(actual_feasible[chosen])
    predicted_feasible = selection["feasible"]
    return dict(selected_actual={name: float(value) for name, value in zip(TARGET_NAMES, actual[chosen])},
        selected_prediction={name: float(value) for name, value in zip(TARGET_NAMES, predictions[chosen])},
        local_operation_completion=bool(actual[chosen, 7] >= .5), constraint_pass=valid,
        selected_base_cost=float(local_cost[chosen]), actual_feasible_count=int(actual_feasible.sum()),
        actual_global_feasible_count=int(global_feasible.sum()), predicted_feasible_count=int(predicted_feasible.sum()),
        false_predicted_feasible_count=int(np.sum(predicted_feasible & ~actual_feasible)),
        false_predicted_infeasible_count=int(np.sum(~predicted_feasible & actual_feasible)),
        selected_false_predicted_feasible=bool(predicted_feasible[chosen] and not valid),
        bank_has_feasible_option=bool(actual_feasible.any()), global_has_feasible_option=bool(global_feasible.any()),
        bank_feasibility_miss=bool(actual_feasible.any() and not valid),
        global_feasibility_miss=bool(global_feasible.any() and not valid),
        best_true_feasible_cost_in_bank=best_local, best_true_feasible_cost_global=best_global,
        feasible_cost_regret_in_bank=float(local_cost[chosen]-best_local) if valid and best_local is not None else None,
        feasible_cost_regret_global=float(local_cost[chosen]-best_global) if valid and best_global is not None else None,
        score_spearman=rank_correlation(selection["scores"], actual_scores))


def margin_coverage(predicted, actual, margins):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    covered = actual[:, :2] <= predicted[:, :2]+np.asarray(margins)[None]
    return dict(examples=len(actual), margins=list(map(float, margins)),
        excursion_covered=int(covered[:, 0].sum()), tension_covered=int(covered[:, 1].sum()),
        jointly_covered=int(covered.all(1).sum()),
        excursion_coverage=float(covered[:, 0].mean()), tension_coverage=float(covered[:, 1].mean()),
        joint_coverage=float(covered.all(1).mean()))


def aggregate(rows):
    result = []
    for key in sorted({(r["predictor"], r["bank"]) for r in rows}):
        subset = [r for r in rows if (r["predictor"], r["bank"]) == key]
        regrets = [r["feasible_cost_regret_global"] for r in subset if r["feasible_cost_regret_global"] is not None]
        correlations = [r["score_spearman"] for r in subset if r["score_spearman"] is not None]
        result.append(dict(predictor=key[0], bank=key[1], contexts=len(subset),
            constraint_passes=sum(r["constraint_pass"] for r in subset),
            local_operation_completions=sum(r["local_operation_completion"] for r in subset),
            fallbacks=sum(r["fallback"] for r in subset),
            contexts_with_actual_feasible_choice=sum(r["bank_has_feasible_option"] for r in subset),
            actual_feasible_candidates=sum(r["actual_feasible_count"] for r in subset),
            predicted_feasible_candidates=sum(r["predicted_feasible_count"] for r in subset),
            false_predicted_feasible=sum(r["false_predicted_feasible_count"] for r in subset),
            false_predicted_infeasible=sum(r["false_predicted_infeasible_count"] for r in subset),
            selected_false_predicted_feasible=sum(r["selected_false_predicted_feasible"] for r in subset),
            candidate_queries=sum(r["candidate_queries"] for r in subset),
            mean_selected_excursion=float(np.mean([r["selected_actual"][TARGET_NAMES[0]] for r in subset])),
            maximum_selected_excursion=float(np.max([r["selected_actual"][TARGET_NAMES[0]] for r in subset])),
            mean_selected_tension=float(np.mean([r["selected_actual"][TARGET_NAMES[1]] for r in subset])),
            maximum_selected_tension=float(np.max([r["selected_actual"][TARGET_NAMES[1]] for r in subset])),
            mean_selected_seconds=float(np.mean([r["selected_actual"][TARGET_NAMES[2]] for r in subset])),
            mean_selected_work=float(np.mean([r["selected_actual"][TARGET_NAMES[3]] for r in subset])),
            regret_eligible_contexts=len(regrets), mean_feasible_cost_regret_global=None if not regrets else float(np.mean(regrets)),
            ranked_contexts=len(correlations), mean_score_spearman=None if not correlations else float(np.mean(correlations))))
    return result


def paired_results(rows):
    index = {}
    for row in rows:
        key = (row["predictor"], row["bank"], row["context_id"])
        if key in index:
            raise ValueError("Duplicate local decision would inflate paired counts")
        index[key] = row
    output = []
    for predictor in sorted({r["predictor"] for r in rows}):
        direct = {cid: r for (p, bank, cid), r in index.items() if p == predictor and bank == "direct9"}
        for bank in BANKS[1:]:
            compared = {cid: r for (p, b, cid), r in index.items() if p == predictor and b == bank}
            if set(direct) != set(compared):
                raise ValueError("Paired banks do not share all physical contexts")
            changes = [dict(context_id=cid, direct=bool(direct[cid]["constraint_pass"]),
                compared=bool(compared[cid]["constraint_pass"]),
                improved=not direct[cid]["constraint_pass"] and compared[cid]["constraint_pass"],
                worsened=direct[cid]["constraint_pass"] and not compared[cid]["constraint_pass"])
                for cid in sorted(direct)]
            output.append(dict(predictor=predictor, bank=bank, contexts=len(changes),
                direct_success=sum(r["direct"] for r in changes), compared_success=sum(r["compared"] for r in changes),
                improved=sum(r["improved"] for r in changes), worsened=sum(r["worsened"] for r in changes), rows=changes))
    return output


def run_audit(data_path, metadata_path, models_path):
    data_path, metadata_path, models_path = map(Path, (data_path, metadata_path, models_path))
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads(models_path.read_text())
    references = {str(data_path): metadata["array_sha256"], str(metadata_path): sha(metadata_path),
                  str(models_path): sha(models_path)}
    if manifest["dataset_sha256"] != references[str(data_path)]:
        raise AssertionError("Frozen models reference a different local dataset")
    references.update(metadata["source_sha256"])
    references.update(manifest["source_sha256"])
    for obj, prefix in ((metadata, "registration"), (manifest, "training_registration")):
        references[obj[prefix+"_path"]] = obj[prefix+"_sha256"]
    for part in metadata["parts"]:
        references[part["path"]] = part["sha256"]
    references[str(Path(__file__))] = sha(__file__)
    references["uav_transport/mission.py"] = sha("uav_transport/mission.py")
    if set(manifest["models"]) != set(MODEL_KEYS):
        raise ValueError("Expected all six prespecified frozen predictor fits")
    models = {}
    for key in MODEL_KEYS:
        entry = manifest["models"][key]
        for suffix, digest in ((".npz", "weights_sha256"), (".json", "metadata_sha256")):
            references[entry["prefix"]+suffix] = entry[digest]
    verify_references(references)
    dataset = load_dataset(data_path)
    blocks, checks = validate_dataset(dataset, metadata)
    for key in MODEL_KEYS:
        models[key] = load_model(manifest["models"][key]["prefix"])
        if models[key].metadata["dataset_sha256"] != references[str(data_path)]:
            raise AssertionError("Predictor metadata references different training data")
    config = MissionConfig()
    rows, diagnostics, prediction_arrays = [], [], {}
    for predictor in (*MODEL_KEYS, "true_option"):
        oracle = predictor == "true_option"
        model = None if oracle else models[predictor]
        margins = (0., 0.) if oracle else tuple(manifest["models"][predictor]["margins"])
        if not oracle:
            for split in ("test", "test_action"):
                prediction = model.predict(dataset["X_"+split])
                prediction_arrays[predictor.replace("/", "_")+"_"+split] = prediction
                diagnostics.append(dict(predictor=predictor, split=split,
                    metrics=prediction_metrics(prediction, dataset["Y_"+split]),
                    margin_coverage=margin_coverage(prediction, dataset["Y_"+split], margins),
                    diagnostic_prediction_rows=len(prediction)))
        for block in blocks:
            global_indices = block["indices"]
            for bank in BANKS:
                selected_local = bank_indices(block["rows"], bank)
                source_indices = global_indices[selected_local]
                # Only the exact predictor reads outcome labels during selection.
                prediction = (dataset["Y_test"][source_indices].copy() if oracle else
                              model.predict(dataset["X_test"][source_indices]))
                selection = select_predicted(prediction, config, margins)
                chosen_global = int(source_indices[selection["index"]])
                evaluation = score_selection(selection, prediction, dataset["Y_test"][source_indices],
                    dataset["Y_test"][global_indices], config)
                selected_row = block["rows"][selected_local[selection["index"]]]
                rows.append(dict(predictor=predictor, bank=bank, context_id=block["context_id"],
                    candidate_queries=len(source_indices), source_row_indices=source_indices.tolist(),
                    candidate_actions=[action_key(block["rows"][i]) for i in selected_local],
                    selected_source_row=chosen_global, selected_action=action_key(selected_row),
                    selected_index_in_bank=selection["index"], validation_margins=list(margins),
                    candidate_predictions=prediction.tolist(), ranking_scores=selection["scores"].tolist(),
                    predicted_feasible=selection["feasible"].tolist(), fallback=selection["fallback"],
                    additional_simulator_steps=0, **evaluation))
    verify_references(references)
    report = dict(audit="held-out local operation selection", data_path=str(data_path),
        metadata_path=str(metadata_path), models_path=str(models_path), mission_config=asdict(config),
        split_checks=checks, banks=list(BANKS),
        allocation="For duration-pair index k=0..8, allocated shift s selects reconfigure=bool((k+s)%2); each pair appears once",
        local_decisions=len(rows), rows=rows, summary=aggregate(rows), paired=paired_results(rows),
        prediction_diagnostics=diagnostics, input_sha256=references, verified_input_files=len(references),
        no_training_or_margin_tuning=True, additional_physics_steps=0,
        interpretation="Saved local counterfactual decisions, separate from complete mission delivery counts. Each fit and allocation repeats the same held-out physical contexts. Regret is conditional on selecting a truly feasible option; infeasible choices are explicit feasibility misses.",
        score_spearman="Average tied ranks of predicted and actual bank-specific constrained scores; undefined constant ranks are null.")
    return report, prediction_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/uav_transport/local_v1.npz"))
    parser.add_argument("--metadata", type=Path, default=Path("data/uav_transport/local_v1.json"))
    parser.add_argument("--models", type=Path, default=Path("data/uav_transport/models_v1_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("data/uav_transport/selection_v1"))
    args = parser.parse_args()
    json_path, array_path = Path(str(args.out)+".json"), Path(str(args.out)+".npz")
    if json_path.exists() or array_path.exists():
        raise FileExistsError("Preserve existing decision audits with a fresh output prefix")
    report, predictions = run_audit(args.data, args.metadata, args.models)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, **predictions)
    report.update(prediction_array_path=str(array_path), prediction_array_sha256=sha(array_path))
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(json_path), checks=report["split_checks"],
        summary=report["summary"], paired=[{k: v for k, v in x.items() if k != "rows"} for x in report["paired"]])), flush=True)


if __name__ == "__main__":
    main()
