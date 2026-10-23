"""Post-primary exploratory prediction diagnostic: retain angular momentum.

The primary physics/model selections, weights, and task evaluations stay frozen.
This diagnostic adds one measured scalar, bar angular velocity, to the 18D body
interface. It neither selects actions nor trains on composed-task observations.
It asks whether a second-order rotational interface is a better small model;
success would not itself establish a cooperative or total-complexity advantage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from vmas_pusht.abstraction_models import (
    ResponseMLP, alias_statistics, body_frame, coarse_groups, error_metrics,
    fit_mlp, normalization, parameter_count, predict, save_artifact,
    summarize_seed_metrics, world_frame_response,
)
from vmas_pusht.stable_local_wm import validate_local_dataset


MODEL_NAME = "coarse_omega_small"
BAR_OMEGA_FULL_INDEX = 26
STRICT_TEST_SPLITS = ("test_action", "test_context", "test_nuisance")
WIDTH = 31


def registered_protocol():
    return dict(stage="post-primary exploratory prediction-only",
        model=MODEL_NAME, representation="body_coarse_omega19", input_dim=19,
        width=WIDTH, parameters=parameter_count(19, WIDTH),
        additional_sensor="Current bar angular velocity", additional_state_scalars=1,
        additional_history_memory_scalars=0,
        optimizer=dict(updates=1000, seeds=[0, 1, 2], batch_size=256, lr=.003),
        data="Exactly the already frozen local data, training row IDs and train-only normalization protocol; no new physical trajectories.",
        output="World-frame bar dx, dy, wrapped dtheta over a five-step constant-action macro.",
        test_splits=list(STRICT_TEST_SPLITS), validation_in_strict_test=False,
        action_selection=False, primary_models_replaced=False,
        interpretation="Angular velocity is legitimate second-order state, not proof that cooperation removes the need for hidden translational state.")


def omega_features(coarse, full, reference_substeps):
    if coarse.ndim != 2 or coarse.shape[1] != 18:
        raise ValueError("coarse input must be B x 18")
    if full is None or full.shape != (len(coarse), 32):
        raise ValueError("angular diagnostic requires full B x 32 solely to read bar omega")
    if not torch.equal(full[:, :15], coarse[:, :15]) or not torch.equal(full[:, 16:18], coarse[:, 16:18]):
        raise ValueError("full input must share the coarse prefix")
    canonical = coarse.clone()
    canonical[:, 15] = float(reference_substeps)
    return torch.cat((body_frame(canonical), full[:, BAR_OMEGA_FULL_INDEX:BAR_OMEGA_FULL_INDEX + 1]), 1)


@torch.no_grad()
def predict_entry(entry, coarse, full=None, history=None):
    """Prediction-only API; history is accepted for caller compatibility, unused."""
    if entry.get("representation") != "body_coarse_omega19":
        raise ValueError("Expected angular-ablation entry")
    model = ResponseMLP(entry["input_dim"], entry["width"])
    model.load_state_dict(entry["state_dict"])
    model.eval()
    inputs = omega_features(coarse, full, entry["reference_substeps"])
    local = predict(model, inputs, entry["normalization"])
    return world_frame_response(coarse, local)


def _metrics_for_predictions(data, predictions, checkpoints):
    rows, target = data["rows"], data["Y"].float()
    train = torch.tensor([row["split"] == "train" for row in rows])
    scale = normalization(data["X_coarse"][train], target[train])["y_scale"]
    metrics = []
    for key, result in sorted(predictions.items()):
        mode, name, seed_label = key.split("/")
        seed = int(seed_label.removeprefix("seed"))
        ids, predicted = result["row_indices"].tolist(), result["prediction"]
        checkpoint = checkpoints[key]
        count = checkpoint.get("parameter_count", parameter_count(checkpoint["input_dim"], checkpoint["width"]))
        for split in (*STRICT_TEST_SPLITS, "strict_test_only"):
            picked = [j for j, i in enumerate(ids) if
                      (rows[i]["split"] in STRICT_TEST_SPLITS if split == "strict_test_only" else rows[i]["split"] == split)]
            if not picked:
                continue
            metrics.append(dict(mode=mode, model=name, seed=seed, split=split, parameters=count,
                **error_metrics(target[[ids[j] for j in picked]], predicted[picked], scale)))
    return metrics


def primary_strict_sidecar(data, primary_artifact):
    """Re-score existing predictions; does not refit or change a primary artifact."""
    validate_local_dataset(data)
    metrics = _metrics_for_predictions(data, primary_artifact["predictions"], primary_artifact["checkpoints"])
    return dict(schema_version=1, stage="post-primary reporting-only regrouping",
        validation_in_strict_test=False, strict_test_splits=list(STRICT_TEST_SPLITS),
        profile=data["profile"], reference_substeps=data["reference_substeps"],
        primary_weights_changed=False, predictions_recomputed=False,
        seed_metrics=metrics, summary=summarize_seed_metrics(metrics))


def fit_angular(data, *, steps=1000, seeds=(0, 1, 2), batch_size=256, lr=.003):
    by_mode = validate_local_dataset(data)
    coarse, full, target = [data[key].float() for key in ("X_coarse", "X_full", "Y")]
    rows = data["rows"]
    train = torch.tensor([row["split"] == "train" for row in rows])
    x = omega_features(coarse, full, data["reference_substeps"])
    local_target = body_frame(coarse, target)[1]
    norm = normalization(x[train], local_target[train])
    checkpoints, predictions = {}, {}
    for mode, indices in sorted(by_mode.items()):
        indices = sorted(indices, key=lambda i: rows[i]["window_id"])
        train_ids = [i for i in indices if rows[i]["split"] == "train"]
        for seed in seeds:
            model = fit_mlp(x[train_ids], local_target[train_ids], norm, WIDTH,
                            steps, seed, batch_size, lr)
            entry = dict(model=MODEL_NAME, support=mode, input_dim=19, width=WIDTH,
                representation="body_coarse_omega19", reference_substeps=data["reference_substeps"],
                parameter_count=parameter_count(19, WIDTH), additional_state_scalars=1,
                additional_sensor="bar angular velocity", additional_history_memory_scalars=0,
                normalization=norm, state_dict=model.state_dict(), profile=data["profile"],
                train_row_indices=torch.tensor(train_ids), optimizer_steps=steps, seed=int(seed),
                macro_steps=5, stage="post-primary exploratory", action_selection_evaluated=False)
            key = f"{mode}/{MODEL_NAME}/seed{seed}"
            checkpoints[key] = entry
            predictions[key] = dict(row_indices=torch.tensor(indices),
                prediction=predict_entry(entry, coarse[indices], full[indices]))
    metrics = _metrics_for_predictions(data, predictions, checkpoints)
    report = dict(schema_version=1, protocol=registered_protocol(), profile=data["profile"],
        reference_substeps=data["reference_substeps"],
        actual_optimizer=dict(updates=steps, seeds=list(seeds), batch_size=batch_size, lr=lr),
        seed_metrics=metrics, summary=summarize_seed_metrics(metrics),
        training_rows_per_support={mode: sum(rows[i]["split"] == "train" for i in indices)
                                   for mode, indices in by_mode.items()},
        limitations=[
            "Exploratory after the primary model protocol was frozen; does not replace or rerank primary task results.",
            "One extra angular-velocity measurement changes information availability. A smaller network alone is not a total sensing/computation advantage.",
            "Good angular-state prediction would not show cooperation-specific benefit; passive, world-anchor and preload comparisons remain necessary.",
            "Only local heldout prediction is evaluated; no planner or task success claim for this diagnostic.",
            "Normalization is pooled across supports using training rows only, matching the primary protocol; all support models use the same optimizer budget and window IDs.",
            "Three seeds quantify optimizer variation, not independent physical replications.",
        ])
    return report, dict(checkpoints=checkpoints, predictions=predictions)


def omega_conditioned_alias(data):
    """Compare exact geometry aliases with and without observing bar omega.

    Conditioning makes some omega groups singletons. Those rows cannot estimate
    residual ambiguity, so both the retained coverage and geometry-only error
    on exactly that same subset are reported. This avoids an artificial gain
    from dropping the hard angular-velocity perturbations.
    """
    rows = data["rows"]
    coarse, full, y = [data[key].float() for key in ("X_coarse", "X_full", "Y")]
    augmented = torch.cat((coarse, full[:, BAR_OMEGA_FULL_INDEX:BAR_OMEGA_FULL_INDEX + 1]), 1)
    if "physical_initial" in data:
        # physical_initial layout is A6, B6, bar6, stem6; bar omega is17.
        if not torch.equal(full[:, BAR_OMEGA_FULL_INDEX], data["physical_initial"][:, 17]):
            raise ValueError("Bar angular-velocity column disagrees with full physical state")
    summaries = []
    for mode, angle in sorted(set((r["mode"], r["angle_deg"]) for r in rows)):
        indices = [i for i, r in enumerate(rows) if (r["mode"], r["angle_deg"]) == (mode, angle)]
        labels, counts = coarse_groups(augmented[indices])
        retained = [indices[j] for j in np.flatnonzero(counts[labels] > 1)]
        summaries.append(dict(mode=mode, angle_deg=angle, total_rows=len(indices),
            repeated_omega_conditioned_rows=len(retained),
            repeated_omega_conditioned_fraction=len(retained)/len(indices),
            observed_omega_values_rad_per_time=sorted(set(float(v) for v in full[indices, BAR_OMEGA_FULL_INDEX])),
            repeated_group_omega_values_rad_per_time=sorted(set(float(v) for v in full[retained, BAR_OMEGA_FULL_INDEX])),
            geometry_all_rows=alias_statistics(coarse[indices], y[indices]),
            geometry_on_same_repeated_omega_subset=alias_statistics(coarse[retained], y[retained]),
            geometry_plus_omega=alias_statistics(augmented[indices], y[indices])))
    return dict(profile=data["profile"], substeps=data["substeps"],
        used_for_training=False, stage="post-primary exploratory exact-alias conditioning",
        added_sensor="bar angular velocity", additional_state_scalars=1, summary=summaries,
        limitations=[
            "Only the repeated omega groups identify remaining finite-support conditional variance; singleton groups are explicitly excluded.",
            "The available constructed grid has multiple translational perturbations only at omega=0. Results cannot establish nuisance suppression over all angular momenta.",
            "Geometry-only variance on all nine perturbations is not a like-for-like effect estimate after conditioning; compare the explicitly matched repeated subset.",
            "Initial conditions are constructed legal states; no task trajectory or reachable-history generalization is established.",
        ])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--primary-models", type=Path, required=True)
    parser.add_argument("--alias-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs = {str(path): sha256(path) for path in (args.data, args.primary_models, args.alias_data)}
    registration = {**registered_protocol(), "actual_updates": args.steps, "actual_seeds": args.seeds,
                    "input_sha256": inputs}
    registration_path = args.output.with_name(args.output.name + "_registration.json")
    registration_path.write_text(json.dumps(registration, indent=2, allow_nan=False)+"\n")
    data = torch.load(args.data, map_location="cpu")
    primary = torch.load(args.primary_models, map_location="cpu")
    alias_data = torch.load(args.alias_data, map_location="cpu")
    strict = primary_strict_sidecar(data, primary)
    alias = omega_conditioned_alias(alias_data)
    report, artifact = fit_angular(data, steps=args.steps, seeds=args.seeds)
    for key, entry in artifact["checkpoints"].items():
        mode, _, seed = key.split("/")
        reference = primary["checkpoints"][f"{mode}/coarse_body_small/{seed}"]
        if not torch.equal(entry["train_row_indices"], reference["train_row_indices"]):
            raise AssertionError("Exploratory model has different training row IDs")
        if entry["optimizer_steps"] != reference["optimizer_steps"]:
            raise AssertionError("Exploratory model has a different update budget")
    provenance = dict(input_sha256=inputs,
        source_sha256={str(path): sha256(path) for path in (Path(__file__),
            Path(__file__).with_name("stable_local_wm.py"), Path(__file__).with_name("abstraction_models.py"))},
        registration_sha256=sha256(registration_path), torch_version=str(torch.__version__),
        primary_model_sha256_before=inputs[str(args.primary_models)],
        elapsed_seconds=time.perf_counter()-started)
    for path, before in inputs.items():
        if sha256(path) != before:
            raise AssertionError("A frozen primary input changed during this diagnostic")
    model_path = args.output.with_name(args.output.name + "_models.pt")
    save_artifact(model_path, provenance, artifact)
    report.update(provenance=provenance, model_sha256=sha256(model_path))
    strict["provenance"] = provenance
    alias["provenance"] = provenance
    for suffix, result in (("_models.json", report), ("_primary_strict_test.json", strict), ("_alias.json", alias)):
        args.output.with_name(args.output.name + suffix).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"model_path": str(model_path), "elapsed_seconds": provenance["elapsed_seconds"],
        "strict_summary": [r for r in report["summary"] if r["split"] == "strict_test_only"]}, indent=2))


if __name__ == "__main__":
    main()
