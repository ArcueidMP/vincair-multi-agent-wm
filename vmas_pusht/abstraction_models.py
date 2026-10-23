"""Falsifiable small-model audit of helper-induced physical abstraction.

This deliberately coarse, pose-only interface is NOT the released 15D WM state:
that state already contains bar/agent velocities. A handcrafted helper has access
to velocity feedback; our pilot does not demonstrate learned abstraction, reduced
total sensing, or a reduction of total learned model/controller complexity.

Train rows and fixed optimizer updates are identical across helper conditions.
Scaling uses training rows only, shared target units permit cross-mode comparison,
and model/width/steps are fixed before validation/test results are inspected.
Empirical identical-input conditional variance is reported separately from model
fit error. It is an empirical finite-support lower bound, not a global theorem.

    python -m vmas_pusht.abstraction_models --data data/novelty/abstraction_physics.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ResponseMLP(nn.Module):
    def __init__(self, inputs: int, width: int, outputs: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, outputs))

    def forward(self, x):
        return self.net(x)


def parameter_count(inputs: int, width: int, outputs: int = 3):
    return (inputs + 1) * width + (width + 1) * width + (width + 1) * outputs


def matched_width(inputs: int, budget: int, outputs: int = 3):
    """Largest integer width not exceeding the coarse model's parameter budget."""
    widths = [w for w in range(1, budget + 1) if parameter_count(inputs, w, outputs) <= budget]
    if not widths:
        raise ValueError("parameter budget is too small")
    return widths[-1]


def validate_dataset(data):
    for name in ("X_coarse", "X_full", "Y", "rows"):
        if name not in data:
            raise ValueError(f"missing dataset field {name}")
    n = len(data["rows"])
    for name in ("X_coarse", "X_full", "Y"):
        values = torch.as_tensor(data[name])
        if values.ndim != 2 or len(values) != n or not torch.isfinite(values).all():
            raise ValueError(f"{name} must be a finite M x D matrix")
    if data["Y"].shape[1] != 3:
        raise ValueError("Y must contain dx, dy, wrapped dtheta")
    if not n:
        raise ValueError("dataset is empty")
    coarse = torch.as_tensor(data["X_coarse"])
    full = torch.as_tensor(data["X_full"])
    if full.shape[1] < coarse.shape[1] or not torch.equal(full[:, :coarse.shape[1]], coarse):
        raise ValueError("full input must start with exactly the coarse input")
    for row in data["rows"]:
        for field in ("mode", "context_id", "action_id", "split"):
            if field not in row:
                raise ValueError(f"missing row field {field}")
        if row["split"] == "train" and (not row.get("context_is_train", True)
                                           or not row.get("action_is_train", True)):
            raise ValueError("heldout context/action marked as a training row")
    # Same context/action row sets ensure data budget parity, rather than merely
    # equal row counts from potentially different training distributions.
    by_mode = defaultdict(list)
    for i, row in enumerate(data["rows"]):
        by_mode[row["mode"]].append(i)
    reference = None
    reference_inputs = None
    for mode, indices in sorted(by_mode.items()):
        keys = [(str(data["rows"][i]["context_id"]), str(data["rows"][i]["action_id"]),
                 str(data["rows"][i]["split"])) for i in indices]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate context/action/split rows in {mode}")
        key_to_index = dict(zip(keys, indices))
        keys = set(keys)
        if reference is not None and keys != reference:
            raise ValueError("helper modes do not have matched context/action row sets")
        if reference_inputs is not None:
            ordered = sorted(keys)
            this = [key_to_index[k] for k in ordered]
            other = [reference_inputs[k] for k in ordered]
            if not torch.equal(coarse[this], coarse[other]) or not torch.equal(full[this], full[other]):
                raise ValueError("matched helper rows have different initial inputs")
        reference = keys
        reference_inputs = key_to_index
    if not any(r["split"] == "train" for r in data["rows"]):
        raise ValueError("missing training rows")
    return by_mode


def coarse_groups(x):
    """Group by actual identical inputs, never by an unverified semantic label."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("coarse inputs must be a finite matrix")
    _, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    return inverse, counts


def alias_statistics(x, y):
    """Empirical best deterministic prediction MSE for exact repeated (phi,u)."""
    x, y = np.asarray(x), np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or y.ndim != 2 or y.shape[1] != 3 or not np.isfinite(y).all():
        raise ValueError("outputs must be finite M x 3 and match the inputs")
    labels, counts = coarse_groups(x)
    means = np.zeros((len(counts), 3))
    np.add.at(means, labels, y)
    means /= counts[:, None]
    residual = y - means[labels]
    repeated = counts[labels] > 1
    within = np.mean(residual[repeated] ** 2, axis=0) if repeated.any() else np.full(3, np.nan)
    mean_responses = means[counts > 1]
    return {
        "rows": len(y), "unique_exact_inputs": len(counts),
        "repeated_input_groups": int((counts > 1).sum()), "rows_in_repeated_groups": int(repeated.sum()),
        "median_replicates_per_repeated_group": float(np.median(counts[counts > 1])) if repeated.any() else None,
        "conditional_rmse_xy_units_rad": np.sqrt(within).tolist() if repeated.any() else None,
        "conditional_translation_rmse_units": float(np.sqrt(within[:2].sum())) if repeated.any() else None,
        "conditional_mse_xy_units2_rad2": within.tolist() if repeated.any() else None,
        "mean_response_range_xy_units_rad": np.ptp(mean_responses, axis=0).tolist() if len(mean_responses) else None,
        "mean_abs_angular_response_rad": float(np.mean(np.abs(mean_responses[:, 2]))) if len(mean_responses) else None,
        "note": "Sample-weighted empirical MSE about each exact-input group mean, computed on repeated groups only. Unobserved states/actions may have additional ambiguity.",
    }


def matched_progress_alias(data, reference_mode="passive", bins=4, substeps=None):
    """Descriptive conditional response spread at overlapping angular progress.

    Exact-input groups are aggregated within a helper mode. Bin by the absolute
    group-mean angular displacement, with common cut points from both modes.
    These post-outcome strata are descriptive, not a randomized causal estimate.
    They cannot identify equal-task-progress effects without sufficient overlap.
    """
    coarse = np.asarray(data["X_coarse"])
    y = np.asarray(data["Y"])
    rows = data["rows"]
    modes = sorted(set(r["mode"] for r in rows))
    if reference_mode not in modes:
        return {}
    grouped = {}
    for mode in modes:
        indices = [i for i, r in enumerate(rows) if r["mode"] == mode and r["split"] != "train"
                   and (substeps is None or r.get("substeps") == substeps)]
        labels, counts = coarse_groups(coarse[indices])
        values = y[indices]
        items = []
        for g, count in enumerate(counts):
            if count < 2:
                continue
            block = values[labels == g]
            items.append((float(abs(block[:, 2].mean())), np.mean((block - block.mean(0)) ** 2, axis=0), int(count)))
        grouped[mode] = items
    results = {}
    for mode in modes:
        if mode == reference_mode:
            continue
        combined = grouped[reference_mode] + grouped[mode]
        if not combined:
            continue
        edges = np.unique(np.quantile([it[0] for it in combined], np.linspace(0, 1, bins + 1)))
        entries = []
        for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            entry = {"angular_progress_interval_rad": [float(lo), float(hi)]}
            for m in (reference_mode, mode):
                bucket = [it for it in grouped[m] if lo <= it[0] and (it[0] <= hi if b == len(edges) - 2 else it[0] < hi)]
                entry[m] = {"exact_input_groups": len(bucket), "rows": sum(it[2] for it in bucket),
                            "conditional_rmse_xy_units_rad": np.sqrt(np.average([it[1] for it in bucket], axis=0,
                                                                         weights=[it[2] for it in bucket])).tolist() if bucket else None}
            entry["has_both_modes"] = all(entry[m]["exact_input_groups"] > 0 for m in (reference_mode, mode))
            entries.append(entry)
        results[mode] = {"reference": reference_mode, "substeps": substeps, "bins": entries,
                         "note": "Heldout rows only; post-outcome descriptive strata. Empty/one-sided bins do not support matched-progress comparisons."}
    return results


def normalization(x, y):
    x, y = x.double(), y.double()
    mean_x = x.mean(0)
    raw_scale_x = x.std(0, correction=0)
    # Constant training coordinates (e.g. the 0-degree orientation) must not
    # amplify an unseen but ordinary 30-degree pose by a factor of one million.
    # Use physical unit scale for unseen constant dimensions, never test data.
    scale_x = torch.where(raw_scale_x < 1e-6, torch.ones_like(raw_scale_x), raw_scale_x)
    mean_y = y.mean(0)
    scale_y = y.std(0, correction=0).clamp_min(torch.tensor([1e-4, 1e-4, 1e-3], dtype=torch.float64))
    return {"x_mean": mean_x.float(), "x_scale": scale_x.float(),
            "y_mean": mean_y.float(), "y_scale": scale_y.float()}


def body_frame(x, y=None):
    """Canonicalize the documented 18D coarse interface about the initial bar.

    This is a declared geometric prior, not a learned invariance. VMAS's
    componentwise friction clipping need not be exactly rotation-equivariant.
    """
    if x.ndim != 2 or x.shape[1] != 18:
        raise ValueError("body_frame requires the documented 18D coarse input")
    angle = torch.atan2(x[:, 6], x[:, 7])
    c, s = angle.cos(), angle.sin()

    def rotate(v):
        return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], 1)

    transformed = x.clone()
    for start in (0, 2, 4, 8, 16):
        transformed[:, start:start + 2] = rotate(x[:, start:start + 2] - x[:, 4:6])
    for start in (6, 10):
        local_angle = torch.atan2(x[:, start], x[:, start + 1]) - angle
        transformed[:, start] = local_angle.sin()
        transformed[:, start + 1] = local_angle.cos()
    transformed[:, 12:14] = rotate(x[:, 12:14])
    if y is None:
        return transformed
    target = y.clone()
    target[:, :2] = rotate(y[:, :2])
    return transformed, target


def world_frame_response(x, prediction):
    angle = torch.atan2(x[:, 6], x[:, 7])
    c, s = angle.cos(), angle.sin()
    world = prediction.clone()
    world[:, 0] = c * prediction[:, 0] - s * prediction[:, 1]
    world[:, 1] = s * prediction[:, 0] + c * prediction[:, 1]
    return world


def fit_mlp(x_train, y_train, norm, width=32, steps=600, seed=0, batch_size=256, lr=.003):
    torch.manual_seed(seed)
    model = ResponseMLP(x_train.shape[1], width)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    xs = (x_train.float() - norm["x_mean"]) / norm["x_scale"]
    ys = (y_train.float() - norm["y_mean"]) / norm["y_scale"]
    generator = torch.Generator().manual_seed(seed + 60007)
    model.train()
    for _ in range(steps):
        batch = torch.randint(len(xs), (min(batch_size, len(xs)),), generator=generator)
        prediction = model(xs[batch])
        loss = (prediction - ys[batch]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, norm):
    return model((x.float() - norm["x_mean"]) / norm["x_scale"]) * norm["y_scale"] + norm["y_mean"]


@torch.no_grad()
def predict_checkpoint(checkpoint, x):
    """Predict world-frame bar displacement from a saved, frozen model entry."""
    model = ResponseMLP(checkpoint["input_dim"], checkpoint["width"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    representation = checkpoint.get("representation", "world")
    if representation == "body_frame_coarse18":
        return world_frame_response(x, predict(model, body_frame(x), checkpoint["normalization"]))
    if representation != "world":
        raise ValueError(f"unknown checkpoint representation {representation}")
    return predict(model, x, checkpoint["normalization"])


def error_metrics(y, prediction, y_scale):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if not len(y):
        return None
    error = prediction - y
    scale = np.asarray(y_scale)
    return {"rows": len(y), "rmse_xy_units_rad": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "translation_rmse_units": float(np.sqrt(np.mean((error[:, :2] ** 2).sum(1)))),
            "mae_xy_units_rad": np.mean(abs(error), axis=0).tolist(),
            "pooled_train_unit_mse": float(np.mean((error / scale) ** 2)),
            "zero_response_rmse_xy_units_rad": np.sqrt(np.mean(y ** 2, axis=0)).tolist(),
            "zero_response_pooled_train_unit_mse": float(np.mean((y / scale) ** 2))}


def summarize_seed_metrics(results):
    summaries = []
    groups = defaultdict(list)
    for row in results:
        groups[(row["mode"], row["model"], row["split"])].append(row)
    for (mode, model, split), entries in sorted(groups.items()):
        rmse = np.array([e["rmse_xy_units_rad"] for e in entries])
        mse = np.array([e["pooled_train_unit_mse"] for e in entries])
        summaries.append({"mode": mode, "model": model, "split": split, "seeds": len(entries),
                          "rows_per_seed": entries[0]["rows"], "rmse_mean_xy_units_rad": rmse.mean(0).tolist(),
                          "rmse_std_over_seeds_xy_units_rad": rmse.std(0).tolist(),
                          "translation_rmse_units_mean": float(np.mean([e["translation_rmse_units"] for e in entries])),
                          "pooled_train_unit_mse_mean": float(mse.mean()),
                          "pooled_train_unit_mse_std_over_seeds": float(mse.std()),
                          "zero_response_pooled_train_unit_mse": entries[0]["zero_response_pooled_train_unit_mse"],
                          "parameters": entries[0]["parameters"]})
    return summaries


def run(data, steps=600, seeds=(0, 1, 2), small_width=32, large_width=128, batch_size=256, lr=.003):
    by_mode = validate_dataset(data)
    coarse = torch.as_tensor(data["X_coarse"]).float()
    full = torch.as_tensor(data["X_full"]).float()
    y = torch.as_tensor(data["Y"]).float()
    rows = data["rows"]
    train = torch.tensor([r["split"] == "train" for r in rows])
    norm_coarse = normalization(coarse[train], y[train])
    norm_full = normalization(full[train], y[train])
    small_budget = parameter_count(coarse.shape[1], small_width)
    full_width = matched_width(full.shape[1], small_budget)
    specs = [("coarse_small", coarse, norm_coarse, small_width, y, "world"),
             ("coarse_large", coarse, norm_coarse, large_width, y, "world"),
             ("full_matched", full, norm_full, full_width, y, "world")]
    if coarse.shape[1] == 18:
        body_x, body_y = body_frame(coarse, y)
        norm_body = normalization(body_x[train], body_y[train])
        specs.append(("coarse_body_small", body_x, norm_body, small_width, body_y, "body_frame_coarse18"))
    results, checkpoints, predictions, alias = [], {}, {}, {}
    splits = sorted(set(r["split"] for r in rows))
    substeps = sorted(set(int(r["substeps"]) for r in rows if "substeps" in r))
    slices = [(split, None) for split in splits + ["all_heldout"]]
    slices += [(split, ss) for split in splits + ["all_heldout"] for ss in substeps]

    def in_slice(row, split, ss):
        return ((row["split"] != "train" if split == "all_heldout" else row["split"] == split)
                and (ss is None or int(row.get("substeps", -1)) == ss))

    def slice_name(split, ss):
        return split if ss is None else f"{split}/substeps={ss}"

    for mode, indices in sorted(by_mode.items()):
        # Sorting by identifiers ensures RNG minibatch selections represent the
        # same physical condition/action in all modes, not just the same row index.
        indices = sorted(indices, key=lambda i: (str(rows[i]["context_id"]), str(rows[i]["action_id"]), str(rows[i]["split"])))
        mode_train = [i for i in indices if rows[i]["split"] == "train"]
        alias[mode] = {}
        for split, ss in slices:
            picked = [i for i in indices if in_slice(rows[i], split, ss)]
            if picked:
                alias[mode][slice_name(split, ss)] = alias_statistics(coarse[picked], y[picked])
        for name, inputs, norm, width, target, representation in specs:
            for seed in seeds:
                model = fit_mlp(inputs[mode_train], target[mode_train], norm, width, steps, seed, batch_size, lr)
                pred = predict(model, inputs[indices], norm)
                if representation == "body_frame_coarse18":
                    pred = world_frame_response(coarse[indices], pred)
                key = f"{mode}/{name}/seed{seed}"
                predictions[key] = {"row_indices": torch.tensor(indices), "prediction": pred}
                checkpoints[key] = {"state_dict": model.state_dict(), "input_dim": inputs.shape[1], "width": width,
                                    "normalization": norm, "train_row_indices": torch.tensor(mode_train),
                                    "optimizer_steps": steps, "seed": seed, "representation": representation}
                for split, ss in slices:
                    picked = [j for j, i in enumerate(indices) if in_slice(rows[i], split, ss)]
                    if not picked:
                        continue
                    metrics = error_metrics(y[[indices[j] for j in picked]], pred[picked], norm_coarse["y_scale"])
                    results.append({"mode": mode, "model": name, "seed": seed, "split": slice_name(split, ss),
                                    "parameters": sum(p.numel() for p in model.parameters()), **metrics})
    report = {
        "schema_version": 1,
        "model_config": {"small_width": small_width, "large_width": large_width, "full_width": full_width,
                         "steps": steps, "seeds": list(seeds), "batch_size": batch_size, "lr": lr,
                         "coarse_features": coarse.shape[1], "full_features": full.shape[1],
                         "shared_target_mean": norm_coarse["y_mean"].tolist(), "shared_target_scale": norm_coarse["y_scale"].tolist(),
                         "selection": "Fixed architecture and optimizer steps; no validation/test-based model selection.",
                         "capacity_comparison": "Full-state model width is the largest integer with parameter count <= coarse_small. Same width/depth per helper mode; per-mode samples are matched context/action rows. Same update count, not equal FLOPs for wide models."},
        "rows_per_mode": {mode: {split: sum(rows[i]["split"] == split for i in ids) for split in splits}
                          for mode, ids in by_mode.items()},
        "seed_metrics": results, "summary": summarize_seed_metrics(results),
        "exact_input_alias_statistics": alias, "matched_progress_alias": matched_progress_alias(data),
        "matched_progress_alias_by_substeps": {str(ss): matched_progress_alias(data, substeps=ss) for ss in substeps},
        "limitations": [
            "A pose-only chosen interface omits velocity; the released 15D world-model input already includes bar/agent velocity. This does not diagnose information loss in the released learned encoder.",
            "The helper is handcrafted and senses velocity. Additional sensing and controller complexity have not been eliminated or learned away. No total-system complexity reduction claim.",
            "Helper feedback may reduce motion or lock the object; prediction accuracy must be interpreted with angular authority, response variance at overlapping progress, force/time, and task completion.",
            "Split labels come from the physics dataset. Geometry holdout alone is not a holdout of friction or motion-history families.",
            "coarse_body_small uses a declared rotation/translation prior and transforms outputs back to world units; componentwise VMAS friction is not guaranteed exactly rotation-equivariant. Other model inputs remain world-coordinate controls.",
            "Empirical conditional variance concerns the sampled nuisance distribution and tested actions. Larger stochastic models can express uncertainty; the bound applies to deterministic point prediction MSE.",
            "Multi-seed spread measures optimization variation, not independent physical environments or a statistical confidence interval.",
            "VMAS lengths are scene units without an SI calibration. Translational errors are not reported as meters or millimeters; angles are radians.",
        ],
    }
    return report, {"checkpoints": checkpoints, "predictions": predictions}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_artifact(path, provenance, artifact):
    """Write tensor/plain-metadata output and verify the safe default loader.

    PyTorch exposes its version as a str subclass (TorchVersion). Persisting
    that subclass can make otherwise portable checkpoints fail weights_only
    loading, so normalize it before torch.save.
    """
    provenance = dict(provenance)
    if "torch_version" in provenance:
        provenance["torch_version"] = str(provenance["torch_version"])
    torch.save({"provenance": provenance, **artifact}, path)
    torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/novelty/abstraction_physics.pt")
    parser.add_argument("--output", default="data/novelty/abstraction_models")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--small-width", type=int, default=32)
    parser.add_argument("--large-width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=.003)
    args = parser.parse_args()
    if args.steps < 1 or min(args.small_width, args.large_width, args.batch_size) < 1:
        parser.error("widths, steps and batch size must be positive")
    torch.set_num_threads(1)
    start = time.perf_counter()
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    report, artifact = run(data, args.steps, args.seeds, args.small_width, args.large_width, args.batch_size, args.lr)
    report["provenance"] = {"dataset": str(Path(args.data).resolve()), "dataset_sha256": _sha256(args.data),
                            "source_sha256": _sha256(__file__), "torch_version": str(torch.__version__),
                            "command_args": vars(args), "elapsed_seconds": time.perf_counter() - start}
    try:
        report["provenance"]["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        report["provenance"]["git_commit"] = None
    report["physics_dataset_config"] = data.get("config", data.get("metadata", {}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    save_artifact(output.with_suffix(".pt"), report["provenance"], artifact)
    print(json.dumps({"output": str(output), "elapsed_seconds": report["provenance"]["elapsed_seconds"],
                      "models": len(artifact["checkpoints"]), "summary": [r for r in report["summary"] if r["split"] == "all_heldout"]}, indent=2))


if __name__ == "__main__":
    main()
