"""Scientific validity checks: split/scaling leakage, aliasing, budget and pairing."""
import numpy as np
import pytest
import torch

from vmas_pusht.abstraction_models import (
    ResponseMLP, alias_statistics, body_frame, error_metrics, matched_progress_alias,
    matched_width, normalization, parameter_count, run, validate_dataset,
    predict_checkpoint, world_frame_response,
    save_artifact,
)


def tiny_dataset():
    rows, xc, xf, y = [], [], [], []
    for mode in ("passive", "damping"):
        for c in range(6):
            for a in range(4):
                split = "train" if a < 2 else "val" if a == 2 else "test_action"
                rows.append({"mode": mode, "context_id": c, "action_id": a,
                             "split": split, "substeps": 5 if c < 3 else 10})
                xc.append([a / 3, 1., 0.])
                xf.append(xc[-1] + [float(c)])
                y.append([.1 * a + .01 * c, 0., .2 * a + (.01 if mode == "damping" else .1) * c])
    return {"X_coarse": torch.tensor(xc), "X_full": torch.tensor(xf),
            "Y": torch.tensor(y), "rows": rows}


def test_empirical_conditional_variance_requires_exact_aliases():
    x = np.array([[0.], [0.], [1.], [1.]])
    y = np.array([[0., 0., 0.], [2., 4., 6.], [4., 8., 12.], [6., 12., 18.]])
    result = alias_statistics(x, y)
    np.testing.assert_allclose(result["conditional_rmse_xy_units_rad"], [1., 2., 3.])
    assert result["repeated_input_groups"] == 2
    assert result["rows_in_repeated_groups"] == 4
    # A one-ULP difference is not treated as an alias by a loose tolerance.
    x[1] = np.nextafter(0., 1.)
    result = alias_statistics(x, y)
    assert result["rows_in_repeated_groups"] == 2
    assert result["repeated_input_groups"] == 1


def test_no_repeated_inputs_does_not_claim_zero_irreducible_error():
    result = alias_statistics(np.eye(3), np.eye(3))
    assert result["conditional_rmse_xy_units_rad"] is None
    assert result["rows_in_repeated_groups"] == 0


def test_variance_is_sample_weighted_not_equal_group_weight():
    x = np.array([[0.], [0.], [0.], [0.], [1.], [1.]])
    y = np.array([[0., 0., 0.], [0., 0., 0.], [2., 0., 0.], [2., 0., 0.], [0., 0., 0.], [4., 0., 0.]])
    result = alias_statistics(x, y)
    assert result["conditional_mse_xy_units2_rad2"][0] == pytest.approx(2.)


def test_budget_matched_full_width_and_parameter_count():
    budget = parameter_count(16, 32)
    width = matched_width(30, budget)
    assert parameter_count(30, width) <= budget < parameter_count(30, width + 1)
    model = ResponseMLP(30, width)
    assert sum(p.numel() for p in model.parameters()) == parameter_count(30, width)


def test_dataset_rejects_unmatched_modes_and_hidden_full_prefix_change():
    data = tiny_dataset()
    assert set(validate_dataset(data)) == {"passive", "damping"}
    data["rows"][-1]["context_id"] = 99
    with pytest.raises(ValueError, match="matched"):
        validate_dataset(data)
    data = tiny_dataset()
    data["X_full"][0, 0] = 42
    with pytest.raises(ValueError, match="exactly"):
        validate_dataset(data)
    data = tiny_dataset()
    data["X_full"][-1, -1] += 1
    with pytest.raises(ValueError, match="different initial inputs"):
        validate_dataset(data)
    data = tiny_dataset()
    data["rows"][0]["context_is_train"] = False
    with pytest.raises(ValueError, match="heldout"):
        validate_dataset(data)


def test_train_only_normalization_and_shared_target_units():
    x = torch.tensor([[0., 1.], [2., 1.]])
    y = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
    norm = normalization(x, y)
    torch.testing.assert_close(norm["x_mean"], torch.tensor([1., 1.]))
    assert norm["x_scale"][1] > 0
    # Persistence receives the same training-unit metric as every predictor.
    result = error_metrics(y, torch.zeros_like(y), norm["y_scale"])
    assert result["pooled_train_unit_mse"] == result["zero_response_pooled_train_unit_mse"]


def test_fixed_steps_reproducible_and_checkpoints_do_not_train_on_holdout():
    torch.set_num_threads(1)
    data = tiny_dataset()
    report, artifacts = run(data, steps=3, seeds=[2], small_width=4, large_width=8)
    repeated, again = run(data, steps=3, seeds=[2], small_width=4, large_width=8)
    assert report["summary"] == repeated["summary"]
    for key, checkpoint in artifacts["checkpoints"].items():
        assert all(data["rows"][i]["split"] == "train" for i in checkpoint["train_row_indices"])
        torch.testing.assert_close(artifacts["predictions"][key]["prediction"], again["predictions"][key]["prediction"], atol=0, rtol=0)
    # Test targets may change evaluation metrics, but never fit, normalization,
    # sample selection, checkpoint parameters, or a single prediction.
    changed = tiny_dataset()
    for i, row in enumerate(changed["rows"]):
        if row["split"] != "train":
            changed["Y"][i] += 123
    _, changed_artifacts = run(changed, steps=3, seeds=[2], small_width=4, large_width=8)
    for key in artifacts["predictions"]:
        torch.testing.assert_close(artifacts["predictions"][key]["prediction"], changed_artifacts["predictions"][key]["prediction"], atol=0, rtol=0)


def test_matched_progress_reports_empty_overlap_instead_of_extrapolating():
    data = tiny_dataset()
    for i, row in enumerate(data["rows"]):
        if row["mode"] == "damping":
            data["Y"][i, 2] += 10
    result = matched_progress_alias(data, bins=4)["damping"]
    assert result["bins"]
    assert not any(b["has_both_modes"] for b in result["bins"])


def test_body_frame_rotation_translation_and_frozen_prediction_inverse():
    x = torch.zeros(2, 18)
    y_local = torch.tensor([[.2, -.1, .4], [.2, -.1, .4]])
    positions = {0: [-.13, .101], 2: [.101, -.101], 4: [0., 0.], 8: [0., -.2], 16: [.101, -.101]}
    for i, angle in enumerate([0., .7]):
        c, s = np.cos(angle), np.sin(angle)
        rotation = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        translation = torch.tensor([i * .2, i * -.3])
        for start, position in positions.items():
            x[i, start:start + 2] = rotation @ torch.tensor(position) + translation
        for start in (6, 10):
            x[i, start:start + 2] = torch.tensor([s, c])
        x[i, 12:14] = rotation @ torch.tensor([.4, .6])
        x[i, 14:16] = torch.tensor([1., 10.])
    y_world = world_frame_response(x, y_local)
    canonical, target = body_frame(x, y_world)
    torch.testing.assert_close(canonical[0], canonical[1], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(target, y_local)
    # A constant body-frame response must become a rotated world response at
    # inference, using checkpoint metadata rather than an external convention.
    model = ResponseMLP(18, 4)
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint = {"input_dim": 18, "width": 4, "state_dict": model.state_dict(),
                  "representation": "body_frame_coarse18",
                  "normalization": {"x_mean": torch.zeros(18), "x_scale": torch.ones(18),
                                    "y_mean": y_local[0], "y_scale": torch.ones(3)}}
    torch.testing.assert_close(predict_checkpoint(checkpoint, x), y_world)


def test_checkpoint_metadata_is_loadable_with_torch_default_weights_only(tmp_path):
    path = tmp_path / "artifact.pt"
    model = ResponseMLP(3, 4)
    artifact = {"checkpoints": {"example": {"state_dict": model.state_dict()}},
                "predictions": {"example": torch.zeros(2, 3)}}
    save_artifact(path, {"torch_version": torch.__version__, "seeds": [0, 1, 2]}, artifact)
    loaded = torch.load(path, map_location="cpu")
    assert type(loaded["provenance"]["torch_version"]) is str
    assert loaded["provenance"]["torch_version"] == str(torch.__version__)
    torch.testing.assert_close(loaded["predictions"]["example"], torch.zeros(2, 3))
