"""Local-only split, genuine history, parameter budget and frozen-inference checks."""
import copy

import pytest
import torch

from vmas_pusht.abstraction_models import ResponseMLP, parameter_count, save_artifact
from vmas_pusht.stable_local_wm import (
    MODELS, PRELUDE_STEPS, collect_local, deduplicate_prefixes, empty_history,
    entry_features, full_body_features, history_body_features, history_record,
    local_contexts, local_programs, predict_entry, push_history, registered_protocol,
    train_models, validate_local_dataset,
)


def small_contexts():
    return [dict(context_id=i, angle_deg=0., primary_x_offset=0.,
                 velocity_omega=[0., 0., .03*i], context_split="train") for i in range(2)]


def small_programs():
    return [dict(program_id=0, direction_deg=0., strength=.3, duration_steps=5, action_split="train"),
            dict(program_id=1, direction_deg=0., strength=.3, duration_steps=10, action_split="train"),
            dict(program_id=2, direction_deg=10., strength=.45, duration_steps=10, action_split="val"),
            dict(program_id=3, direction_deg=35., strength=.75, duration_steps=15, action_split="test_action")]


@pytest.fixture(scope="module")
def small_data():
    torch.set_num_threads(1)
    return collect_local("original", substeps=2, contexts=small_contexts(), programs=small_programs())


def test_registration_and_groups_do_not_cross_program_families():
    registration = registered_protocol()
    assert registration["parameter_budget"] == 1763
    assert registration["helper"]["learned_parameters"] == 0
    programs = local_programs()
    families = {}
    for row in programs:
        families.setdefault((row["direction_deg"], row["strength"]), set()).add(row["action_split"])
    assert all(len(splits) == 1 for splits in families.values())
    assert .15 in registration["train"]["strengths"]
    assert all(c["angle_deg"] == 0 for c in local_contexts() if c["context_split"] == "train")


def test_history_records_have_true_start_geometry_and_mean_joint_action():
    geometry = torch.arange(24).reshape(2, 12).float()
    action = torch.arange(40).reshape(2, 5, 2, 2).float()
    rec = history_record(geometry, action)
    torch.testing.assert_close(rec[:, :12], geometry)
    torch.testing.assert_close(rec[:, 12:16], action.flatten(-2).mean(1))
    assert torch.all(rec[:, 16] == 1)
    h = empty_history(torch.cat((geometry, torch.zeros(2, 6)), 1))
    h = push_history(h, rec)
    assert torch.all(h[:, 0, 16] == 0)
    torch.testing.assert_close(h[:, 1], rec)


def test_collection_prelude_dedup_constant_future_and_budget(small_data):
    data = small_data
    assert data["exact_matched_initial_physical_states"]
    assert data["duplicate_prefixes_removed"] == 6
    assert data["nominal_window_count"] == len(data["rows"]) + 6
    assert torch.all(data["X_history"][:, :, 16] == 1)
    assert all(r["environment_start_step"] >= PRELUDE_STEPS for r in data["rows"])
    assert all(r["window_end"] - r["window_start"] == 5 for r in data["rows"])
    assert all(e["total_simulated_steps"] == PRELUDE_STEPS + e["duration_steps"] for e in data["episodes"])
    assert all(e["max_action_norm"] <= 1.000001 and e["finite"] for e in data["episodes"])
    budgets = validate_local_dataset(data)
    assert len(set(len(v) for v in budgets.values())) == 1
    # These records come from actual prelude simulation, not three copies of
    # a pose; at least one visible body moves under the initial angular motion.
    first = [i for i,r in enumerate(data["rows"]) if r["window_start"] == 0]
    assert (data["X_history"][first, 0, :12] != data["X_history"][first, 1, :12]).any()


def test_dataset_rejects_split_leakage_and_unequal_support_windows(small_data):
    data = copy.deepcopy(small_data)
    data["rows"][0]["split"] = "test_action"
    with pytest.raises(ValueError, match="family"):
        validate_local_dataset(data)
    data = copy.deepcopy(small_data)
    data["rows"][0]["window_id"] = "unmatched"
    with pytest.raises(ValueError, match="unequal"):
        validate_local_dataset(data)


def test_invalid_history_or_complete_input_fails_instead_of_filling_unknowns(small_data):
    coarse, full = small_data["X_coarse"][:2], small_data["X_full"][:2]
    with pytest.raises(ValueError, match="requires"):
        history_body_features(coarse, None)
    with pytest.raises(ValueError, match="requires"):
        full_body_features(coarse, None)
    bad = full.clone(); bad[:, 0] += .1
    with pytest.raises(ValueError, match="prefix"):
        full_body_features(coarse, bad)


def test_fixed_resolution_feature_does_not_change_predictions(small_data):
    data = small_data
    _, artifacts = train_models(data, steps=2, seeds=[0])
    for entry in artifacts["checkpoints"].values():
        x, f, h = data["X_coarse"][:4], data["X_full"][:4], data["X_history"][:4]
        expected = predict_entry(entry, x, f, h)
        other_x, other_f = x.clone(), f.clone()
        other_x[:, 15] = 320; other_f[:, 15] = 320
        torch.testing.assert_close(predict_entry(entry, other_x, other_f, h), expected, atol=0, rtol=0)


def test_parameter_match_and_train_only_norm_and_fixed_fits(small_data, tmp_path):
    report, original = train_models(small_data, steps=3, seeds=[0])
    assert report["training"]["learned_parameters_by_model"] == {
        "coarse_small": 1763, "coarse_body_small": 1763, "full_matched": 1731, "history_matched": 1741}
    changed = copy.deepcopy(small_data)
    for i, row in enumerate(changed["rows"]):
        if row["split"] != "train":
            changed["Y"][i] += 50
    _, perturbed = train_models(changed, steps=3, seeds=[0])
    for key, entry in original["checkpoints"].items():
        assert all(small_data["rows"][i]["split"] == "train" for i in entry["train_row_indices"])
        for name, weight in entry["state_dict"].items():
            torch.testing.assert_close(weight, perturbed["checkpoints"][key]["state_dict"][name], atol=0, rtol=0)
        torch.testing.assert_close(original["predictions"][key]["prediction"],
                                   perturbed["predictions"][key]["prediction"], atol=0, rtol=0)
    path = tmp_path / "frozen.pt"
    save_artifact(path, {"torch_version": str(torch.__version__)}, original)
    assert set(torch.load(path)["checkpoints"]) == set(original["checkpoints"])


def test_history_features_are_global_rigid_transform_invariant(small_data):
    coarse = small_data["X_coarse"][:3].clone()
    history = small_data["X_history"][:3].clone()
    full = small_data["X_full"][:3].clone()
    angle = .6
    rotation = torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle))],
                             [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle))]])
    translation = torch.tensor([.3, -.2])
    moved, past, velocities = coarse.clone(), history.clone(), full.clone()
    for start in (0, 2, 4, 8, 16):
        moved[:, start:start+2] = coarse[:, start:start+2] @ rotation.T + translation
    for start in (6, 10):
        a = torch.atan2(coarse[:, start], coarse[:, start+1]) + angle
        moved[:, start:start+2] = torch.stack((a.sin(), a.cos()), 1)
    moved[:, 12:14] = coarse[:, 12:14] @ rotation.T
    for h in range(2):
        for start in (0, 2, 4, 8):
            past[:, h, start:start+2] = history[:, h, start:start+2] @ rotation.T + translation
        for start in (6, 10):
            a = torch.atan2(history[:, h, start], history[:, h, start+1]) + angle
            past[:, h, start:start+2] = torch.stack((a.sin(), a.cos()), 1)
        for start in (12, 14):
            past[:, h, start:start+2] = history[:, h, start:start+2] @ rotation.T
    velocities[:, :18] = moved
    for start in (18, 21, 24, 27):
        velocities[:, start:start+2] = full[:, start:start+2] @ rotation.T
    torch.testing.assert_close(history_body_features(coarse, history), history_body_features(moved, past), atol=2e-7, rtol=1e-5)
    torch.testing.assert_close(full_body_features(coarse, full), full_body_features(moved, velocities), atol=2e-7, rtol=1e-5)
