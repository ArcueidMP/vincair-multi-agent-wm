"""Local-only option data, real histories, model budgets, and frozen inference."""
import copy

import pytest
import torch

from vmas_pusht.option_models import (collect_support, entry_features, local_contexts,
    local_programs, load_model_pack, predict_entry, prepare_local_faces, registered_protocol, train_models,
    validate_dataset)


@pytest.fixture(scope="module")
def data():
    torch.set_num_threads(1)
    return collect_support("world_anchor", substeps=2,
        contexts=local_contexts(dict(train=2, val=1, test=2)), chunk_size=512)


def test_protocol_separates_contexts_and_unseen_programs():
    contexts = local_contexts()
    assert len(contexts) == 48+12+24
    assert len({r["context_id"] for r in contexts}) == len(contexts)
    assert all(abs(r["angle_rad"]) <= .2 for r in contexts if r["split"] in ("train", "val"))
    assert len(local_programs("train")) == 39
    programs = {}
    for split in ("train", "val", "test"):
        programs[split] = {(r["direction_deg"], r["strength"], r["push_steps"]) for r in local_programs(split)}
    assert not programs["train"] & programs["val"]
    assert not programs["train"] & programs["test"]
    assert not programs["val"] & programs["test"]
    assert .01 in registered_protocol()["anchor_insets"]["world_anchor"]
    assert "task" not in registered_protocol()["train"]


def test_collected_options_use_real_histories_modes_and_elapsed_clocks(data):
    validate_dataset(data)
    assert data["X_coarse"].shape[1] == 24
    assert data["X_full"].shape[1] == 38
    assert data["X_history"][:, :, 16].eq(1).all()
    assert (data["X_history"][:, 0, :12] != data["X_history"][:, 1, :12]).any()
    assert all(r["environment_start_step"] >= 20 for r in data["rows"])
    for i, row in enumerate(data["rows"]):
        assert row["environment_end_step"]-row["environment_start_step"] == row["actual_steps"]
        assert row["finite"] and row["max_action_norm"] <= 1.000001
        if row["option_completed"]:
            assert row["pushed_steps"] == row["push_steps"]
            assert row["actual_steps"] >= row["push_steps"]
            if row["route_id"] == 0:
                assert row["actual_steps"] == row["push_steps"]
        assert data["Y"][i, 4] == float(row["option_completed"])
    assert not data["preparation"]["generated_from_task_trajectories"]


def test_failed_route_attempts_and_failed_preparation_are_not_silently_dropped():
    contexts = local_contexts(dict(train=1, val=0, test=0))
    contexts[0]["outward_error"] = .8  # Legal far-away start; route budget can expire.
    sample = collect_support("passive", substeps=2, contexts=contexts, chunk_size=256)
    failures = [i for i, r in enumerate(sample["rows"]) if r["option_failed"]]
    assert failures
    assert sample["Y"][failures, 4].eq(0).all()
    assert len(sample["preparation"]["statuses"]) == 3
    assert any(not r["preparation_completed"] for r in sample["preparation"]["statuses"])
    assert all(r["preparation_completed"] for r in sample["rows"])


def test_acquired_faces_include_genuine_post_push_stop_and_hold_queries():
    contexts = local_contexts(dict(train=7, val=0, test=0))
    prepared = prepare_local_faces("world_anchor", [contexts[i] for i in (0, 3, 6)], substeps=2)
    assert {r["entry_origin"] for r in prepared["continuation_queries"]} == {"after_push", "after_stop", "after_hold"}
    assert len(prepared["continuation_queries"]) == prepared["acquisition_success_count"]
    record = prepared["record"]
    for i, row in enumerate(record["rows"]):
        assert row["environment_start_step"] == 20+row["preparation_steps"]
        assert record["history"][i, :, 16].eq(1).all()
        if row["entry_origin"] == "after_push":
            assert record["history"][i, 1, 12:14].norm() > .1
        if row["entry_origin"] == "after_stop":
            assert record["history"][i, 1, 12:14].eq(0).all()


def test_validation_rejects_split_and_duration_leakage(data):
    changed = copy.deepcopy(data)
    changed["rows"][0]["split"] = "test_action"
    with pytest.raises(ValueError, match="family"):
        validate_dataset(changed)
    changed = copy.deepcopy(data)
    changed["Y"][0, 3] += .2
    with pytest.raises(ValueError, match="Elapsed"):
        validate_dataset(changed)
    changed = copy.deepcopy(data)
    first_val = next(i for i, r in enumerate(changed["rows"]) if r["split"] == "val")
    changed["X_coarse"][first_val] = changed["X_coarse"][0]
    changed["X_full"][first_val, :18] = changed["X_coarse"][0, :18]
    changed["X_full"][first_val, 32:] = changed["X_coarse"][0, 18:]
    with pytest.raises(ValueError, match="Exact"):
        validate_dataset(changed)


def test_training_uses_only_train_labels_and_matched_parameter_budgets(data):
    report, original = train_models(data, steps=3, seeds=[0])
    assert report["learned_parameters"] == dict(coarse_body_small=2021, full_matched=1949,
                                                history_matched=1919, full_large=22149)
    changed = copy.deepcopy(data)
    heldout = [i for i, r in enumerate(changed["rows"]) if r["split"] != "train"]
    changed["Y"][heldout, :3] += 5.
    _, altered = train_models(changed, steps=3, seeds=[0])
    for key, entry in original["checkpoints"].items():
        assert all(data["rows"][int(i)]["split"] == "train" for i in entry["train_row_indices"])
        for name, weight in entry["state_dict"].items():
            torch.testing.assert_close(weight, altered["checkpoints"][key]["state_dict"][name], atol=0, rtol=0)
        torch.testing.assert_close(original["predictions"][key]["prediction"], altered["predictions"][key]["prediction"], atol=0, rtol=0)


def test_predictors_are_frozen_safe_loaded_and_resolution_reference_is_fixed(data, tmp_path):
    _, pack = train_models(data, steps=2, seeds=[0])
    path = tmp_path/"models.pt"
    torch.save(dict(pack, provenance={}, profile=data["profile"], reference_substeps=2), path)
    loaded = load_model_pack(path)
    before = copy.deepcopy(loaded["checkpoints"])
    for key, entry in loaded["checkpoints"].items():
        c, f, h = data["X_coarse"][:7].clone(), data["X_full"][:7].clone(), data["X_history"][:7].clone()
        expected = loaded["predictors"][key](c, f, h)
        c[:, 15] = 320; f[:, 15] = 320
        actual = predict_entry(entry, c, f, h)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert actual.shape == (7, 5)
        assert (actual[:, 3] >= 0).all()
        assert ((actual[:, 4] >= 0) & (actual[:, 4] <= 1)).all()
        for name, weight in entry["state_dict"].items():
            torch.testing.assert_close(weight, before[key]["state_dict"][name], atol=0, rtol=0)


def test_body_features_transform_geometry_velocities_actions_history_and_modes(data):
    c, f, h = data["X_coarse"][:4].clone(), data["X_full"][:4].clone(), data["X_history"][:4].clone()
    angle = torch.tensor(.63)
    rotation = torch.tensor([[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]])
    translation = torch.tensor([.2, -.3])
    moved, full, history = c.clone(), f.clone(), h.clone()
    for start in (0, 2, 4, 8, 16):
        moved[:, start:start+2] = c[:, start:start+2]@rotation.T+translation
    for start in (6, 10):
        a = torch.atan2(c[:, start], c[:, start+1])+angle
        moved[:, start:start+2] = torch.stack((a.sin(), a.cos()), 1)
    moved[:, 12:14] = c[:, 12:14]@rotation.T
    full[:, :18] = moved[:, :18]
    for start in (18, 21, 24, 27):
        full[:, start:start+2] = f[:, start:start+2]@rotation.T
    for index in range(2):
        for start in (0, 2, 4, 8):
            history[:, index, start:start+2] = h[:, index, start:start+2]@rotation.T+translation
        for start in (6, 10):
            a = torch.atan2(h[:, index, start], h[:, index, start+1])+angle
            history[:, index, start:start+2] = torch.stack((a.sin(), a.cos()), 1)
        for start in (12, 14):
            history[:, index, start:start+2] = h[:, index, start:start+2]@rotation.T
    for representation in ("option_body_coarse24", "option_body_full38", "option_body_history58"):
        entry = dict(representation=representation, reference_substeps=2)
        torch.testing.assert_close(entry_features(entry, c, f, h), entry_features(entry, moved, full, history), atol=2e-7, rtol=1e-5)
    torch.testing.assert_close(c[:, 18:], moved[:, 18:], atol=0, rtol=0)


def test_missing_full_or_history_information_is_rejected(data):
    with pytest.raises(ValueError, match="Full"):
        entry_features(dict(representation="option_body_full38", reference_substeps=2), data["X_coarse"][:2])
    with pytest.raises(ValueError, match="requires"):
        entry_features(dict(representation="option_body_history58", reference_substeps=2), data["X_coarse"][:2])
