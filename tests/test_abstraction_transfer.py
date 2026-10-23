import copy

import pytest
import torch

import vmas_pusht.abstraction_transfer as transfer
from vmas_pusht.abstraction_models import ResponseMLP


def constant_checkpoint():
    model = ResponseMLP(18, 4)
    for value in model.parameters():
        value.data.zero_()
    return {"input_dim": 18, "width": 4, "state_dict": model.state_dict(),
            "representation": "body_frame_coarse18",
            "normalization": {"x_mean": torch.zeros(18), "x_scale": torch.ones(18),
                              "y_mean": torch.tensor([0., 0., .01]), "y_scale": torch.ones(3)}}


def test_candidates_are_fixed_distinct_and_euclidean_bounded():
    candidates = transfer.main_candidates(torch.tensor([0., .7]))
    assert candidates.shape == (2, 10, 2)
    assert torch.all(candidates.norm(dim=-1) <= .900001)
    torch.testing.assert_close(candidates[:, 0], torch.zeros(2, 2))
    for i in range(2):
        assert len(torch.unique(candidates[i], dim=0)) == 10
    torch.testing.assert_close(candidates, transfer.main_candidates(torch.tensor([0., .7])), rtol=0, atol=0)


def test_new_goal_success_uses_own_strict_translation_and_angle_gates():
    initial = torch.zeros(3, 3)
    pose = torch.tensor([[0., 0., .08], [.06, 0., .08], [0., 0., .10]])
    success, angle, translation = transfer.task_metrics(pose, initial, torch.full((3,), .08))
    assert success.tolist() == [True, False, False]
    assert translation[1] > .05
    assert angle[2] > .015
    # The original environment's goal remains at setup pose, so its success
    # flag would incorrectly report this nonzero new goal as already solved.
    success, _, _ = transfer.task_metrics(torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([.04]))
    assert not success.item()


def test_transfer_does_not_reset_state_between_macros_and_keeps_force_constant(monkeypatch):
    original = transfer.setup_scene
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(transfer, "setup_scene", counted)
    result = transfer.run_group(supports=["world_anchor"], planners=["analytic_feedback", "zero_push"],
                                angles_deg=[0.], goals=[.04], nuisances=[(0., 0., 0.)], rotation_steps=10)
    assert calls == [1]
    assert result["final_env_steps"].tolist() == [10., 10.]
    assert result["trajectory"].shape == (2, 11, 24)
    assert torch.isfinite(result["trajectory"]).all()
    for start in [0, 5]:
        torch.testing.assert_close(result["actions"][:, start:start + 5, 0],
                                   result["actions"][:, start:start + 1, 0].expand(-1, 5, -1), rtol=0, atol=0)
    zero_idx = next(i for i, row in enumerate(result["rows"]) if row["planner"] == "zero_push")
    assert torch.equal(result["actions"][zero_idx, :, 0], torch.zeros(10, 2))
    assert torch.all(result["actions"].norm(dim=-1) <= 1.000001)


def test_composition_is_physical_and_requires_all_completion_gates(monkeypatch):
    original = transfer.setup_scene
    setup_count = []
    def once(*args, **kwargs):
        setup_count.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(transfer, "setup_scene", once)
    result = transfer.run_group(supports=["world_anchor"], planners=["analytic_feedback"], angles_deg=[0.],
                                nuisances=[(0., 0., 0.)], rotation_steps=5, retreat_steps=2, reacquire_steps=3,
                                composed=True)
    assert setup_count == [1]
    assert result["final_env_steps"].item() == 15
    assert result["trajectory"].shape == (1, 16, 24)
    assert result["phases"] == ["rotate_first"] * 5 + ["retreat"] * 2 + ["reacquire"] * 3 + ["rotate_final"] * 5
    assert result["actions"][0, 5:7, 1].abs().sum() > 0  # Actual finite-force retreat.
    row = result["rows"][0]
    expected = all(row[k] for k in ["terminal_success", "first_goal_success_at_phase_boundary", "retreated_clear",
                                    "contact_reacquired", "primary_repositioned"])
    assert row["task_success"] is expected
    # Memory updates once, after physical reacquisition, not at every query.
    torch.testing.assert_close(result["helper_anchor"][:, :10], result["helper_anchor"][:, :1].expand(-1, 10, -1))
    torch.testing.assert_close(result["helper_anchor"][:, 10:], result["helper_anchor"][:, 10:11].expand(-1, 6, -1))


def test_frozen_checkpoint_and_initial_matched_states_unchanged():
    checkpoint = constant_checkpoint()
    before = copy.deepcopy(checkpoint)
    result = transfer.run_group(checkpoints={"world_anchor": checkpoint}, supports=["world_anchor"],
                                planners=["frozen_wm", "analytic_feedback", "zero_push"], angles_deg=[0.],
                                goals=[.04], nuisances=[(0., 0., 0.)], rotation_steps=5)
    for key in checkpoint["state_dict"]:
        torch.testing.assert_close(checkpoint["state_dict"][key], before["state_dict"][key], atol=0, rtol=0)
    for key in checkpoint["normalization"]:
        torch.testing.assert_close(checkpoint["normalization"][key], before["normalization"][key], atol=0, rtol=0)
    initial = result["trajectory"][:, 0]
    torch.testing.assert_close(initial, initial[:1].expand_as(initial), atol=0, rtol=0)
    assert result["macro_prediction_valid"].tolist() == [[True], [False], [False]]
    assert result["selected_macros"][0]["chosen_index"].tolist()[1:] == [-1, -1]


def test_preload_anchor_memory_matches_training_reference_once():
    result = transfer.run_group(supports=["preloaded_anchor", "world_anchor"], planners=["zero_push"],
                                angles_deg=[0.], goals=[.04], nuisances=[(0., 0., 0.)], rotation_steps=5)
    initial = result["trajectory"][:, 0]
    torch.testing.assert_close(initial[0], initial[1], atol=0, rtol=0)
    anchor_difference = result["helper_anchor"][0, 0] - result["helper_anchor"][1, 0]
    torch.testing.assert_close(anchor_difference, torch.tensor([-.01, .01]))
    # A preload command acts despite the same actual initial body positions.
    torch.testing.assert_close(result["actions"][0, 0, 1], torch.tensor([-.3, .3]), atol=1e-6, rtol=0)
    torch.testing.assert_close(result["actions"][1, 0, 1], torch.zeros(2), atol=1e-6, rtol=0)


def test_absent_model_is_explicit_and_uneven_macro_budget_rejected(tmp_path):
    checkpoint, provenance, skipped = transfer.load_checkpoints(tmp_path / "absent.pt", tmp_path / "also_absent.pt")
    assert not checkpoint and not provenance and len(skipped) == 2
    with pytest.raises(ValueError, match="no frozen checkpoint"):
        transfer.run_group(supports=["world_anchor"], planners=["frozen_wm"])
    with pytest.raises(ValueError, match="multiple of five"):
        transfer.run_group(supports=["world_anchor"], planners=["analytic_feedback"], rotation_steps=7)
