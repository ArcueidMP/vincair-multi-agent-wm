import copy

import pytest
import torch

import vmas_pusht.abstraction_composition as composition
import vmas_pusht.rebrace as rebrace
from vmas_pusht.abstraction_models import ResponseMLP
from vmas_pusht.physical_abstraction import norm_bound, setup_scene
from vmas_pusht.scenario import make


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def checkpoint():
    model = ResponseMLP(18, 4)
    for parameter in model.parameters():
        parameter.data.zero_()
    return {"input_dim": 18, "width": 4, "state_dict": model.state_dict(),
            "representation": "body_frame_coarse18",
            "normalization": {"x_mean": torch.zeros(18), "x_scale": torch.ones(18),
                              "y_mean": torch.tensor([0., 0., .01]), "y_scale": torch.ones(3)}}


def tiny_run(**overrides):
    arguments = dict(supports=["world_anchor"], planners=["analytic_feedback"],
                     angles_deg=[0.], nuisances=[(0., 0., 0.)], rotation_steps=5,
                     retreat_steps=2, max_recovery=6, substeps=5)
    arguments.update(overrides)
    return composition.run_group(**arguments)


def test_fixed_model_input_leaves_physical_substeps_unchanged(monkeypatch):
    env = make(num_envs=2, device="cpu", substeps=40)
    anchor = setup_scene(env)
    rows = [dict(support="world_anchor", planner="frozen_wm") for _ in range(2)]
    seen = []
    def record(_, x):
        seen.append(x.clone())
        return torch.zeros(len(x), 3)
    monkeypatch.setattr(composition, "predict_checkpoint", record)
    args = (env, anchor, torch.full((2,), .08), env.scenario.t_pose().clone(),
            rows, {"world_anchor": {}}, torch.tensor([True, False]))
    _, chosen, _, inputs, queried = composition.query_frozen_model(*args, model_substeps=20)
    assert env.scenario.substeps == 40
    assert env.world._substeps == 40
    assert seen[0].shape == (10, 18)
    assert torch.equal(seen[0][:, 15], torch.full((10,), 20.))
    assert queried.tolist() == [True, False]
    assert chosen[1] == -1
    native = composition.query_frozen_model(*args, model_substeps=None)[3]
    assert torch.equal(native[:, :, 15], torch.full((2, 10), 40.))
    torch.testing.assert_close(inputs[:, :, :15], native[:, :, :15], atol=0, rtol=0)
    torch.testing.assert_close(inputs[:, :, 16:], native[:, :, 16:], atol=0, rtol=0)


def test_timed_machine_is_bounded_physical_and_retains_failed_rows(monkeypatch):
    calls = []
    original = composition.setup_scene
    def once(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(composition, "setup_scene", once)
    result = tiny_run(variant="timer_baseline", supports=["passive", "world_anchor"],
                      planners=["analytic_feedback", "zero_push"])
    assert calls == [1]
    assert len(result["rows"]) == 4
    assert result["trajectory"].shape == (4, 19, 24)
    assert result["final_env_steps"].tolist() == [18.] * 4
    assert result["active"].all()
    for i, row in enumerate(result["rows"]):
        assert row["total_steps"] == row["maximum_steps"] == 18
        assert row["phase_steps"] == dict(rotate_first=5, retreat=2, recovery=6, rotate_final=5)
        assert row["phase_end_steps"] == dict(rotate_first=5, retreat=7, recovery=13, rotate_final=18)
        assert row["failure_reasons"]
        assert not row["task_success"]
        rotation = (result["phases"][i] == 0) | (result["phases"][i] == 3)
        if row["support"] == "passive":
            assert torch.count_nonzero(result["actions"][i, rotation, 1]) == 0
        if row["planner"] == "zero_push":
            assert torch.count_nonzero(result["actions"][i, rotation, 0]) == 0
    assert torch.isfinite(result["trajectory"]).all()
    assert result["actions"].norm(dim=-1).max() <= 1.000001


def test_late_final_pose_cannot_repair_failed_first_goal(monkeypatch):
    original = composition.task_metrics
    def final_only(pose, initial, relative_goal, **kwargs):
        _, error, translation = original(pose, initial, relative_goal, **kwargs)
        return relative_goal > .05, error, translation
    monkeypatch.setattr(composition, "task_metrics", final_only)
    result = tiny_run(variant="guarded_all")
    row = result["rows"][0]
    assert row["terminal_success"]
    assert row["ever_final_goal_success"]
    assert row["final_goal_dwell_success"]
    assert not row["first_goal_success_at_phase_boundary"]
    assert not row["first_goal_ever_success"]
    assert not row["task_success"]


def test_corrected_timer_rechecks_contact_instead_of_latching_past_ready(monkeypatch):
    class BriefReadiness(rebrace.RebraceController):
        def update(self, active_mask=None):
            result = super().update(active_mask)
            # A previously stable contact is lost before the fixed deadline.
            result["ready"] = (self.elapsed == 2) & active_mask
            result["dwell"] = torch.where(result["ready"], 5, 0)
            return result
    monkeypatch.setattr(rebrace, "RebraceController", BriefReadiness)
    result = tiny_run(variant="corrected_timer", max_recovery=8)
    row = result["rows"][0]
    assert row["ever_contact_reacquired"]
    assert not row["contact_reacquired"]
    assert row["phase_timeout"]["recovery"]
    assert row["phase_steps"]["recovery"] == 8


def test_handoff_preserves_controller_anchor_and_actual_helper_law(monkeypatch):
    class ImmediateReadiness(rebrace.RebraceController):
        def update(self, active_mask=None):
            result = super().update(active_mask)
            self.anchor[active_mask] = self.env.world.agents[1].state.pos[active_mask] + torch.tensor([-.01, .01])
            result["anchor"] = self.anchor.clone()
            result["ready"] = active_mask.clone()
            result["dwell"] = torch.where(active_mask, 5, 0)
            return result
    monkeypatch.setattr(rebrace, "RebraceController", ImmediateReadiness)
    result = tiny_run(variant="guarded_recovery", supports=["world_anchor", "passive"])
    for i, row in enumerate(result["rows"]):
        assert row["phase_steps"]["recovery"] == 1
        start = row["phase_end_steps"]["recovery"]
        anchor = torch.tensor(row["helper_anchor_at_handoff"])
        torch.testing.assert_close(result["helper_anchor"][i, start:],
                                   anchor.expand_as(result["helper_anchor"][i, start:]))
        state = result["trajectory"][i, start]
        expected = norm_bound(30. * (anchor - state[6:8]) - 4. * state[8:10])
        if row["support"] == "passive":
            expected.zero_()
        torch.testing.assert_close(result["actions"][i, start, 1], expected, atol=1e-6, rtol=0)


def test_frozen_weights_normalizers_and_continuation_query_anchor_are_recorded():
    models = {"world_anchor": checkpoint()}
    before = copy.deepcopy(models)
    result = tiny_run(variant="corrected_timer", planners=["frozen_wm"], checkpoints=models)
    assert composition.checkpoint_digest(models) == composition.checkpoint_digest(before)
    row = result["rows"][0]
    assert row["model_substeps_input"] == 20
    assert row["substeps"] == 5
    assert row["model_conditioning"] == "fixed_reference"
    assert row["model_macro_queries"] == 2
    final_query = result["model_inputs"][-1]
    anchor = result["helper_anchor"][0, final_query["step"]]
    torch.testing.assert_close(final_query["inputs"][0, :, 16:18], anchor.expand(10, 2))
    assert row["continuation_is_outside_reset_only_training_support"]


def test_batched_early_endpoints_exclude_later_padding_and_preserve_all_rows(monkeypatch):
    class Ready(rebrace.RebraceController):
        def update(self, active_mask=None):
            result = super().update(active_mask)
            result["ready"] = active_mask.clone()
            result["dwell"] = torch.where(active_mask, 5, 0)
            return result
    original = composition.task_metrics
    def varied(pose, initial, relative_goal, **kwargs):
        _, error, translation = original(pose, initial, relative_goal, **kwargs)
        success = relative_goal > .05
        success[0] = True
        return success, error, translation
    monkeypatch.setattr(rebrace, "RebraceController", Ready)
    monkeypatch.setattr(composition, "task_metrics", varied)
    result = tiny_run(variant="guarded_all", supports=["world_anchor", "passive"],
                      rotation_steps=10, goal_dwell=2)
    lengths = [row["total_steps"] for row in result["rows"]]
    assert lengths[0] < lengths[1]
    assert len(result["rows"]) == 2
    for i, row in enumerate(result["rows"]):
        steps = row["total_steps"]
        assert result["active"][i, :steps].all()
        assert not result["active"][i, steps:].any()
        assert torch.count_nonzero(result["actions"][i, steps:]) == 0
        torch.testing.assert_close(result["terminal_state"][i], result["trajectory"][i, steps])
        assert sum(row["phase_steps"].values()) == steps
    assert not result["rows"][1]["task_success"]


def test_invalid_or_unbounded_protocol_rejected_before_execution():
    with pytest.raises(ValueError, match="variant"):
        tiny_run(variant="until_it_works")
    with pytest.raises(ValueError, match="multiple"):
        tiny_run(rotation_steps=7)
    with pytest.raises(ValueError, match="handoff"):
        tiny_run(max_recovery=5)
    with pytest.raises(ValueError, match="no frozen checkpoint"):
        tiny_run(planners=["frozen_wm"])
