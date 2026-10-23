import copy

import pytest
import torch

import vmas_pusht.stable_composition as composition
from vmas_pusht.abstraction_models import ResponseMLP
from vmas_pusht.physical_abstraction import physical_state, setup_scene
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.snapshot import restore


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
    arguments = dict(profile="original", supports=["world_anchor"], planners=["analytic_feedback"],
        angles_deg=[0.], nuisances=[(0., 0., 0.)], rotation_steps=5, retreat_steps=2,
        max_recovery=6, substeps=5, model_training_profile="unit_test")
    arguments.update(overrides)
    return composition.run_group(**arguments)


def test_candidate_banks_keep_original_ten_and_add_declared_low_strength():
    angles = torch.tensor([0., .5])
    old = composition.candidates_for_bank(angles, "bank10")
    expanded = composition.candidates_for_bank(angles, "bank13")
    assert old.shape == (2, 10, 2)
    assert expanded.shape == (2, 13, 2)
    assert torch.max(expanded.norm(dim=-1)) <= .900001
    for row in range(2):
        assert torch.unique(expanded[row], dim=0).shape[0] == 13
        for action in old[row]:
            assert ((expanded[row] - action).norm(dim=-1) < 1e-6).any()
    with pytest.raises(ValueError):
        composition.candidates_for_bank(angles, "adaptive_test_tuned")


def test_history_uses_only_completed_past_windows_and_mean_real_actions():
    geometry = [torch.full((2, 12), float(i)) for i in range(14)]
    actions = [torch.full((2, 2, 2), float(i)) for i in range(13)]
    history = composition.history_inputs(geometry, actions)
    torch.testing.assert_close(history[:, 0, :12], torch.full((2, 12), 3.))
    torch.testing.assert_close(history[:, 1, :12], torch.full((2, 12), 8.))
    torch.testing.assert_close(history[:, 0, 12:16], torch.full((2, 4), 5.))
    torch.testing.assert_close(history[:, 1, 12:16], torch.full((2, 4), 10.))
    assert history[:, :, 16].eq(1).all()
    short = composition.history_inputs(geometry[:4], actions[:3])
    assert short[:, :, 16].eq(0).all()
    assert short[:, :, :12].eq(3).all()


def test_prelude_is_physical_accounted_and_goal_reference_is_after_preparation(monkeypatch):
    calls = []
    original = composition.setup_scene
    def once(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(composition, "setup_scene", once)
    result = tiny_run(supports=["passive", "preloaded_anchor"], planners=["zero_push"])
    assert len(calls) == 1
    assert result["prelude_actions"].shape == (2, 10, 2, 2)
    assert result["prelude_actions"][:, :, 0].eq(0).all()
    assert result["prelude_actions"][0, :, 1].eq(0).all()
    for i, row in enumerate(result["rows"]):
        assert row["total_steps"] == row["task_steps"] + 10
        assert row["maximum_steps"] == 28
        assert row["target_reference"] == "physical pose after common zero-primary prelude"
        assert row["norm_force_integral_by_agent"][1] >= row["prelude_force_integral_by_agent"][1]
        torch.testing.assert_close(result["initial_pose"][i, :2], result["prelude_trajectory"][i, -1, 12:14])
        torch.testing.assert_close(result["trajectory"][i, 0], result["prelude_trajectory"][i, -1])
    initial_history = result["counterfactual_snapshots"][0]["history"]
    assert initial_history[:, :, 16].eq(1).all()


def test_external_goals_are_not_conflated_with_active_contact_gate(monkeypatch):
    original = composition.task_metrics
    def always(pose, initial, goal, **kwargs):
        _, error, translation = original(pose, initial, goal, **kwargs)
        return torch.ones(len(pose), dtype=torch.bool), error, translation
    monkeypatch.setattr(composition, "task_metrics", always)
    result = tiny_run(supports=["passive"])
    row = result["rows"][0]
    assert row["ordered_goal_success"]
    assert row["task_success"]
    assert not row["contact_reacquired"]
    assert not row["interface_completion_success"]
    assert "recovery_gate" in row["failure_reasons"]


def test_later_goal_cannot_repair_failed_first_goal(monkeypatch):
    original = composition.task_metrics
    def final_only(pose, initial, goal, **kwargs):
        _, error, translation = original(pose, initial, goal, **kwargs)
        return goal > .05, error, translation
    monkeypatch.setattr(composition, "task_metrics", final_only)
    result = tiny_run()
    row = result["rows"][0]
    assert row["terminal_success"]
    assert row["final_goal_dwell_success"]
    assert not row["ordered_goal_success"]
    assert not row["interface_completion_success"]


def test_counterfactual_snapshot_preserves_selected_full_physics_state_without_aliasing():
    env = make_profile("original", num_envs=2, substeps=5)
    anchor = setup_scene(env, hidden_v=[[0., 0.], [.03, 0.]], hidden_omega=[0., .2])
    geometry = composition.coarse_state(env)
    record = composition.snapshot_record(env, [1], anchor, composition.empty_history(geometry),
                                         [{"id": 0}, {"id": 1}], "test")
    saved = physical_state(env)[1].clone()
    env.scenario.stem.state.vel.add_(4.)
    target = make_profile("original", num_envs=1, substeps=5)
    restore(target, composition.snapshot_from_record(record))
    torch.testing.assert_close(physical_state(target)[0], saved, atol=0, rtol=0)
    assert record["rows"][0]["composition_row"] == 1


def test_legacy_model_is_frozen_and_counterfactuals_include_only_unseen_pushes():
    models = dict(legacy={"world_anchor": checkpoint()}, new={})
    before = copy.deepcopy(models)
    result = tiny_run(planners=["legacy_wm"], checkpoints=models)
    cf = composition.counterfactual_group(result, models)
    assert composition.checkpoint_digest(models) == composition.checkpoint_digest(before)
    assert len(cf["rows"]) == 8
    assert {row["direction_deg"] for row in cf["rows"]} == {-35., 35.}
    assert {row["snapshot_stage"] for row in cf["rows"]} == {"post_prelude", "post_recovery"}
    assert all(row["finite"] for row in cf["rows"])
    for row in result["rows"]:
        assert row["model_candidate_predictions"] == row["model_macro_queries"] * 13
    summary = composition.summarize_counterfactual(cf["rows"])
    assert len(summary) == 2
    assert all(row["rows"] == 4 for row in summary)


def test_missing_new_checkpoint_fails_before_simulation():
    with pytest.raises(ValueError, match="Missing retrained model"):
        tiny_run(planners=["new_full_matched"])
    with pytest.raises(ValueError, match="two real history windows"):
        tiny_run(prelude_steps=5)


def test_new_model_query_receives_real_full_state_history_and_fixed_reference(monkeypatch):
    import vmas_pusht.stable_local_wm as local
    env = make_profile("dry_light", num_envs=1, substeps=40)
    anchor = setup_scene(env, hidden_v=(.03, 0.), hidden_omega=.2)
    row = dict(support="world_anchor", planner="new_full_matched")
    models = dict(new={"world_anchor/full_matched/seed0": {"reference_substeps": 160}}, legacy={})
    history = composition.empty_history(composition.coarse_state(env))
    history[:, :, 16] = 1
    seen = []
    def record(entry, coarse, full=None, history=None):
        seen.append((entry, coarse.clone(), full.clone(), history.clone()))
        return torch.zeros(len(coarse), 3)
    monkeypatch.setattr(local, "predict_entry", record)
    composition.query_frozen_model(env, anchor, torch.tensor([.08]), env.scenario.t_pose(),
        [row], models, torch.tensor([True]), history=history, candidate_bank="bank13")
    entry, coarse, full, observed_history = seen[0]
    assert entry["reference_substeps"] == 160
    assert coarse.shape == (13, 18)
    assert coarse[:, 15].eq(160).all()
    assert full.shape == (13, 32)
    assert observed_history.shape == (13, 2, 17)
    assert observed_history[:, :, 16].eq(1).all()
    torch.testing.assert_close(full[:, :18], coarse)
    assert full[:, -2:].eq(torch.tensor([.25, .10])).all()
    assert env.scenario.substeps == 40
    assert env.world._substeps == 40


def test_merging_task_snapshots_preserves_every_time_and_state():
    env = make_profile("original", num_envs=1, substeps=5)
    anchor = setup_scene(env)
    env.step([torch.zeros(1, 2), torch.zeros(1, 2)])
    initial = composition.snapshot_record(env, [0], anchor,
        composition.empty_history(composition.coarse_state(env)), [{"id": 0}], "initial")
    env.step([torch.tensor([[0., -.3]]), torch.zeros(1, 2)])
    later = composition.snapshot_record(env, [0], anchor,
        composition.empty_history(composition.coarse_state(env)), [{"id": 0}], "later")
    merged = composition.merge_snapshot_records([initial, later])
    target = make_profile("original", num_envs=2, substeps=5)
    restore(target, composition.snapshot_from_record(merged))
    assert target.steps.tolist() == [1., 2.]
    assert [row["snapshot_stage"] for row in merged["rows"]] == ["initial", "later"]
    torch.testing.assert_close(physical_state(target)[1], physical_state(env)[0], atol=0, rtol=0)
