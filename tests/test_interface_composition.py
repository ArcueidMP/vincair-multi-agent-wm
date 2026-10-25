import copy

import pytest
import torch

import vmas_pusht.interface_composition as composition
from vmas_pusht.physical_abstraction import physical_state
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.snapshot import restore
from vmas_pusht.stable_composition import snapshot_from_record


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def zero_selector(env, interface, rows, initial_pose, goals, remaining, history, query_mask,
                  model_pack=None, allow_routes=None):
    options = torch.tensor([0., 0., 0., 5.]).expand(env.num_envs, 4).clone()
    return options, dict(selected_prediction=torch.full((env.num_envs, 5), float("nan")),
                         chosen_index=torch.zeros(env.num_envs, dtype=torch.long))


def tiny(monkeypatch, **overrides):
    monkeypatch.setattr(composition, "select_options", zero_selector)
    args = dict(profile="dry_light", substeps=5, planners=["feedback_upper"],
        supports=["world_anchor"], angles_deg=[0.], nuisances=[(0., 0., 0.)],
        rotation_steps=5, retreat_steps=2, max_recovery=6, observe=False)
    args.update(overrides)
    return composition.run_group(**args)


def _geometry(state):
    return torch.cat((state[..., :2], state[..., 6:8], state[..., 12:14],
        state[..., 16:17].sin(), state[..., 16:17].cos(), state[..., 18:20],
        state[..., 22:23].sin(), state[..., 22:23].cos()), -1)


def test_common_prelude_and_real_history_are_preserved_without_replacement(monkeypatch):
    seen, setups = [], []
    original = composition.setup_scene
    def setup(*args, **kwargs):
        setups.append(1)
        return original(*args, **kwargs)
    def selector(*args, **kwargs):
        seen.append(args[6].clone())
        return zero_selector(*args, **kwargs)
    monkeypatch.setattr(composition, "setup_scene", setup)
    monkeypatch.setattr(composition, "select_options", selector)
    result = composition.run_group(substeps=5, planners=["feedback_upper"], supports=["world_anchor", "passive"],
        angles_deg=[0.], nuisances=[(0., 0., 0.)], rotation_steps=5, retreat_steps=2, max_recovery=6, observe=False)
    assert len(setups) == 1
    assert result["prelude_actions"].shape == (2, 10, 2, 2)
    assert result["prelude_actions"][:, :, 0].eq(0).all()
    assert seen[0][:, :, 16].eq(1).all()
    torch.testing.assert_close(seen[0][:, 0, :12], _geometry(result["prelude_trajectory"][:, 0]))
    torch.testing.assert_close(seen[0][:, 1, :12], _geometry(result["prelude_trajectory"][:, 5]))
    torch.testing.assert_close(seen[0][:, 0, 12:16], result["prelude_actions"][:, :5].mean(1).flatten(1))
    torch.testing.assert_close(seen[0][:, 1, 12:16], result["prelude_actions"][:, 5:10].mean(1).flatten(1))
    torch.testing.assert_close(result["initial_pose"][:, :2], result["prelude_trajectory"][:, -1, 12:14])
    torch.testing.assert_close(result["trajectory"][:, 0], result["prelude_trajectory"][:, -1], atol=0, rtol=0)
    assert all(r["total_steps"] == r["task_steps"]+10 and r["maximum_steps"] == 28 for r in result["rows"])


def test_final_success_does_not_repair_first_goal_failure(monkeypatch):
    original = composition.task_metrics
    def final_only(pose, initial, goal):
        _, error, translation = original(pose, initial, goal)
        return goal > .05, error, translation
    monkeypatch.setattr(composition, "task_metrics", final_only)
    result = tiny(monkeypatch)
    row = result["rows"][0]
    assert row["terminal_success"] and row["final_goal_dwell_success"]
    assert not row["first_goal_success_at_phase_boundary"]
    assert not row["ordered_goal_success"]
    assert "first_goal_dwell" in row["external_goal_failure_reasons"]


def test_external_goals_are_separate_from_recovery_contact_gate(monkeypatch):
    original = composition.task_metrics
    def always(pose, initial, goal):
        _, error, translation = original(pose, initial, goal)
        return torch.ones(len(pose), dtype=torch.bool), error, translation
    monkeypatch.setattr(composition, "task_metrics", always)
    result = tiny(monkeypatch, supports=["passive"])
    row = result["rows"][0]
    assert row["ordered_goal_success"]
    assert not row["contact_reacquired"]
    assert not row["interface_completion_success"]
    assert row["phase_steps"]["rotate_first"] == row["phase_steps"]["rotate_final"] == 5


def test_atomic_options_are_interrupted_by_real_stage_deadlines_not_credited_as_complete(monkeypatch):
    calls = []
    def long_option(env, interface, rows, initial_pose, goals, remaining, history, query_mask,
                    model_pack=None, allow_routes=None):
        calls.append((remaining.clone(), allow_routes.clone()))
        options = torch.tensor([0., 0., 0., 20.]).expand(env.num_envs, 4).clone()
        prediction = torch.zeros(env.num_envs, 5); prediction[:, 3:] = torch.tensor([2., 1.])
        return options, dict(selected_prediction=prediction, chosen_index=torch.ones(env.num_envs, dtype=torch.long))
    original = composition.task_metrics
    def never(pose, initial, goal):
        _, error, translation = original(pose, initial, goal)
        return torch.zeros(len(pose), dtype=torch.bool), error, translation
    monkeypatch.setattr(composition, "task_metrics", never)
    monkeypatch.setattr(composition, "select_options", long_option)
    result = composition.run_group(substeps=5, planners=["feedback_upper"], supports=["world_anchor"],
        angles_deg=[0.], nuisances=[(0., 0., 0.)], rotation_steps=10, retreat_steps=2,max_recovery=6,observe=False)
    assert len(calls) == 2
    assert not calls[0][1].any() and calls[1][1].all()
    assert len(result["option_records"]) == 2
    for option in result["option_records"]:
        assert option["elapsed_steps"] == 10
        assert option["option"][3] == 20
        assert option["interrupted"] and not option["completed"]
        assert option["termination_reason"] == "stage_deadline"
        assert "prediction_error" not in option
    assert result["rows"][0]["task_steps"] <= 28


def test_direct_model_never_gets_route_permission_and_models_remain_frozen(monkeypatch):
    model = dict(checkpoints={"world_anchor/full_matched/seed0": {"state_dict": {"w": torch.ones(1)}}})
    before = copy.deepcopy(model)
    seen = []
    def selector(*args, **kwargs):
        seen.append(kwargs["allow_routes"].clone())
        return zero_selector(*args, **kwargs)
    monkeypatch.setattr(composition, "select_options", selector)
    result = composition.run_group(substeps=5, model_pack=model, planners=["wm_full_direct"], supports=["world_anchor"],
        angles_deg=[0.], nuisances=[(0., 0., 0.)], rotation_steps=5, retreat_steps=2,max_recovery=6,observe=False)
    assert seen and not any(x.any() for x in seen)
    assert composition.checkpoint_digest(model) == composition.checkpoint_digest(before)
    assert result["rows"][0]["expanded_final_interface"] is False


def test_missing_model_fails_before_any_scene_initialization(monkeypatch):
    monkeypatch.setattr(composition, "setup_scene", lambda *args, **kwargs: pytest.fail("No simulation expected"))
    with pytest.raises(ValueError, match="Missing frozen local option model"):
        composition.run_group(planners=["wm_full"], supports=["world_anchor"])


def test_complete_snapshot_replay_and_actual_work_including_prelude(monkeypatch):
    result = tiny(monkeypatch, supports=["passive", "world_anchor"], observe=True)
    initial = result["counterfactual_snapshots"][0]
    env = make_profile("dry_light", num_envs=2, substeps=5, seed=101)
    restore(env, snapshot_from_record(initial))
    assert torch.equal(physical_state(env), result["trajectory"][:, 0])
    for t, command in enumerate(result["actions"].unbind(1)):
        env.step([command[:, 0], command[:, 1]])
        assert torch.equal(physical_state(env), result["trajectory"][:, t+1])
    assert len(result["prelude_measurements"]) == 10
    assert len(result["measurements"]) == result["actions"].shape[1]
    for i, row in enumerate(result["rows"]):
        assert row["actuator_absolute_work_by_agent"][0] >= row["prelude_actuator_absolute_work_by_agent"][0]
        assert row["actuator_absolute_work_by_agent"][1] >= row["prelude_actuator_absolute_work_by_agent"][1]
        assert max(row["maximum_action_norm_by_agent"]) <= 1.000001
    assert result["resource_accounting"]["exact_substep_work"]


def test_padding_after_completion_has_zero_commands_and_no_extra_cost(monkeypatch):
    original = composition.task_metrics
    def one_case_finishes(pose, initial, goal):
        _, error, translation = original(pose, initial, goal)
        return torch.arange(len(pose)) == 0, error, translation
    monkeypatch.setattr(composition, "task_metrics", one_case_finishes)
    result = tiny(monkeypatch, supports=["passive", "world_anchor"], rotation_steps=10)
    for i, row in enumerate(result["rows"]):
        assert int(result["active"][i].sum()) == row["task_steps"]
        assert result["actions"][i, ~result["active"][i]].eq(0).all()
    assert result["rows"][0]["task_steps"] < result["rows"][1]["task_steps"]
    assert result["rows"][0]["ordered_goal_success"]
    assert not result["rows"][1]["ordered_goal_success"]


def test_unfinished_navigation_consumes_the_rotation_budget_and_is_interrupted(monkeypatch):
    original = composition.task_metrics
    def never(pose, initial, goal):
        _, error, translation = original(pose, initial, goal)
        return torch.zeros(len(pose), dtype=torch.bool), error, translation
    def routed(env, interface, rows, initial_pose, goals, remaining, history, query_mask,
               model_pack=None, allow_routes=None):
        options = torch.tensor([0., 0., 0., 5.]).expand(env.num_envs, 4).clone()
        options[allow_routes, 0] = 1
        options[allow_routes, 2] = .3
        return options, {}
    monkeypatch.setattr(composition, "task_metrics", never)
    monkeypatch.setattr(composition, "select_options", routed)
    result = composition.run_group(substeps=5, planners=["feedback_upper"], supports=["world_anchor"],
        angles_deg=[0.], nuisances=[(0., 0., 0.)], rotation_steps=5, retreat_steps=2,max_recovery=6,observe=False)
    routed_options = [r for r in result["option_records"] if r["option"][0] == 1]
    assert len(routed_options) == 1
    record = routed_options[0]
    assert record["elapsed_steps"] == 5
    assert record["interrupted"] and record["termination_reason"] == "stage_deadline"
    assert not record["completed"]
    assert result["rows"][0]["phase_steps"]["rotate_final"] == 5
    assert result["rows"][0]["total_steps"] == result["rows"][0]["task_steps"]+10
    assert result["rows"][0]["route_options_started"] == 1
    assert result["rows"][0]["route_options_completed"] == 0
