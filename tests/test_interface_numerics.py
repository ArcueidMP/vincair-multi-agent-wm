import copy

import pytest
import torch

from vmas_pusht import interface_numerics as audit
from vmas_pusht.physics_profiles import get_profile


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def synthetic():
    n, horizon = 2, 22
    base = torch.zeros(24)
    base[:2] = torch.tensor([-.5, .5]); base[6:8] = torch.tensor([.5, -.5])
    base[18:20] = torch.tensor([0., -.2])
    states = base.expand(n, horizon+1, 24).clone()
    states[0, 1:6, 16] = .04
    states[0, 13:18, 16] = .08
    states[1, 10:12, 16] = .04  # A late first goal must not repair its phase.
    states[1, 18:23, 16] = .08
    states[0, 18:, 0] = 1.3  # Post-stop wall contact and bad pose are padding.
    states[0, 18:, 16] = .4
    active = torch.ones(n, horizon, dtype=torch.bool); active[0, 17:] = False
    phases = torch.full((n, horizon), 4, dtype=torch.long)
    rows = []
    for i, end in enumerate((17, 22)):
        phases[i, :5] = 0; phases[i, 5:7] = 1; phases[i, 7:12] = 2; phases[i, 12:end] = 3
        rows.append(dict(support="world_anchor", planner="synthetic_test", substeps=5,
            first_goal_rad=.04, relative_goal_rad=.08, initial_angle_deg=0., nuisance_id=i,
            phase_end_steps=dict(rotate_first=5, retreat=7, recovery=12, rotate_final=end),
            task_steps=end, first_goal_success_at_phase_boundary=i == 0,
            final_goal_dwell_success=True, terminal_success=True,
            ordered_goal_success=i == 0, wall=True))
    group = dict(rows=rows, trajectory=states.clone(), active=active, phases=phases,
        physics_profile=get_profile("dry_light"), initial_pose=torch.zeros(n, 3),
        prelude_trajectory=base.expand(n, 11, 24).clone(), actions=torch.zeros(n, horizon, 2, 2))
    finite = torch.ones(n, horizon+1, dtype=torch.bool)
    walls = torch.zeros_like(finite); walls[0, 18:] = True
    run = dict(trajectory=states, finite=finite, touched_wall=walls, substeps=5,
        time=torch.arange(horizon+1)*.1, weld_xy=torch.zeros(n, horizon+1),
        weld_angle=torch.zeros(n, horizon+1))
    return group, run


def test_original_phase_endpoints_retain_failed_first_goal_and_ignore_padding():
    group, run = synthetic()
    score = audit.score_original_endpoints(group, run)
    assert score["first_goal_dwell_success"].tolist() == [True, False]
    assert score["final_goal_dwell_success"].tolist() == [True, True]
    assert score["ordered_goal_success"].tolist() == [True, False]
    assert not score["wall"].any()
    assert score["first_stop_steps"].tolist() == [5, 5]
    assert score["final_stop_steps"].tolist() == [17, 22]


def test_fine_terminal_pose_alone_cannot_replace_five_consecutive_states():
    group, coarse = synthetic()
    fine = copy.deepcopy(coarse); fine["substeps"] = 10
    fine["trajectory"][0, 13, 16] = .11
    fine["trajectory"][0, 19:, 16] = .08  # Later padding cannot supply missing dwell.
    result = audit.summarize(group, coarse, fine)
    assert result["source_replay_bitwise"] and result["source_scoring_reproduced"]
    assert result["rows"][0]["replay_fine"]["final_terminal_goal_success"]
    assert result["rows"][0]["replay_fine"]["final_last_consecutive_goal_steps"] == 4
    assert result["stop_time_outcomes"]["ordered_goal_success"]["lost"] == 1
    assert result["stop_time_outcomes"]["ordered_goal_success"]["gained"] == 0
    assert result["refinement"]["rotate_final"]["max_angle_rad"] > .029
    assert not result["is_fine_closed_loop_experiment"]
    assert result["planner_queries"] == result["world_model_queries"] == 0


def test_prelude_safety_is_computed_from_prefix_not_original_future_flags():
    group, run = synthetic()
    assert group["rows"][0]["wall"]
    assert audit.score_original_endpoints(group, run)["ordered_goal_success"][0]
    group["prelude_trajectory"][0, 2, 0] = 1.3
    score = audit.score_original_endpoints(group, run)
    assert score["prelude_wall"][0]
    assert not score["ordered_goal_success"][0]
    group["prelude_trajectory"][1, 0, 3] = torch.nan
    assert not audit.prelude_validity(group)["finite"][1]


def test_invalid_recorded_phase_clock_is_rejected():
    group, _ = synthetic()
    group["phases"][0, 1] = 3
    with pytest.raises(ValueError, match="Phase clock"):
        audit.validate_clock(group)


def test_real_whole_task_replay_preserves_all_states_and_scoring(monkeypatch):
    import vmas_pusht.interface_composition as composition
    def zero(env, interface, rows, initial, goals, remaining, history, query_mask,
             model_pack=None, allow_routes=None):
        return torch.tensor([0., 0., 0., 5.]).expand(env.num_envs, 4).clone(), {}
    monkeypatch.setattr(composition, "select_options", zero)
    group = composition.run_group(substeps=5, planners=["feedback_upper"], supports=["passive", "world_anchor"],
        goal_pair=(.8, 1.6), angles_deg=[0.], nuisances=[(0., 0., 0.)],
        rotation_steps=5, retreat_steps=2, max_recovery=6, observe=False)
    coarse = audit.replay_group(group, 5)
    fine = audit.replay_group(group, 10)
    result = audit.summarize(group, coarse, fine)
    assert result["source_replay_bitwise"]
    assert result["source_scoring_reproduced"]
    assert result["full_batch_and_order_preserved"]
    assert result["identical_complete_start"]
    assert set(audit.PHASES) <= set(result["refinement"])
    assert result["stop_time_outcomes"]["ordered_goal_success"]["source_success"] == 0


def test_aggregate_reports_offsetting_success_losses_and_gains_separately():
    group, coarse = synthetic()
    fine = copy.deepcopy(coarse); fine["substeps"] = 10
    fine["trajectory"][0, 13, 16] = .11
    fine["trajectory"][1, 1:6, 16] = .04
    result = audit.summarize(group, coarse, fine)
    result.update(source_path="synthetic.pt", group_index=0)
    summary, changes = audit.aggregate_results([result])
    assert len(summary) == 1
    assert summary[0]["source_ordered"] == summary[0]["fine_ordered"] == 1
    assert summary[0]["ordered_lost"] == summary[0]["ordered_gained"] == 1
    assert {(c["row"], c["metric"], c["change"]) for c in changes} == {
        (0, "final_goal_dwell_success", "lost"), (0, "ordered_goal_success", "lost"),
        (1, "first_goal_dwell_success", "gained"), (1, "ordered_goal_success", "gained")}
    assert all(c["source_path"] == "synthetic.pt" for c in changes)
