import copy

import pytest
import torch

from vmas_pusht.stable_task_numerics import error_summary, replay_group, validate_source


def _metadata():
    return dict(rows=[dict(support="preloaded_anchor", planner="new_coarse_small",
        initial_angle_deg=0., nuisance_id=0, goal_pair=[.04, .08])],
        phases=torch.tensor([[0, 1, 2, 4]]), active=torch.tensor([[True, True, True, False]]),
        phase_names=("rotate_first", "retreat", "recovery", "rotate_final", "done"))


def test_common_clock_summary_separates_post_done_padding_and_reports_first_phase():
    group = _metadata()
    left = torch.zeros(1, 5, 24)
    right = left.clone()
    right[0, 2, 16] = .004
    right[0, 4, 12] = 10.
    active = torch.tensor([[True, True, True, True, False]])
    summary = error_summary(left, right, active, group)
    assert summary["max_xy_units"] == 0.
    assert summary["max_angle_rad"] == pytest.approx(.004, abs=1e-6)
    assert summary["first_angle_breach"]["state_step"] == 2
    assert summary["first_angle_breach"]["phase_at_preceding_action"] == "retreat"
    assert summary["first_angle_breach"]["seconds_after_post_prelude"] == .2
    assert not summary["passes"]
    padding = error_summary(left, right, ~active, group)
    assert padding["max_xy_units"] == 10.
    assert not padding["max_xy_location"]["source_was_active"]


def test_error_summary_wraps_angles_without_aligning_events():
    group = _metadata()
    left = torch.zeros(1, 5, 24)
    right = left.clone()
    left[0, 1, 16] = torch.pi-.0005
    right[0, 1, 16] = -torch.pi+.0005
    summary = error_summary(left, right, torch.ones(1, 5, dtype=torch.bool), group)
    assert summary["max_angle_rad"] == pytest.approx(.001, abs=1e-6)
    assert summary["passes"]
    assert summary["first_angle_breach"] is None


def test_replay_copies_full_snapshot_and_matches_source_with_inactive_rows():
    from vmas_pusht.physical_abstraction import physical_state, setup_scene
    from vmas_pusht.physics_profiles import make_profile
    from vmas_pusht.stable_composition import snapshot_record
    env = make_profile("dry_light", num_envs=2, substeps=5)
    anchor = setup_scene(env, angle=[0., .3], hidden_v=(.03, 0.), hidden_omega=.2)
    rows = [dict(support="preloaded_anchor", planner="new_coarse_small",
        initial_angle_deg=0., nuisance_id=i, goal_pair=[.04, .08]) for i in range(2)]
    record = snapshot_record(env, [0, 1], anchor, torch.zeros(2, 2, 17), rows, "post_prelude")
    commands = torch.tensor([[[[0., -.3], [-.1, .1]], [[0., -.3], [-.1, .1]], [[0., 0.], [0., 0.]]],
                             [[[0., -.6], [-.1, .1]], [[0., -.6], [-.1, .1]], [[0., -.6], [-.1, .1]]]])
    trajectory = [physical_state(env).clone()]
    for action in commands.unbind(1):
        env.step([action[:, 0], action[:, 1]])
        trajectory.append(physical_state(env).clone())
    group = dict(rows=rows, trajectory=torch.stack(trajectory, 1), actions=commands,
        active=torch.tensor([[True, True, False], [True, True, True]]),
        phases=torch.tensor([[0, 3, 4], [0, 2, 3]]),
        counterfactual_snapshots=[record], physics_profile="dry_light")
    replay = replay_group(group, 5)
    assert torch.equal(replay["trajectory"], group["trajectory"])
    assert replay["full_batch_size"] == 2
    assert replay["finite"].all()
    refined = replay_group(group, 10)
    assert torch.equal(refined["trajectory"][:, 0], group["trajectory"][:, 0])
    bad = copy.deepcopy(group)
    bad["actions"][0, 2, 0, 0] = .1
    with pytest.raises(ValueError, match="padding"):
        validate_source(bad)
    bad = copy.deepcopy(group)
    bad["counterfactual_snapshots"][0]["rows"].reverse()
    with pytest.raises(ValueError, match="order"):
        validate_source(bad)
