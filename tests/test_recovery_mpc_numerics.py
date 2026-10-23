import copy

import pytest
import torch

from vmas_pusht.recovery_mpc_numerics import replay, score_original_stops, validate_source


def test_scoring_uses_each_original_stop_and_requires_terminal_five_step_dwell():
    # Row0 had an early five-step dwell but drifted before its specified stop.
    # Row1 stopped after five successful steps; later padding must not erase it.
    source = dict(active=torch.tensor([[True]*7, [True]*5+[False]*2]),
        rows=[dict(initial_pose=[0., 0., 0.], relative_goal_rad=.1,
                   first_goal_pass=True, original_prefix_finite=True, original_prefix_wall=False) for _ in range(2)])
    trajectory = torch.zeros(2, 8, 24)
    trajectory[:, 1:6, 16] = .1
    trajectory[:, 6:, 16] = .2
    run = dict(trajectory=trajectory, finite=torch.ones(2, 8, dtype=torch.bool),
               touched_wall=torch.zeros(2, 8, dtype=torch.bool))
    score = score_original_stops(source, run)
    assert score["stop_steps"].tolist() == [7, 5]
    assert score["ever_five_step_dwell"].tolist() == [True, True]
    assert score["terminal_five_step_dwell"].tolist() == [False, True]
    assert score["continuation_goal_success"].tolist() == [False, True]
    assert score["ordered_goal_success"].tolist() == [False, True]
    assert score["terminal_pose"][:, 2].tolist() == pytest.approx([.2, .1])
    run["finite"][1, 7] = False  # Inactive padding is not part of row1's task.
    run["touched_wall"][1, 7] = True
    assert score_original_stops(source, run)["continuation_goal_success"].tolist() == [False, True]


def test_original_prefix_and_wall_gates_are_preserved_without_reconstructing_history():
    source = dict(active=torch.ones(2, 5, dtype=torch.bool),
        rows=[dict(initial_pose=[0., 0., 0.], relative_goal_rad=.1,
            first_goal_pass=i == 0, original_prefix_finite=True, original_prefix_wall=False,
            original_wall=True) for i in range(2)])
    trajectory = torch.zeros(2, 6, 24)
    trajectory[:, 1:, 16] = .1
    run = dict(trajectory=trajectory, finite=torch.ones(2, 6, dtype=torch.bool),
               touched_wall=torch.zeros(2, 6, dtype=torch.bool))
    score = score_original_stops(source, run)
    assert score["continuation_goal_success"].tolist() == [True, True]
    # Both original trajectories may hit a wall in their old FUTURE. That must
    # not erase a replacement continuation after a wall-free common prefix.
    assert score["ordered_goal_success"].tolist() == [True, False]
    run["touched_wall"][0, 3] = True
    assert score_original_stops(source, run)["continuation_goal_success"].tolist() == [False, True]
    source["rows"][1]["first_goal_pass"] = True
    source["rows"][1]["original_prefix_finite"] = False
    assert not score_original_stops(source, run)["ordered_goal_success"].any()


def test_replay_preserves_full_batch_true_start_and_original_zero_padding():
    from vmas_pusht.physics_profiles import make_profile
    from vmas_pusht.physical_abstraction import physical_state, setup_scene
    from vmas_pusht.stable_composition import snapshot_record
    env = make_profile("dry_light", num_envs=2, substeps=5)
    anchor = setup_scene(env, angle=[0., .3], hidden_v=(.03, 0.), hidden_omega=.2)
    rows = [dict(source_group=0, source_row=i, continuation_steps=2+i) for i in range(2)]
    start = snapshot_record(env, [0, 1], anchor, torch.zeros(2, 2, 17), rows, "post_recovery")
    commands = torch.tensor([[[[0., -.3], [-.1, .1]], [[0., -.3], [-.1, .1]], [[0., 0.], [0., 0.]]],
                             [[[0., -.6], [-.1, .1]], [[0., -.6], [-.1, .1]], [[0., -.6], [-.1, .1]]]])
    trajectory = [physical_state(env).clone()]
    for command in commands.unbind(1):
        env.step([command[:, 0], command[:, 1]])
        trajectory.append(physical_state(env).clone())
    source = dict(rows=rows, start=start, actions=commands, trajectory=torch.stack(trajectory, 1),
        active=torch.tensor([[True, True, False], [True, True, True]]))
    reproduced = replay(source, 5)
    assert torch.equal(reproduced["trajectory"], source["trajectory"])
    fine = replay(source, 10)
    assert torch.equal(fine["trajectory"][:, 0], source["trajectory"][:, 0])
    invalid = copy.deepcopy(source)
    invalid["rows"][0]["continuation_steps"] = 3
    with pytest.raises(ValueError, match="stop times"):
        validate_source(invalid)
    invalid = copy.deepcopy(source)
    invalid["start"]["rows"].reverse()
    with pytest.raises(ValueError, match="row order"):
        validate_source(invalid)
    invalid = copy.deepcopy(source)
    invalid["actions"][0, 2, 0, 0] = .1
    with pytest.raises(ValueError, match="padding"):
        validate_source(invalid)
