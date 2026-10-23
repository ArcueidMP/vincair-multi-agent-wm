"""Regression tests of actual contact/rotation replay, not only round-trip copies."""
import io

import pytest
import torch

from vmas_pusht.collect import contacts
from vmas_pusht.scenario import STEM_OFFSET, make, set_state, state
from vmas_pusht.snapshot import capture, restore


def moving_env():
    env = make(num_envs=3, seed=17, max_steps=30)
    compact = torch.zeros(3, 15)
    compact[:, :2] = torch.tensor([-.249, .025])
    compact[:, 4:6] = torch.tensor([.17, -.105])
    set_state(env, compact, torch.tensor([[.3, .25, .5]]))
    # Two contacts exert different torques; the bar and stem are genuinely
    # integrated independently before capturing a state, rather than fabricated.
    for _ in range(8):
        env.step([
            torch.tensor([[.9, .1], [1., -.15], [.8, .2]]),
            torch.tensor([[0., 1.], [-.1, .9], [.1, .8]]),
        ])
    assert contacts(env).all()
    assert env.scenario.bar.state.ang_vel.abs().min() > .01
    return env


def compare_snapshots(actual, expected, indices=None):
    if indices is None:
        indices = torch.arange(expected.batch_size)
    assert actual.physics == expected.physics
    for actual_entity, expected_entity in zip(actual.entities, expected.entities):
        assert actual_entity.name == expected_entity.name
        for field in ("state", "action"):
            a, b = getattr(actual_entity, field), getattr(expected_entity, field)
            if b is None:
                assert a is None
                continue
            assert a.keys() == b.keys()
            for key in a:
                if b[key] is None:
                    assert a[key] is None
                else:
                    torch.testing.assert_close(a[key], b[key][indices], rtol=0, atol=0)
    for key in actual.scenario:
        if expected.scenario[key] is None:
            assert actual.scenario[key] is None
        else:
            torch.testing.assert_close(actual.scenario[key], expected.scenario[key][indices], rtol=0, atol=0)
    torch.testing.assert_close(actual.steps, expected.steps[indices], rtol=0, atol=0)


def actions(horizon):
    return torch.rand(horizon, 3, 2, 2, generator=torch.Generator().manual_seed(181)) * 2 - 1


def compare_outputs(actual, expected, indices=None):
    if indices is None:
        indices = torch.arange(expected[2].shape[0])
    for category in (0, 1):  # observations and rewards
        for a, b in zip(actual[category], expected[category]):
            torch.testing.assert_close(a, b[indices], rtol=0, atol=0)
    assert torch.equal(actual[2], expected[2][indices])
    for a, b in zip(actual[3], expected[3]):
        for key in a:
            torch.testing.assert_close(a[key], b[key][indices], rtol=0, atol=0)


@pytest.mark.parametrize("horizon", [1, 5, 25])
def test_live_contact_rollout_equals_exactly_restored_rollout(horizon):
    source = moving_env()
    snapshot = capture(source)
    target = make(num_envs=3, seed=41, max_steps=30)
    restore(target, snapshot)
    for action in actions(horizon):
        expected = source.step([action[:, 0].clone(), action[:, 1].clone()])
        actual = target.step([action[:, 0].clone(), action[:, 1].clone()])
        compare_outputs(actual, expected)
        compare_snapshots(capture(target), capture(source))
    assert torch.equal(snapshot.steps, torch.full((3,), 8.))
    if horizon == 25:
        assert actual[2].all()  # restoring steps also restores the time limit.


def test_same_environment_can_replay_without_mutating_saved_snapshot():
    env = moving_env()
    snapshot = capture(env)
    stream = actions(25)
    for action in stream:
        env.step([action[:, 0], action[:, 1]])
    expected = capture(env)
    restore(env, snapshot)
    compare_snapshots(capture(env), snapshot)
    for action in stream:
        env.step([action[:, 0], action[:, 1]])
    compare_snapshots(capture(env), expected)
    restore(env, snapshot)
    env.world.agents[0].state.pos.add_(10)
    env.scenario.goal_pose.add_(10)
    env.steps.add_(10)
    restore(env, snapshot)
    compare_snapshots(capture(env), snapshot)


def test_repeated_and_reordered_source_rows_replay_exactly():
    source = moving_env()
    rows = torch.tensor([2, 0, 2, 1, 0, 1, 1])
    target = make(num_envs=len(rows), seed=61, max_steps=30)
    restore(target, capture(source), indices=rows)
    for action in actions(25):
        expected = source.step([action[:, 0].clone(), action[:, 1].clone()])
        actual = target.step([action[rows, 0], action[rows, 1]])
        compare_outputs(actual, expected, rows)
        compare_snapshots(capture(target), capture(source), rows)


def test_singleton_expansion_and_goal_reward_counter_buffers_are_independent():
    source = make(num_envs=1, max_steps=7, seed=8)
    source.step([torch.ones(1, 2), torch.zeros(1, 2)])
    snapshot = capture(source)
    target = make(num_envs=4, max_steps=7, seed=9)
    restore(target, snapshot)
    rows = torch.zeros(4, dtype=torch.long)
    compare_snapshots(capture(target), snapshot, rows)
    source.scenario.goal_pose.zero_()
    source.scenario.prev_err.zero_()
    source.scenario.rew.fill_(99)
    source.scenario.goal_bar.state.pos.fill_(99)
    source.steps.fill_(99)
    for agent in source.world.agents:
        agent.state.force.fill_(99)
        agent.state.torque.fill_(99)
        agent.action.u.fill_(99)
    compare_snapshots(capture(target), snapshot, rows)
    target.scenario.goal_pose[0].fill_(77)
    assert not torch.equal(target.scenario.goal_pose[0], target.scenario.goal_pose[1])
    restore(target, snapshot)
    compare_snapshots(capture(target), snapshot, rows)


def test_absent_reward_and_action_buffers_restore_after_a_later_step():
    env = make(num_envs=1)
    snapshot = capture(env)
    assert snapshot.scenario["rew"] is None
    env.step([torch.ones(1, 2), torch.zeros(1, 2)])
    restore(env, snapshot)
    assert not hasattr(env.scenario, "rew")
    assert env.world.agents[0].action.u is None
    compare_snapshots(capture(env), snapshot)
    env.step([torch.ones(1, 2), torch.zeros(1, 2)])


@pytest.mark.parametrize("kwargs", [
    {"substeps": 2}, {"friction_ratio": 1.2}, {"n_agents": 3},
    {"agent_force": .7}, {"max_steps": 31},
])
def test_incompatible_configuration_rejected_before_mutation(kwargs):
    source = moving_env()
    target = make(num_envs=3, **({"max_steps": 30} | kwargs))
    before = capture(target)
    with pytest.raises(ValueError, match="identical physics"):
        restore(target, capture(source))
    compare_snapshots(capture(target), before)


@pytest.mark.parametrize("indices", [None, [0], [0, 1, 2, 3], [0, 1, 1, -1], [0., 1., 1., 0.], [True] * 4])
def test_bad_source_indices_rejected(indices):
    source = moving_env()
    target = make(num_envs=4, max_steps=30)
    with pytest.raises(ValueError):
        restore(target, capture(source), indices=indices)


def test_noisy_dynamics_are_explicitly_unsupported():
    env = make(num_envs=1)
    env.world.agents[0].action._u_noise = .1
    with pytest.raises(ValueError, match="noiseless"):
        capture(env)


@pytest.mark.parametrize("n_agents", [1, 2, 3])
def test_compact_placement_uses_rigid_body_velocity_and_correct_agent_offset(n_agents):
    env = make(num_envs=2, n_agents=n_agents)
    offset = 4 * n_agents
    compact = torch.zeros(1, offset + 7)
    compact[:, offset:offset + 6] = torch.tensor([.1, .2, .7, .3, -.2, 1.4])
    set_state(env, compact)
    assert state(env).shape == (2, offset + 7)
    torch.testing.assert_close(state(env), compact.expand(2, -1), rtol=0, atol=1e-7)
    r = env.scenario.stem.state.pos - env.scenario.bar.state.pos
    expected_velocity = compact[:, offset + 3:offset + 5] + 1.4 * torch.stack([-r[:, 1], r[:, 0]], dim=1)
    torch.testing.assert_close(env.scenario.stem.state.vel, expected_velocity, rtol=0, atol=1e-7)
    torch.testing.assert_close(r.norm(dim=1), torch.full((2,), abs(STEM_OFFSET)), rtol=0, atol=1e-7)


def test_compact_state_cannot_reconstruct_ongoing_contact_physics():
    source = moving_env()
    approximate = make(num_envs=3, max_steps=30)
    exact = make(num_envs=3, max_steps=30)
    restore(exact, capture(source))
    # Even the corrected rigid velocity cannot reconstruct soft-joint residuals.
    set_state(approximate, state(source), source.scenario.goal_pose)
    assert (source.scenario.stem.state.pos - approximate.scenario.stem.state.pos).abs().max() > .001
    action = actions(1)[0]
    source.step([action[:, 0].clone(), action[:, 1].clone()])
    exact.step([action[:, 0].clone(), action[:, 1].clone()])
    approximate.step([action[:, 0].clone(), action[:, 1].clone()])
    compare_snapshots(capture(exact), capture(source))
    assert (state(source) - state(approximate)).abs().max() > .001


def test_snapshot_serialization_preserves_replay():
    source = moving_env()
    snapshot = capture(source)
    buffer = io.BytesIO()
    torch.save(snapshot, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    target = make(num_envs=3, max_steps=30)
    restore(target, loaded)
    compare_snapshots(capture(target), snapshot)
