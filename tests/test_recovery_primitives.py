import pytest
import torch

from vmas_pusht.boundary import _at_wall, surface_gaps
from vmas_pusht.physical_abstraction import helper_action, physical_state, setup_scene
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.recovery_primitives import (advance_primitive, orient_push, primary_action,
    primitive_action, route_control, start_primitive)
from vmas_pusht.scenario import wrap


def test_mixed_route_memory_branches_independently_and_orients_the_same_bank():
    env = make_profile("dry_light", num_envs=3, substeps=5)
    setup_scene(env)
    state = start_primitive(env, ["upper_right", "lower_left", "stay"])
    before = physical_state(env).clone()
    nav, ready = route_control(env, state)
    assert torch.equal(physical_state(env), before)
    assert ready.tolist() == [False, False, True]
    assert nav.norm(dim=-1).max() <= 1.000001
    assert nav[2].eq(0).all()
    bank = torch.tensor([[[0., -.3], [.1, -.6]]]).expand(3, 2, 2).clone()
    oriented = orient_push(env, state, bank)
    assert torch.equal(oriented[0], bank[0])
    assert torch.equal(oriented[1], -bank[1])
    assert torch.equal(oriented[2], bank[2])
    clone = state.clone([1, 1, 0])
    assert clone.mode.tolist() == [1, 1, 0]
    clone.phase[0] = 3
    clone.initial_local[0, 0] += .1
    assert state.phase.tolist() == [0, 0, 3]
    assert not torch.equal(clone.initial_local[0], clone.initial_local[1])


def test_admissible_start_failure_and_brake_never_teleport_or_silently_acquire():
    env = make_profile("dry_light", num_envs=2, substeps=5)
    setup_scene(env, primary_local=(.30, -.20))
    env.world.agents[0].set_vel(torch.tensor([[.1, -.2], [.1, -.2]]), batch_index=None)
    state = start_primitive(env, ["upper_right", "brake"])
    before = physical_state(env).clone()
    action, ready = route_control(env, state)
    assert state.failed.tolist() == [True, False]
    assert not ready.any()
    assert (action * env.world.agents[0].state.vel).sum(-1).lt(0).all()
    assert action.norm(dim=-1).max() <= 1.000001
    advance_primitive(env, state)
    assert torch.equal(physical_state(env), before)
    assert state.first_push_step.tolist() == [-1, -1]
    with pytest.raises(ValueError):
        start_primitive(env, "upper_right", strength=1.01)
    with pytest.raises(ValueError):
        primitive_action(env, state, torch.ones(2, 2))


def test_finite_force_routes_reach_reverse_contacts_and_produce_negative_rotation():
    # Constructed local development state, not a selected successful task slice.
    # 60 real steps include every retreat, transit and acquisition step.
    torch.set_num_threads(1)
    env = make_profile("dry_light", num_envs=3, substeps=160)
    anchor = setup_scene(env)
    state = start_primitive(env, ["upper_right", "lower_left", "upper_right"],
                            strength=torch.tensor([.9, .3, 0.]))
    initial = env.scenario.t_pose().clone()
    minimum_gap = torch.ones(3)
    norm_impulse = torch.zeros(3, 2)
    work = torch.zeros(3, 2)
    walls = torch.zeros(3, dtype=torch.bool)
    for _ in range(60):
        helper = helper_action(env, "preloaded_anchor", anchor)
        command = primitive_action(env, state, helper)
        assert command.norm(dim=-1).max() <= 1.000001
        old_velocity = torch.stack([a.state.vel.clone() for a in env.world.agents], 1)
        env.step([command[:, 0], command[:, 1]])
        advance_primitive(env, state)
        new_velocity = torch.stack([a.state.vel for a in env.world.agents], 1)
        norm_impulse += command.norm(dim=-1) * env.world.dt
        work += (command * (old_velocity+new_velocity)/2).sum(-1).abs() * env.world.dt
        minimum_gap = torch.minimum(minimum_gap, surface_gaps(env)[:, 0].amin(-1))
        walls |= _at_wall(env)
    rotation = wrap(env.scenario.bar.state.rot[:, 0]-initial[:, 2])
    assert (state.first_push_step > 0).all()
    assert (state.first_push_step <= 40).all()
    assert not state.failed.any()
    assert rotation[0] < -.04
    assert rotation[1] < -.01
    assert rotation[0] < rotation[2]-.04  # Matched route, no-push control.
    assert (minimum_gap > -.006).all()   # No through-body shortcut.
    assert not walls.any()
    assert torch.isfinite(work).all() and (work[:, 0] > 0).all()
    assert (norm_impulse <= 6.00001).all()
    assert torch.isfinite(physical_state(env)).all()


def test_ready_route_allows_zero_or_direction_shifted_push_without_goal_input():
    env = make_profile("dry_light", num_envs=2, substeps=5)
    setup_scene(env)
    state = start_primitive(env, ["upper_right", "lower_left"])
    state.phase[:] = 3  # Test only the already-acquired force interface.
    navigation, ready = route_control(env, state)
    assert ready.all() and navigation.eq(0).all()
    assert primary_action(env, state, strength=0.).eq(0).all()
    shifted = primary_action(env, state, strength=.6, direction_deg=20.)
    torch.testing.assert_close(shifted[0], -shifted[1])
    torch.testing.assert_close(shifted.norm(dim=-1), torch.full((2,), .6))
