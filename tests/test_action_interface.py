from dataclasses import fields

import pytest
import torch

from vmas_pusht.action_interface import (ActionInterface, candidate_table,
    model_inputs, valid_options)
from vmas_pusht.physical_abstraction import model_features, physical_state, setup_scene
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.recovery_primitives import PrimitiveConfig, PrimitiveState
from vmas_pusht.snapshot import capture, restore


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def scene(n=2, substeps=5, supports="passive", **kwargs):
    env = make_profile("dry_light", num_envs=n, substeps=substeps)
    anchor = setup_scene(env, **kwargs)
    return env, ActionInterface(env, supports, anchor)


def tick(env, interface, mask=None):
    command = interface.command(active_mask=mask)
    env.step([command[:, 0], command[:, 1]])
    return command, interface.observe(active_mask=mask)


def test_candidate_library_preserves_direct_options_and_declared_durations():
    direct, expanded = candidate_table(False), candidate_table(True)
    assert direct.shape == (39, 4) and expanded.shape == (117, 4)
    torch.testing.assert_close(direct, expanded[:39], atol=0, rtol=0)
    assert torch.unique(expanded, dim=0).shape == expanded.shape
    assert sorted(set(expanded[:, 3].tolist())) == [5., 10., 20.]
    assert int((expanded[:, 2] == 0).sum()) == 9
    assert expanded[:, 2].max() == .9
    heldout = candidate_table(True, directions=(-35., 35.), strengths=(.45, .75), durations=(15, 25))
    assert heldout.shape == (30, 4)
    for kwargs in (dict(durations=(0,)), dict(strengths=(1.01,)), dict(directions=(0, 0))):
        with pytest.raises(ValueError):
            candidate_table(**kwargs)


def test_face_legality_and_failed_memory_only_allow_safe_zero_continuation():
    env, interface = scene(3)
    interface.reset_contact(torch.ones(3, dtype=torch.bool), mode=torch.tensor([3, 0, 1]))
    table = candidate_table()
    legal = interface.valid_options(table)
    assert legal.sum(1).tolist() == [117, 39, 39]
    interface.state.failed[1] = True
    interface.state.phase[1] = 2
    legal = valid_options(env, interface.mode, table, state=interface.state)
    assert legal.sum(1).tolist() == [117, 3, 39]
    assert table[legal[1], 0].eq(0).all() and table[legal[1], 2].eq(0).all()
    with pytest.raises(ValueError, match="second"):
        interface.start(torch.tensor([[0, 0, 0, 5], [0, 0, 0, 5], [1, 0, .3, 5]]))


def test_model_inputs_keep_original_layout_and_do_not_read_future_or_write_state():
    env, interface = scene(3, substeps=20)
    interface.reset_contact(torch.ones(3, dtype=torch.bool), mode=torch.tensor([3, 0, 1]))
    options = torch.tensor([[0., 0., .3, 5.], [0., 0., .3, 10.], [0., 0., .3, 20.]])
    before = physical_state(env).clone()
    coarse, full = model_inputs(env, interface.anchor, interface.mode, options)
    assert coarse.shape == (3, 24) and full.shape == (3, 38)
    torch.testing.assert_close(coarse[:, :18], full[:, :18], atol=0, rtol=0)
    torch.testing.assert_close(coarse[:, 18:], full[:, 32:], atol=0, rtol=0)
    assert coarse[:, 15].eq(160).all() and env.scenario.substeps == 20
    assert coarse[:, 14].tolist() == [.5, 1., 2.]
    torch.testing.assert_close(coarse[:, 12:14], torch.tensor([[0., -.3], [0., -.3], [0., .3]]))
    torch.testing.assert_close(coarse[:, 18:21], torch.eye(3))
    c, f = model_inputs(env, interface.anchor, interface.mode, options[:, None].expand(3, 2, 4))
    assert c.shape == (3, 2, 24) and f.shape == (3, 2, 38)
    expected_c, expected_f = model_features(env, coarse[:, 12:14], options[:, 3], interface.anchor)
    expected_c[:, 15] = expected_f[:, 15] = 160
    torch.testing.assert_close(coarse[:, :18], expected_c, atol=0, rtol=0)
    torch.testing.assert_close(full[:, :32], expected_f, atol=0, rtol=0)
    assert torch.equal(physical_state(env), before)


def test_interface_memory_operations_never_place_or_integrate_bodies():
    env, interface = scene()
    before, time = physical_state(env).clone(), env.steps.clone()
    interface.start(torch.tensor([[1, 0, .3, 5], [0, 0, .6, 10]]))
    interface.command()
    assert torch.equal(physical_state(env), before)
    with pytest.raises(RuntimeError, match="real environment step"):
        interface.observe()
    interface.cancel(torch.tensor([True, False]))
    interface.reset_contact(torch.tensor([True, False]), mode=3)
    assert torch.equal(physical_state(env), before)
    assert torch.equal(env.steps, time)


def test_mixed_direct_options_hold_world_force_and_count_only_real_push_steps():
    env, interface = scene(supports="world_anchor")
    interface.start(torch.tensor([[0, -20, .6, 5], [0, 20, .3, 10]]))
    first = interface.command().clone()
    for t in range(10):
        command, status = tick(env, interface)
        assert command.norm(dim=-1).max() <= 1.000001
        if t < 5:
            torch.testing.assert_close(command[:, 0], first[:, 0], atol=0, rtol=0)
        else:
            assert command[0].eq(0).all()
            torch.testing.assert_close(command[1, 0], first[1, 0], atol=0, rtol=0)
        if t == 4:
            assert status["newly_done"].tolist() == [True, False]
            assert interface.busy.tolist() == [False, True]
    assert interface.completed.all()
    assert interface.elapsed.tolist() == [5, 10]
    assert interface.pushed_steps.tolist() == [5, 10]
    assert interface.command().eq(0).all()


def test_bad_geometry_failure_consumes_a_real_step_and_zero_fallback_preserves_failure():
    env, interface = scene(1, primary_local=(.3, -.2))
    env.world.agents[0].set_vel(torch.tensor([[.1, -.2]]), batch_index=None)
    interface.start(torch.tensor([[1, 0, .3, 5]]))
    command, _ = tick(env, interface)
    assert interface.done.item() and interface.failed.item()
    assert interface.elapsed.item() == 1 and interface.pushed_steps.item() == 0
    assert not interface.contact_ready.item()
    assert command[0, 0].norm() <= 1.000001
    before_phase = interface.state.phase.clone()
    with pytest.raises(ValueError, match="before acquisition"):
        interface.start(torch.tensor([[0, 0, .3, 5]]))
    interface.start(torch.tensor([[0, 0, 0, 5]]))
    for _ in range(5):
        command, _ = tick(env, interface)
        assert command[:, 0].eq(0).all()
    assert interface.completed.item() and interface.elapsed.item() == 5
    assert interface.state.failed.item() and not interface.contact_ready.item()
    assert torch.equal(interface.state.phase, before_phase)
    assert int(interface.valid_options(candidate_table()).sum()) == 3


def test_acquisition_timeout_reports_failure_without_free_placement_or_extra_steps():
    env, base = scene(1)
    interface = ActionInterface(env, "passive", base.anchor,
        config=PrimitiveConfig(acquisition_limit=2, contact_confirm_steps=2))
    interface.start(torch.tensor([[1, 0, .3, 5]]))
    tick(env, interface)
    assert interface.busy.item()
    tick(env, interface)
    assert interface.failed.item() and interface.done.item()
    assert interface.elapsed.item() == 2 and interface.pushed_steps.item() == 0
    assert interface.state.phase.item() < 3


def test_clone_and_masked_cancellation_preserve_independent_memory():
    env, interface = scene()
    interface.start(torch.tensor([[1, 0, .3, 5], [2, 20, .6, 10]]))
    tick(env, interface)
    target = make_profile("dry_light", num_envs=3, substeps=5)
    restore(target, capture(env), indices=[1, 1, 0])
    clone = interface.clone_to(target, [1, 1, 0])
    assert clone.mode.tolist() == [1, 1, 0]
    clone.anchor[0, 0] += .1
    clone.state.initial_local[0, 0] += .1
    clone.cancel(torch.tensor([True, False, False]))
    assert clone.cancelled.tolist() == [True, False, False]
    assert interface.busy.all() and not interface.cancelled.any()
    assert not torch.equal(clone.anchor[0], clone.anchor[1])
    assert not torch.equal(clone.state.initial_local[0], clone.state.initial_local[1])
    assert torch.equal(clone.anchor[1], interface.anchor[1])
    for field in fields(PrimitiveState):
        value = getattr(clone.state, field.name)
        if torch.is_tensor(value):
            assert value.data_ptr() != getattr(interface.state, field.name).data_ptr()


def test_real_reverse_routes_charge_acquisition_and_then_exact_push_duration():
    env = make_profile("dry_light", num_envs=2, substeps=160)
    anchor = setup_scene(env) + torch.tensor([[-.01, .01]])
    interface = ActionInterface(env, "preloaded_anchor", anchor)
    interface.start(torch.tensor([[1, 0, .3, 5], [2, 0, .3, 5]]))
    for _ in range(55):
        if not interface.busy.any():
            break
        command, _ = tick(env, interface)
        assert command.norm(dim=-1).max() <= 1.000001
    assert interface.completed.all() and not interface.failed.any()
    assert interface.mode.tolist() == [0, 1]
    assert interface.contact_ready.all()
    assert (interface.state.first_push_step > 0).all()
    torch.testing.assert_close(interface.elapsed, interface.state.first_push_step+5, atol=0, rtol=0)
    assert interface.pushed_steps.tolist() == [5, 5]
    assert interface.held_force[0, 1] < 0 and interface.held_force[1, 1] > 0
