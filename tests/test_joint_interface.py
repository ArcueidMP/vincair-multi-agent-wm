import pytest
import torch

from vmas_pusht.action_interface import ActionInterface, candidate_table
from vmas_pusht.joint_interface import HELPERS, INPUT_DIM, JointActionInterface, features, joint_table
from vmas_pusht.joint_models import execute, validate
from vmas_pusht.joint_planner import select_options, tables_for
from vmas_pusht.physical_abstraction import physical_state, setup_scene
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.snapshot import capture, restore


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def scene(n=7, substeps=160):
    env = make_profile("dry_light", num_envs=n, substeps=substeps, seed=173)
    anchor = setup_scene(env)
    return env, JointActionInterface(env, "world_anchor", anchor)


def test_joint_budget_preserves_all_primary_helper_pairs_across_route_allocations():
    for planned in (False, True):
        direct = joint_table(planned_helper=planned)
        assert len(direct) == 39*(7 if planned else 1)
        assert len(torch.unique(direct, dim=0)) == len(direct)
        expanded = [joint_table(planned_helper=planned, expanded=True, allocation=s) for s in range(3)]
        for table in expanded:
            assert torch.equal(table[:, 1:], direct[:, 1:])
        nonzero = direct[:, 2] > 0
        assert torch.equal(torch.stack([t[nonzero, 0] for t in expanded], 1).sort(1).values,
                           torch.tensor([0., 1., 2.]).expand(int(nonzero.sum()), 3))


def test_helper_zero_exactly_reproduces_fixed_support_commands_and_physics():
    env, joint = scene(3)
    other = make_profile("dry_light", num_envs=3, substeps=160, seed=173)
    restore(other, capture(env))
    fixed = ActionInterface(other, "world_anchor", joint.anchor)
    commands = torch.tensor([[0., -20., .15, 5.], [0., 0., .3, 10.], [0., 20., .6, 20.]])
    fixed.start(commands)
    joint.start(torch.cat((commands, torch.zeros(3, 1)), 1))
    for _ in range(20):
        a, b = fixed.command(), joint.command()
        assert torch.equal(a, b)
        other.step([a[:, 0], a[:, 1]]); env.step([b[:, 0], b[:, 1]])
        fixed.observe(); joint.observe()
        assert torch.equal(physical_state(other), physical_state(env))


def test_helper_choices_actuate_physically_without_changing_initial_state_or_goal():
    env, interface = scene()
    before = physical_state(env).clone()
    option = torch.tensor([[0., 0., .3, 5., float(i)] for i in range(7)])
    x = features(env, interface, option)
    interface.start(option)
    assert torch.equal(before, physical_state(env))
    action = interface.command()
    assert action[:, 1].unique(dim=0).shape[0] >= 4
    assert action.norm(dim=-1).max() <= 1.000001
    assert action[1, 1].eq(0).all()
    env.scenario.goal_pose[:] = 10.
    assert torch.equal(x, features(env, interface, option))
    env.step([action[:, 0], action[:, 1]]); interface.observe()
    assert not torch.equal(before, physical_state(env))
    assert interface.elapsed.eq(1).all()
    assert x.shape == (7, INPUT_DIM)


def test_cloned_joint_memory_replays_inflight_actions_and_has_no_aliases():
    env, interface = scene()
    option = torch.tensor([[0., 0., .3, 10., float(i)] for i in range(7)])
    interface.start(option)
    action = interface.command(); env.step([action[:, 0], action[:, 1]]); interface.observe()
    other = make_profile("dry_light", num_envs=7, substeps=160, seed=173)
    restore(other, capture(env))
    copied = interface.clone_to(other)
    for _ in range(9):
        a, b = interface.command(), copied.command()
        assert torch.equal(a, b)
        env.step([a[:, 0], a[:, 1]]); other.step([b[:, 0], b[:, 1]])
        interface.observe(); copied.observe()
        assert torch.equal(physical_state(env), physical_state(other))
    copied.helper_target.add_(1.)
    assert not torch.equal(copied.helper_target, interface.helper_target)


def test_joint_tables_fall_back_to_current_face_without_changing_the_budget():
    env, interface = scene(1)
    for mode in (0, 1, 3):
        interface.reset_contact(torch.tensor([True]), mode=mode)
        table = tables_for(env, interface, torch.tensor([True]), planned_helper=True, expanded=True, allocation=0)
        assert table.shape == (1, 273, 5)
        if mode != 3:
            assert table[..., 0].eq(0).all()


def test_invalid_helper_cannot_mutate_memory_or_physics():
    env, interface = scene(1)
    before = physical_state(env).clone()
    for invalid in (-1., 7., .5, float("nan")):
        with pytest.raises(ValueError):
            interface.start(torch.tensor([[0., 0., .3, 5., invalid]]))
        assert not interface.busy.any()
        assert torch.equal(before, physical_state(env))


def test_learned_joint_selection_uses_no_simulator_and_returns_both_choices(monkeypatch):
    env, interface = scene(1)
    class Predictor:
        entry = dict(duration_margin=0.)
        def __call__(self, x):
            out = torch.zeros(len(x), 5)
            out[:, 2] = x[:, 38+5]*.04
            out[:, 3] = .5; out[:, 4] = 1.
            return out
    def forbidden(*args, **kwargs):
        raise AssertionError("Learned selection queried simulator")
    monkeypatch.setattr(env, "step", forbidden)
    monkeypatch.setattr("vmas_pusht.joint_planner.oracle", forbidden)
    options, stats = select_options(env, interface, [{}], env.scenario.t_pose().clone(),
        torch.tensor([.04]), torch.tensor([60]), None, torch.tensor([True]), predictor=Predictor())
    assert options[0, 4] == 5
    assert stats["evaluated_candidates"].tolist() == [273]
    assert stats["active_simulator_steps"].sum() == 0


def test_simulator_queries_leave_live_state_and_helper_memory_unchanged():
    env, interface = scene(1)
    before = physical_state(env).clone()
    option, diagnostics = select_options(env, interface, [{}], env.scenario.t_pose().clone(),
        torch.tensor([.04]), torch.tensor([5]), None, torch.tensor([True]), planned_helper=False)
    assert torch.equal(before, physical_state(env))
    assert interface.elapsed.eq(0).all() and not interface.busy.any()
    assert diagnostics["evaluated_candidates"].tolist() == [13]
    assert option.shape == (1, 5)
