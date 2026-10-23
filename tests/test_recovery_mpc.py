from dataclasses import fields

import pytest
import torch

from vmas_pusht import recovery_mpc as mpc
from vmas_pusht.physical_abstraction import setup_scene, physical_state, coarse_state
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.recovery_primitives import start_primitive
from vmas_pusht.snapshot import capture
from vmas_pusht.stable_composition import snapshot_record, empty_history


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def env_and_record():
    env = make_profile("dry_light", num_envs=2, substeps=2)
    anchor = setup_scene(env, primary_local=(-.5, .5), helper_local=(.5, -.5))
    rows = [dict(support="passive", planner="synthetic_test", initial_pose=[0., 0., 0.],
                 relative_goal_rad=0., first_goal_pass=first,
                 prefix_steps=30, original_prefix_finite=True, original_prefix_wall=False)
            for first in (True, False)]
    record = snapshot_record(env, [0, 1], anchor, empty_history(coarse_state(env)), rows, "post_recovery")
    return env, record


def test_declared_library_is_nested_and_keeps_bank13_choices():
    basic = mpc.template_table("bank13")
    expanded = mpc.template_table("expanded")
    assert basic.shape == (39, 3)
    assert expanded.shape == (117, 3)
    torch.testing.assert_close(basic, expanded[:39], atol=0, rtol=0)
    assert set(basic[:, 1].tolist()) == set(range(13))
    assert set(expanded[:, 0].tolist()) == {0, 1, 2}


def test_navigation_memory_and_side_survive_replanning():
    env = make_profile("dry_light", num_envs=2, substeps=2)
    setup_scene(env)
    state = start_primitive(env, ["lower_left", "upper_right"])
    state.phase[:] = torch.tensor([3, 1]); state.elapsed[:] = torch.tensor([31, 12])
    templates = torch.tensor([[0, 6, 2], [0, 0, 0]])
    plan = mpc.begin_plan(env, state, templates)
    assert plan["state"].mode.tolist() == [1, 0]
    assert plan["state"].phase.tolist() == [3, 1]
    assert plan["state"].elapsed.tolist() == [31, 12]
    plan["state"].phase[0] = 0
    assert state.phase[0] == 3
    second_route = mpc.begin_plan(env, state, torch.tensor([[1, 6, 2], [2, 6, 2]]))
    assert not second_route["valid"].any()


def test_macro_force_is_held_and_lower_contact_reverses_normal():
    env = make_profile("dry_light", num_envs=2, substeps=2)
    anchor = setup_scene(env)
    state = start_primitive(env, ["stay", "lower_left"])
    state.phase[:] = 3; state.failed[:] = False
    plan = mpc.begin_plan(env, state, torch.tensor([[0, 7, 1], [0, 7, 1]]))
    initial = env.scenario.t_pose().clone()
    goals = torch.tensor([.1, .1]); active = torch.ones(2, dtype=torch.bool)
    first, _ = mpc.plan_command(env, plan, initial, goals, anchor, torch.zeros(2, dtype=torch.bool), active)
    assert first[0, 0, 1] < 0 and first[1, 0, 1] > 0
    plan["pushed_steps"][:] = 1
    env.scenario.bar.state.rot.add_(.1)
    held, _ = mpc.plan_command(env, plan, initial, goals, anchor, torch.zeros(2, dtype=torch.bool), active)
    torch.testing.assert_close(first[:, 0], held[:, 0], atol=0, rtol=0)
    assert held[:, 1].eq(0).all()


def test_simulator_lookahead_leaves_live_full_snapshot_unchanged():
    env, record = env_and_record()
    before = capture(env)
    state = start_primitive(env, "stay")
    optimizer = mpc.ShootingMPC(2, "bank13", 2)
    templates, info = optimizer.choose(env, state, env.scenario.t_pose().clone(),
        torch.zeros(2), record["anchor"], torch.zeros(2, dtype=torch.bool), 10, torch.zeros(2, dtype=torch.long))
    after = capture(env)
    for old, new in zip(before.entities, after.entities):
        for key, value in old.state.items():
            if value is not None:
                torch.testing.assert_close(value, new.state[key], atol=0, rtol=0)
        if old.action is not None:
            for key, value in old.action.items():
                if value is not None:
                    torch.testing.assert_close(value, new.action[key], atol=0, rtol=0)
    torch.testing.assert_close(before.steps, after.steps, atol=0, rtol=0)
    assert info["horizon"] == 10
    assert info["candidate_environment_steps"] == 2*39*10
    assert info["predicted_success"].all()


def test_external_scoring_requires_original_first_gate_and_real_five_step_dwell():
    _, record = env_and_record()
    result = mpc.run_mpc(record, "bank13", substeps=2, horizon=10, observe=False)
    assert [r["continuation_goal_success"] for r in result["rows"]] == [True, True]
    assert [r["ordered_goal_success"] for r in result["rows"]] == [True, False]
    assert [r["continuation_steps"] for r in result["rows"]] == [5, 5]
    assert [r["total_steps"] for r in result["rows"]] == [35, 35]
    assert result["actions"].shape[1] == 5
    assert result["active"].all()


def test_incomplete_dwell_and_wall_do_not_count_as_success(monkeypatch):
    _, record = env_and_record()
    monkeypatch.setattr(mpc, "_at_wall", lambda env: torch.ones(env.num_envs, dtype=torch.bool))
    result = mpc.run_mpc(record, "bank13", substeps=2, horizon=5, observe=False)
    assert not any(r["continuation_goal_success"] for r in result["rows"])
    assert all(r["wall"] for r in result["rows"])


def test_record_selection_preserves_source_states_and_ids():
    env, record = env_and_record()
    selected = mpc.select_record(record, [1, 0, 1])
    snap = mpc.snapshot_from_record(selected)
    assert snap.batch_size == 3
    assert [r["first_goal_pass"] for r in selected["rows"]] == [False, True, False]
    primary = next(e for e in snap.entities if e.name == env.world.agents[0].name)
    torch.testing.assert_close(primary.state["pos"], env.world.agents[0].state.pos[[1, 0, 1]])


def test_partial_force_macro_remains_locked_after_contact_acquisition_and_replan():
    env, record = env_and_record()
    state = start_primitive(env, "stay")
    old = mpc.begin_plan(env, state, torch.tensor([[0, 7, 0], [0, 7, 0]]))
    old["pushed_steps"][:] = 2
    old["held"][:] = torch.tensor([0., -.6])
    new = mpc.begin_plan(env, state, torch.tensor([[0, 0, 0], [0, 0, 0]]), mpc.unfinished_macro(old))
    initial = env.scenario.t_pose().clone()
    for i in range(4):
        command, ready = mpc.plan_command(env, new, initial, torch.zeros(2), record["anchor"],
            torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
        assert command[:, 0, 1].eq(-.6 if i < 3 else 0.).all()
        env.step([command[:, 0], command[:, 1]])
        mpc.after_step(env, new, ready, torch.ones(2, dtype=torch.bool))


def test_recorded_wall_predicate_matches_live_geometry_and_does_not_read_future():
    env, _ = env_and_record()
    before = physical_state(env).clone()
    env.world.agents[0].state.pos[0, 0] = 1.18
    later = physical_state(env).clone()
    torch.testing.assert_close(mpc.trajectory_at_wall(later, env), mpc._at_wall(env))
    assert not mpc.trajectory_at_wall(before, env).any()
    assert mpc.trajectory_at_wall(torch.stack([before, later], 1), env)[0, 1]


def test_finished_row_does_not_inherit_an_unexecuted_later_route(monkeypatch):
    _, record = env_and_record()
    record["rows"][1]["relative_goal_rad"] = .5
    calls = []
    def choose(self, env, state, initial, goals, anchor, actuated, remaining, actual_dwell, carry=None):
        calls.append(remaining)
        route = 0 if remaining == 10 else 1
        return torch.tensor([[route, 0, 0], [route, 0, 0]]), dict(candidate_environment_steps=2*117*remaining)
    monkeypatch.setattr(mpc.ShootingMPC, "choose", choose)
    result = mpc.run_mpc(record, "expanded", substeps=2, horizon=10, observe=False)
    assert calls == [10, 5]
    assert result["rows"][0]["continuation_goal_success"]
    assert result["rows"][0]["final_route_mode"] == 3
    assert result["rows"][0]["navigation_steps"] == 0
    assert result["actions"][0, 5:].eq(0).all()
    assert result["rows"][1]["final_route_mode"] == 0
