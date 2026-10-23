"""Meaningful mechanics/data-integrity checks for the bounded-contact pilot."""

import math

import pytest
import torch

from vmas_pusht.boundary import _rotate, surface_gaps
from vmas_pusht.physical_abstraction import (
    MODES, PRELOAD_MODES, action_grid, audit_group, coarse_state, helper_action, matched_progress,
    model_features, norm_bound, nuisance_grid, physical_state, setup_scene,
)
from vmas_pusht.scenario import make


@pytest.fixture(autouse=True)
def single_cpu_thread():
    torch.set_num_threads(1)


def test_legal_aliases_have_same_geometry_but_consistent_distinct_velocities():
    env = make(num_envs=3)
    setup_scene(env, angle=.4, hidden_v=[[0, 0], [.03, 0], [0, -.03]],
                hidden_omega=torch.tensor([0., .2, -.2]))
    z = coarse_state(env)
    assert torch.equal(z, z[:1].expand_as(z))
    assert surface_gaps(env).min() > .00099
    assert not torch.equal(physical_state(env)[0], physical_state(env)[1])
    bar, stem = env.scenario.bar, env.scenario.stem
    d = stem.state.pos - bar.state.pos
    cross = bar.state.ang_vel * torch.stack((-d[:, 1], d[:, 0]), 1)
    assert torch.allclose(stem.state.vel - bar.state.vel, cross)


@pytest.mark.parametrize("mode", MODES)
def test_helper_commands_are_bounded_and_do_not_read_task_goals(mode):
    env = make(num_envs=3)
    anchor = setup_scene(env, hidden_v=[[0, 0], [1, 2], [-10, -20]], hidden_omega=[0., 4., -3.])
    original = helper_action(env, mode, anchor)
    env.scenario.goal_pose[:] = torch.tensor([.5, -.4, 2.])
    changed_goal = helper_action(env, mode, anchor)
    assert torch.equal(original, changed_goal)
    assert (original.norm(dim=-1) <= 1.000001).all()
    assert original.isfinite().all()
    if mode == "passive":
        assert torch.count_nonzero(original) == 0


def test_norm_constraint_does_not_allow_diagonal_force_advantage():
    bounded = norm_bound(torch.tensor([[10., 10.], [0., 0.], [-.3, .4]]))
    assert torch.allclose(bounded.norm(dim=-1), torch.tensor([1., 0., .5]))


def test_explicit_helper_memory_survives_current_position_change():
    env = make(num_envs=1)
    anchor = setup_scene(env)
    env.world.agents[1].set_pos(anchor + .01, batch_index=None)
    x, full = model_features(env, torch.tensor([[0., -.6]]), 10, anchor)
    assert x.shape == (1, 18) and full.shape == (1, 32)
    assert torch.equal(x[:, -2:], anchor)
    assert not torch.equal(x[:, 2:4], anchor)
    assert torch.equal(full[:, :18], x)


def test_grid_has_real_intervention_geometry_and_nuisance_holdouts():
    actions = action_grid()
    train = [r for r in actions if r["action_split"] == "train" and r["strength"] > 0]
    test = [r for r in actions if r["action_split"] == "test_action" and r["strength"] > 0]
    assert {r["direction_deg"] for r in train}.isdisjoint({r["direction_deg"] for r in test})
    assert {r["duration"] for r in train}.isdisjoint({r["duration"] for r in test})
    assert len({r["action_id"] for r in actions}) == len(actions)
    assert {r["nuisance_is_train"] for r in nuisance_grid()} == {True, False}


def test_rollout_matches_modes_and_exposes_alias_prediction_floor():
    actions = [dict(action_id=0, direction_deg=0, strength=.6, duration=5, action_split="train")]
    data = audit_group(friction_ratio=1.5, substeps=5, angles_deg=(0,), actions=actions,
                        nuisances=nuisance_grid()[:3], modes=MODES)
    assert data["exact_matched_starts"]
    assert data["X_coarse"].shape == (12, 18)
    assert data["X_full"].shape == (12, 32)
    assert data["Y"].shape == (12, 3)
    assert torch.equal(data["X_coarse"], data["X_coarse"][:1].expand(12, 18))
    assert data["Y"].isfinite().all()
    assert torch.equal(data["initial_physical_state"].reshape(3, 4, 24),
                       data["initial_physical_state"].reshape(3, 4, 24)[:, :1].expand(3, 4, 24))
    # Coarse aliases genuinely have different futures, so a deterministic
    # geometry-only predictor cannot fit every nuisance outcome exactly.
    assert (data["Y"][::4].var(dim=0) > 1e-9).any()
    for r in data["rows"]:
        assert r["finite_all_steps"] and not r["touched_arena_boundary"]
        assert max(r["maximum_action_norm_by_agent"]) <= 1.000001
        assert 0 <= r["helper_double_contact_fraction"] <= 1
        assert 0 <= r["helper_saturation_fraction"] <= 1
        assert r["max_joint_position_residual"] >= 0
    # Y is measured exactly at the requested duration from the saved trajectory.
    t = data["trajectories"]
    assert torch.allclose(data["Y"][:, :2], t[:, 5, 12:14] - t[:, 0, 12:14])


def test_rotation_equivariance_of_setup_and_track_contact_feedback():
    env = make(num_envs=2)
    anchor = setup_scene(env, angle=[0., math.pi / 2], hidden_v=[[.03, 0.], [0., .03]])
    force = helper_action(env, "track_contact", anchor)
    expected = torch.stack((-force[0, 1], force[0, 0]))
    assert torch.allclose(force[1], expected, atol=1e-6)


def test_preloaded_aliases_change_only_finite_controller_reference_points():
    env = make(num_envs=2)
    angles = torch.tensor([0., .4])
    anchor = setup_scene(env, angle=angles)
    before = physical_state(env).clone()
    shifted = anchor + _rotate(torch.tensor([-.01, .01]).expand(2, 2), angles)
    a = helper_action(env, "preloaded_anchor", anchor, reference_angle=angles)
    assert torch.allclose(a, helper_action(env, "world_anchor", shifted))
    b = helper_action(env, "preloaded_track", anchor)
    assert torch.allclose(b, helper_action(env, "track_contact", anchor, helper_local=(.091, -.091)))
    assert torch.equal(before, physical_state(env))
    assert torch.allclose(a.norm(dim=-1), torch.full((2,), math.sqrt(2)*.3), atol=1e-6)
    assert not set(PRELOAD_MODES) & set(MODES)


def test_finite_preload_sustains_contact_without_placement_overlap():
    actions = [dict(action_id=0, direction_deg=0, strength=.6, duration=20, action_split="train")]
    data = audit_group(friction_ratio=1.5, substeps=10, angles_deg=(0,), actions=actions,
                        nuisances=nuisance_grid()[:1], modes=("passive", "world_anchor"), preload=.01)
    passive, anchor = data["rows"]
    assert anchor["initial_minimum_gap"] > 0
    assert anchor["helper_double_contact_fraction"] > .9
    assert passive["helper_double_contact_fraction"] < anchor["helper_double_contact_fraction"]
    assert max(anchor["maximum_action_norm_by_agent"]) <= 1.000001
    progress = matched_progress(data, thresholds=(.005,))
    assert len(progress) == 2
    assert all(r["common_reached_rows"] <= r["reached_rows"] <= r["eligible_rows"] for r in progress)
