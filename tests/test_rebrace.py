import copy

import pytest
import torch
from vmas.simulator.utils import LINE_MIN_DIST

import vmas_pusht.rebrace as rebrace
from vmas_pusht.boundary import _rotate, surface_gaps
from vmas_pusht.physical_abstraction import norm_bound, setup_scene
from vmas_pusht.scenario import make


def valid_metrics(n=1):
    return {"finite": torch.ones(n, dtype=torch.bool), "helper_gaps": torch.full((n, 2), .003),
            "helper_normal_forces": torch.full((n, 2), .3),
            "helper_region_error": torch.zeros(n), "primary_position_error": torch.zeros(n),
            "primary_relative_speed": torch.zeros(n), "helper_relative_speed": torch.zeros(n),
            "raw_support_force_norm": torch.full((n,), .3), "weld_position_residual": torch.zeros(n),
            "weld_angle_residual": torch.zeros(n)}


@pytest.mark.parametrize("key,value", [
    ("finite", False), ("helper_gaps", [.008, .001]), ("helper_gaps", [-.02, .001]),
    ("helper_normal_forces", [.3, 0.]), ("helper_region_error", .08),
    ("primary_position_error", .1), ("primary_relative_speed", .2),
    ("helper_relative_speed", .2), ("raw_support_force_norm", 1.),
    ("weld_position_residual", .02), ("weld_angle_residual", .1),
])
def test_readiness_rejects_wrong_contact_wrong_face_speed_saturation_and_deformation(key, value):
    metrics = valid_metrics()
    assert rebrace.readiness_from_metrics(metrics).item()
    metrics[key][0] = torch.as_tensor(value)
    assert not rebrace.readiness_from_metrics(metrics).item()


def test_force_diagnostic_equals_actual_engine_collision_routine_and_preserves_state(monkeypatch):
    env = make(num_envs=3, device="cpu", substeps=20)
    setup_scene(env, helper_error=[[0., 0.], [.005, -.005], [.008, -.008]])
    before = [entity.state.pos.clone() for entity in env.world.entities]
    actual = []
    def record(entity_a, force_a, torque_a, entity_b, force_b, torque_b):
        assert entity_b is env.world.agents[1]
        actual.append(force_b.clone())
    monkeypatch.setattr(env.world, "update_env_forces", record)
    env.world._box_sphere_vectorized_collision([
        (env.scenario.bar, env.world.agents[1]), (env.scenario.stem, env.world.agents[1])])
    reported = rebrace.helper_contact_forces(env)
    torch.testing.assert_close(reported, torch.stack(actual, 1), atol=1e-7, rtol=1e-6)
    # Positive geometric gaps inside LINE_MIN_DIST carry real normal force.
    assert (surface_gaps(env)[:2, 1] > 0).all()
    assert (reported[:2].norm(dim=-1) > .001).all()
    assert torch.equal(reported[2], torch.zeros(2, 2))
    assert abs(LINE_MIN_DIST - .006666666666666667) < 1e-12
    for entity, old in zip(env.world.entities, before):
        torch.testing.assert_close(entity.state.pos, old, rtol=0, atol=0)


def test_persistence_resets_on_break_and_timeout_never_means_ready(monkeypatch):
    env = make(num_envs=1, device="cpu", substeps=20)
    setup_scene(env)
    controller = rebrace.RebraceController(env, ["world_anchor"], max_steps=6, confirm_steps=3)
    controller.phase[:] = 2
    metrics = valid_metrics()
    def fake(*args, **kwargs):
        result = copy.deepcopy(metrics)
        result["instantaneous_ready"] = rebrace.readiness_from_metrics(result)
        return result
    monkeypatch.setattr(rebrace, "readiness_metrics", fake)
    assert not controller.update()["ready"].item()
    assert controller.update()["dwell"].item() == 2
    metrics["helper_normal_forces"][:] = 0
    assert controller.update()["dwell"].item() == 0
    metrics["helper_normal_forces"][:] = .3
    assert not controller.update()["ready"].item()
    metrics["helper_normal_forces"][:] = 0
    controller.update()
    status = controller.update()
    assert status["failed"].item() and not status["ready"].item()
    assert not status["accepted"].item()


def test_ready_is_current_accepted_is_separate_monotonic_event(monkeypatch):
    env = make(num_envs=1, device="cpu", substeps=20)
    setup_scene(env)
    controller = rebrace.RebraceController(env, ["world_anchor"], confirm_steps=2)
    controller.phase[:] = 2
    metrics = valid_metrics()
    def fake(*args, **kwargs):
        result = copy.deepcopy(metrics)
        result["instantaneous_ready"] = rebrace.readiness_from_metrics(result)
        return result
    monkeypatch.setattr(rebrace, "readiness_metrics", fake)
    controller.update(); status = controller.update()
    assert status["ready"].item() and status["accepted"].item()
    metrics["helper_normal_forces"][:] = 0
    status = controller.update()
    assert not status["ready"].item() and status["accepted"].item()


def test_physical_recovery_survives_twenty_real_handoff_steps_with_zero_and_nonzero_main():
    supports = ["world_anchor", "world_anchor", "preloaded_anchor", "preloaded_anchor"]
    env = make(num_envs=4, device="cpu", substeps=20, friction_ratio=1.5)
    setup_scene(env)
    target = env.world.agents[1].state.pos.clone() + torch.tensor([.15, -.15])
    for _ in range(20):
        helper = env.world.agents[1]
        force = norm_bound(30 * (target - helper.state.pos) - 4 * helper.state.vel)
        env.step([torch.zeros(4, 2), force])
    assert (surface_gaps(env)[:, 1].amin(-1) > LINE_MIN_DIST).all()
    controller = rebrace.RebraceController(env, supports)
    finished = torch.zeros(4, dtype=torch.bool)
    for _ in range(80):
        force = controller.step()
        assert force.norm(dim=-1).max() <= 1.000001
        env.step([force[:, 0], force[:, 1]])
        status = controller.update(active_mask=~finished)
        finished |= status["ready"]
        if finished.all():
            break
    assert finished.all()
    anchor = controller.anchor.clone()
    for _ in range(20):
        angle = env.scenario.bar.state.rot[:, 0]
        main = torch.stack((angle.sin(), -angle.cos()), -1) * torch.tensor([0., .3, 0., .3])[:, None]
        env.step([main, rebrace.support_action(env, anchor, supports)])
        assert (rebrace.helper_contact_forces(env).norm(dim=-1) >= .001).all()
    assert (env.steps > 40).all()
    torch.testing.assert_close(controller.anchor, anchor, rtol=0, atol=0)
    assert controller.anchor_ood.tolist() == [True, True, False, False]


def test_passive_handoff_is_zero_force_and_inactive_rows_do_not_advance():
    env = make(num_envs=2, device="cpu", substeps=20)
    setup_scene(env)
    controller = rebrace.RebraceController(env, ["passive", "world_anchor"])
    controller.anchor += torch.tensor([-.01, .01])
    force = rebrace.support_action(env, controller.anchor, controller.supports)
    assert torch.equal(force[0], torch.zeros(2))
    assert force[1].norm() > 0
    controller.update(active_mask=torch.tensor([False, True]))
    assert controller.elapsed.tolist() == [0, 1]
    controller.reset(torch.tensor([False, True]))
    assert controller.elapsed.tolist() == [0, 0]


def test_invalid_support_and_time_budget_are_rejected():
    env = make(num_envs=1, device="cpu")
    with pytest.raises(ValueError, match="supports"):
        rebrace.RebraceController(env, ["magic_fixture"])
    with pytest.raises(ValueError, match="positive"):
        rebrace.RebraceController(env, ["passive"], max_steps=0)
