import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.geometry import PAYLOAD_RADIUS, UAV_RADIUS, team_clearances
from uav_arena.scenarios import (CONDITIONS, Condition, Mover, Pillar, Scenario, _free_path, make_scenario,
                                 suite, team_clearance)


def smooth_transfer(obs, duration=4.0):
    """Open-loop reference: shift the hover formation to the goal along a minimum-jerk straight line."""
    start, shift = obs["recent_setpoints"][-1], obs["goal"]-obs["payload_pos"]

    def policy(obs):
        s = min(1.0, (obs["decision_time"]+0.1)/duration)
        return start+shift*(10*s**3-15*s**4+6*s**5)
    return policy


def open_field(latency_ms=0, pillars=()):
    return Scenario(seed=0, condition=Condition("static", latency_ms), start=(-1.0, 0., 1.2),
                    goal=(1.0, 0., 1.2), pillars=tuple(pillars))


def test_scenarios_are_deterministic_and_paired_across_conditions():
    assert make_scenario(3, CONDITIONS[5]) == make_scenario(3, CONDITIONS[5])
    scenarios = [make_scenario(3, c) for c in CONDITIONS]
    assert len({(s.start, s.goal, s.pillars) for s in scenarios}) == 1
    assert all(not s.movers for s in scenarios if s.condition.obstacles == "static")
    moving = [s.movers for s in scenarios if s.condition.obstacles == "static+moving"]
    assert moving[0] and all(m == moving[0] for m in moving)
    assert make_scenario(3, CONDITIONS[0]).pillars != make_scenario(4, CONDITIONS[0]).pillars


def test_every_development_layout_leaves_a_team_corridor():
    for scenario in suite("dev", conditions=CONDITIONS[:1]):
        assert _free_path(scenario.pillars, scenario.start, scenario.goal, team_clearance())


def test_mover_follows_its_segment_with_the_reported_velocity():
    mover = Mover(a=(0., -1., 1.), b=(0., 1., 1.5), speed=0.4, phase=0.3, radius=0.15)
    for t in np.linspace(0., 20., 41):
        position, velocity = mover.state(t)
        later, _ = mover.state(t+1e-4)
        assert np.isclose(np.linalg.norm(velocity), 0.4)
        if np.allclose((later-position)/1e-4, velocity, atol=1e-6):
            continue
        # only at a turning point may the finite difference disagree
        assert min(np.linalg.norm(position-np.array(mover.a)), np.linalg.norm(position-np.array(mover.b))) < 1e-3


def test_clearances_cover_aircraft_payload_and_cables():
    pillars = np.array([[0., 0., 0.1]])
    clear = team_clearances([[0.3, 0., 1.], [0.3, 0.5, 1.], [0.3, -0.5, 1.]], [0.5, 0., 0.5], pillars, [], [])
    assert clear["uav"] == pytest.approx(0.3-0.1-UAV_RADIUS)
    assert clear["payload"] == pytest.approx(0.5-0.1-PAYLOAD_RADIUS)
    assert clear["cable"] == pytest.approx(0.3-0.1)
    crossing = team_clearances([[0.3, 0., 1.], [0.3, 0.5, 1.], [0.3, -0.5, 1.]], [-0.3, 0., 0.5], pillars, [], [])
    assert crossing["cable"] == pytest.approx(-0.1)
    sphere = team_clearances([[0., 0., 2.], [1., 0., 2.], [2., 0., 2.]], [0., 0., 1.], np.zeros((0, 3)),
                             [[0.5, 0., 1.5]], [0.2])
    assert sphere["cable"] == pytest.approx(-0.2)   # the sphere centre lies on the second cable
    assert sphere["payload"] == pytest.approx(np.sqrt(0.5)-0.2-PAYLOAD_RADIUS)
    assert sphere["uav"] == pytest.approx(np.sqrt(0.5)-0.2-UAV_RADIUS)


@needs_cc
def test_batched_controller_matches_the_pinned_native_adapter():
    from uav_arena.env import quaternion_matrices
    from uav_arena.lowlevel import BatchedLee
    from uav_transport.control import mix_bounded, rotor_mixer
    from uav_transport.external_controller import NativeLee
    from uav_transport.menagerie import MenagerieTransportConfig

    config = MenagerieTransportConfig()
    rng = np.random.default_rng(0)
    quaternion = rng.normal(size=(3, 4)); quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    state = dict(position=rng.normal(size=(3, 3)), quaternion_wxyz=quaternion, velocity=rng.normal(size=(3, 3))*0.3,
                 omega_body=rng.normal(size=(3, 3)), goal=rng.normal(size=(3, 3)), acceleration=rng.normal(size=(3, 3)),
                 goal_velocity=rng.normal(size=(3, 3))*0.5)
    batched = BatchedLee(3, mass=config.drone_mass, inertia=config.inertia, mixer=rotor_mixer(config),
                         rotor_max=config.rotor_max)
    native = [NativeLee(config.drone_mass, config.inertia, 1/500) for _ in range(3)]
    rotations = quaternion_matrices(quaternion)
    for _ in range(3):
        forces = batched.step(**state).copy()
        for i, controller in enumerate(native):
            wrench = controller.step(state["position"][i], state["velocity"][i], rotations[i], state["omega_body"][i],
                                     state["goal"][i], state["goal_velocity"][i], state["acceleration"][i])
            np.testing.assert_allclose(batched.wrenches.reshape(3, 4)[i], wrench, rtol=1e-4, atol=1e-7)
            np.testing.assert_allclose(forces[i], mix_bounded(wrench, rotor_mixer(config), config.rotor_max),
                                       rtol=1e-4, atol=1e-7)


@needs_cc
def test_bound_feedback_tick_matches_the_array_interface():
    from uav_arena.env import ArenaEnv, NOMINAL_PAYLOAD_MASS
    env = ArenaEnv(open_field())
    for _ in range(3):
        env.step(env.true_state()["uav_pos"]+[0.1, 0., 0.])
    memory = env.lee.capture()
    q, v = env.data.qpos, env.data.qvel
    positions = q[env._uav_pos].reshape(3, 3)
    cables = positions-q[env._payload_pos]
    directions = cables/np.linalg.norm(cables, axis=1, keepdims=True)
    tension = NOMINAL_PAYLOAD_MASS*env.config.gravity/(3*np.maximum(directions[:, 2], 0.2))
    expected = env.lee.step(positions, q[env._uav_quat], v[env._uav_lin], v[env._uav_ang], env.setpoints,
                            tension[:, None]*directions/env.config.drone_mass, env.setpoint_velocity).copy()
    assert np.abs(env.setpoint_velocity).max() > 0.1
    env.lee.restore(memory)
    env._tick()
    np.testing.assert_allclose(env.data.ctrl[env._ctrl].reshape(3, 4), expected, rtol=1e-6, atol=1e-9)


@needs_cc
def test_warm_start_hover_is_settled():
    from uav_arena.env import ArenaEnv
    env = ArenaEnv(open_field())
    start = env.true_state()
    for _ in range(20):
        env.step(start["uav_pos"])
    assert np.linalg.norm(env.true_state()["payload_pos"]-start["payload_pos"]) < 0.005
    assert env.outcome == "running" and env.true_state()["time"] == pytest.approx(2.0)


@needs_cc
def test_observations_lag_the_true_state_by_the_latency():
    from uav_arena.env import ArenaEnv
    fresh, delayed, half = (ArenaEnv(open_field(d)) for d in (0, 100, 50))
    observations = {0: [fresh.reset()], 100: [delayed.reset()], 50: [half.reset()]}
    for k in range(12):
        setpoints = observations[0][-1]["uav_pos"]+[0.05*np.cos(k), 0.05, 0.]
        for latency, env in ((0, fresh), (100, delayed), (50, half)):
            observations[latency].append(env.step(setpoints)[0])
    for k in range(2, 13):
        np.testing.assert_array_equal(observations[100][k]["uav_pos"], observations[0][k-1]["uav_pos"])
        assert observations[100][k]["time"] == pytest.approx(observations[0][k]["time"]-0.1)
        record = next(r for r in fresh.log if abs(r["time"]-(observations[0][k]["time"]-0.05)) < 1e-9)
        np.testing.assert_array_equal(observations[50][k]["payload_pos"], record["payload_pos"])
    assert observations[100][5]["recent_setpoints"].shape == (4, 3, 3)


@needs_cc
def test_snapshot_restore_replays_bitwise():
    from uav_arena.env import ArenaEnv
    env = ArenaEnv(make_scenario(2, Condition("static+moving", 50)))
    obs = env.reset()
    for _ in range(10):
        obs, *_ = env.step(obs["uav_pos"]+[0.1, 0., 0.])
    snapshot = env.capture()
    plan = [obs["uav_pos"]+[0.1, 0.05*k, 0.] for k in range(8)]
    first = [env.step(p)[0] for p in plan]
    env.restore(snapshot)
    second = [env.step(p)[0] for p in plan]
    for a, b in zip(first, second):
        for key in ("uav_pos", "uav_vel", "payload_pos", "payload_vel", "mover_pos"):
            np.testing.assert_array_equal(a[key], b[key])


@needs_cc
def test_flying_through_a_pillar_fails_and_open_transport_succeeds():
    from uav_arena.env import ArenaEnv
    env = ArenaEnv(open_field(pillars=[Pillar(0., 0., 0.15)]))
    obs = env.reset()
    policy = smooth_transfer(obs)
    while not env.done:
        obs, reward, done, info = env.step(policy(obs))
    assert info["outcome"] == "collision_obstacle" and reward < -5
    env = ArenaEnv(open_field(latency_ms=200))
    obs = env.reset()
    policy = smooth_transfer(obs)
    while not env.done:
        obs, reward, done, info = env.step(policy(obs))
    assert info["outcome"] == "success" and info["distance_to_goal"] < 0.15
