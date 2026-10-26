import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena import crazymarl_task as task
from uav_arena.geometry import team_clearances
from uav_arena.scenarios import Condition, Mover, make_scenario


def test_upstream_rotor_layout_maps_onto_the_arena_rotors():
    from uav_arena.crazymarl_policy import UPSTREAM_ROTOR_FOR_ARENA, UPSTREAM_ROTOR_SITES, UPSTREAM_YAW_SIGNS
    from uav_transport.control import rotor_mixer
    from uav_transport.environment import MENAGERIE, make_config
    from uav_transport.plant import rotor_geometry
    config = make_config(MENAGERIE)
    positions, _ = rotor_geometry(config)
    yaw = np.sign(rotor_mixer(config)[3])
    for arena, upstream in enumerate(UPSTREAM_ROTOR_FOR_ARENA):
        np.testing.assert_allclose(positions[arena, :2], UPSTREAM_ROTOR_SITES[upstream], atol=1e-3)
        assert yaw[arena] == UPSTREAM_YAW_SIGNS[upstream]


def test_shared_mover_kinematics_and_clearance_match_the_arena():
    rng = np.random.default_rng(0)
    for seed in range(5):
        scenario = make_scenario(seed, Condition("static+moving", 0))
        movers = np.array([[*m.a, *m.b, m.speed, m.phase, m.radius] for m in scenario.movers])
        pillars = np.array([[p.x, p.y, p.radius] for p in scenario.pillars])
        for t in rng.uniform(0, 20, size=10):
            position, velocity = task.mover_states(np, movers, t)
            expected = [Mover(**dict(a=tuple(m[0:3]), b=tuple(m[3:6]), speed=m[6], phase=m[7], radius=m[8])).state(t)
                        for m in movers]
            np.testing.assert_allclose(position, [e[0] for e in expected], atol=1e-12)
            np.testing.assert_allclose(velocity, [e[1] for e in expected], atol=1e-12)
            payload = rng.uniform([-1.5, -1, 0.6], [1.5, 1, 1.6])
            uav = payload+rng.uniform(-0.4, 0.4, (3, 3))+[0, 0, 0.5]
            clearance = task.team_clearance(np, uav, payload, pillars, position, movers[:, 8])
            assert clearance == pytest.approx(min(team_clearances(uav, payload, pillars, position, movers[:, 8]).values()))
    absent = np.zeros((1, 9))
    position, velocity = task.mover_states(np, absent, 3.0)
    assert np.all(position == 0) and np.all(velocity == 0)


def test_agent_observations_select_the_upstream_terms_plus_obstacles():
    sizes = {len(task.agent_indices(i)) for i in range(3)}
    assert sizes == {61} and task.GLOBAL_SIZE == 108 and task.noise_scale().shape == (108,)
    own = [set(task.agent_indices(i)[6:28]) for i in range(3)]
    assert not (own[0] & own[1]) and not (own[1] & own[2])
    delayed = np.arange(task.DELAYED_SIZE, dtype=float)
    obs = task.global_observation(np, goal=np.zeros(3), delayed=delayed, own_pos=np.ones((3, 3)),
                                  rotations=np.tile(np.eye(3), (3, 1, 1)), velocities=np.zeros((3, 3)),
                                  rates=np.zeros((3, 3)), last_action=np.full(12, 2.0),
                                  pillars=np.array([[5., 0., .1], [1., 0., .1], [3., 0., .1], [2., 0., .1]]),
                                  mover_radius=np.array([0.15, 0.0]), latency=0.1)
    assert obs.shape == (108,) and obs[-1] == pytest.approx(0.1)
    assert np.linalg.norm(obs[0:3]) == pytest.approx(1.0)                  # capped payload error, as upstream
    assert np.all(obs[6+18:6+22] == 1.0)                                   # last action clipped to [-1, 1]
    pillar_x = obs[task.OBSTACLE_START:task.OBSTACLE_START+12].reshape(4, 3)[:, 0]
    assert np.all(np.diff(pillar_x) > 0)                                   # nearest pillar first
    assert np.all(obs[task.OBSTACLE_START+12+7:task.OBSTACLE_START+26] == 0)   # absent mover


@needs_cc
def test_pilot_sees_teammates_late_and_writes_its_rotors(tmp_path):
    from uav_arena.crazymarl_policy import CrazyMARLPilot
    from uav_arena.env import ArenaEnv
    task_path = tmp_path/"task.npz"
    task.build_task(task_path, seeds=range(100_000, 100_001))

    class Recorder:
        def __init__(self):
            self.seen = []

        def __call__(self, observations):
            self.seen.append(observations.copy())
            return np.zeros((3, 4))

    actor = Recorder()
    pilot = CrazyMARLPilot(actor, task_path)
    env = ArenaEnv(make_scenario(0, Condition("static", 100)))
    env.reset()
    pilot.reset(env)
    start = env.true_state()
    env.step_motors(pilot)
    assert len(actor.seen) == 25                                           # 250 Hz over 100 ms
    np.testing.assert_allclose(env.data.ctrl[env._ctrl], 0.5*task.EVALUATION_MAX_THRUST)
    # 100 ms of delay: at the end of the first period the payload error still comes from t = 0.
    last = actor.seen[-1][0]
    error = env.goal-start["payload_pos"]
    np.testing.assert_allclose(last[0:3], error/max(np.linalg.norm(error), 1.0), atol=1e-12)
