import json
import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.env import ACTION_LIMIT, ArenaEnv
from uav_arena.gym_env import OBSERVATION_SIZE, TRAINING_SEEDS, ArenaGymEnv, flatten
from uav_arena.scenarios import CONDITIONS, Condition, make_scenario

MOVERS = slice(66, 80)
SETPOINTS = slice(80, 116)


def test_passes_the_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    check_env(ArenaGymEnv(), skip_render_check=True)


def test_steps_are_arena_steps_with_the_planners_increments():
    rng = np.random.default_rng(0)
    env = ArenaGymEnv()
    observation, info = env.reset(options=dict(seed=3, condition="static+moving/100ms"))
    arena = ArenaEnv(make_scenario(3, Condition("static+moving", 100)))
    obs, done, k = arena.observation(), False, 0
    np.testing.assert_array_equal(observation, flatten(obs, 20.0))
    while not done:
        action = rng.uniform(-0.4, 0.4, 9).astype(np.float32)
        action[k % 9] = 1.5 if k % 2 else -1.5         # outside the bounds: clipped, as the planner clips
        obs, reward, done, arena_info = arena.step(
            obs["recent_setpoints"][-1]+ACTION_LIMIT*np.clip(action.astype(float), -1, 1).reshape(3, 3))
        observation, gym_reward, terminated, truncated, info = env.step(action)
        np.testing.assert_array_equal(observation, flatten(obs, 20.0))
        assert gym_reward == reward and (terminated or truncated) == done
        k += 1
    assert len(env.arena.log) == len(arena.log)
    for ours, theirs in zip(env.arena.log, arena.log):
        for key in ours:
            np.testing.assert_array_equal(ours[key], theirs[key])
    assert (info["outcome"], info["seed"], info["condition"]) == (arena_info["outcome"], 3, "static+moving/100ms")
    assert terminated == (info["outcome"] != "timeout") and "payload_mass" not in info


def test_observation_reads_only_what_a_decision_maker_may_see():
    env = ArenaGymEnv()
    observation, info = env.reset(options=dict(seed=0, condition="static/200ms"))
    obs = env.arena.observation()
    allowed = ("payload_pos", "payload_vel", "uav_pos", "uav_vel", "uav_rot6", "uav_omega", "goal", "pillars",
               "mover_pos", "mover_vel", "mover_radius", "recent_setpoints", "latency", "decision_time")
    np.testing.assert_array_equal(flatten({key: obs[key] for key in allowed}, 20.0), observation)
    assert observation.shape == (OBSERVATION_SIZE,) == (118,) and observation.dtype == np.float32
    np.testing.assert_array_equal(observation[:3], obs["payload_pos"].astype(np.float32))
    assert not observation[MOVERS].any()               # no movers in the static conditions
    hover = observation[SETPOINTS].reshape(4, 9)
    np.testing.assert_array_equal(hover, np.repeat(hover[-1:], 4, axis=0))
    assert observation[-2] == pytest.approx(0.2) and observation[-1] == 1.0
    moving, _ = env.reset(options=dict(seed=0, condition="static+moving/0ms"))
    assert moving[MOVERS].reshape(2, 7)[:, 6].all()


def test_reset_draws_training_layouts_reproducibly():
    env, drawn = ArenaGymEnv(), []
    for seed in (0, 1, 0):
        env.reset(seed=seed)
        assert env.scenario.seed in TRAINING_SEEDS and env.scenario.condition in CONDITIONS
        drawn.append((env.scenario.seed, env.scenario.condition.name))
    assert drawn[0] == drawn[2] != drawn[1]


def test_events_terminate_and_the_time_limit_truncates():
    env = ArenaGymEnv()
    env.reset(options=dict(seed=0, condition="static/0ms"))
    steps, done = 0, False
    while not done:
        _, reward, terminated, truncated, info = env.step(np.zeros(9, dtype=np.float32))
        steps, done = steps+1, terminated or truncated
    assert (info["outcome"], terminated, truncated, steps) == ("timeout", False, True, 200) and reward < -9
    env.reset(options=dict(seed=0, condition="static/0ms"))
    uav = env.arena.observation()["uav_pos"]
    toward = (uav[1]-uav[0])/np.linalg.norm(uav[1]-uav[0])
    action = np.concatenate([toward, -toward, np.zeros(3)]).astype(np.float32)
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert (info["outcome"], terminated, truncated) == ("collision_team", True, False)


def test_evaluation_writes_run_records(tmp_path, monkeypatch):
    from uav_arena import gym_eval
    out = tmp_path/"hold.json"
    monkeypatch.setattr("sys.argv", ["gym_eval", "--policy", "uav_arena.gym_eval:hold", "--suite", "dev",
                                     "--seeds", "0:1", "--conditions", "static/0ms", "--out", str(out)])
    gym_eval.main()
    result = json.loads(out.read_text())
    record, = result["records"]
    assert list(record) == ["seed", "condition", "payload_mass", "outcome", "time", "decisions", "distance_to_goal",
                            "min_clearance", "plan_seconds_mean", "plan_seconds_max"]
    assert (record["seed"], record["condition"], record["outcome"]) == (0, "static/0ms", "timeout")
    assert result["summary"] == [dict(condition="static/0ms", episodes=1, success=0, outcomes=dict(timeout=1),
                                      success_time_mean=None)]
