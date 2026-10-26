from dataclasses import replace
import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_worldmodel")
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.collect import collect_episode
from uav_arena.env import ArenaEnv
from uav_arena.openloop import PER_DECISION, evaluate, observation
from uav_arena.planning import PhysicsPredictor
from uav_arena.scenarios import Condition, Scenario
import json


def recorded(seed=200_003, decisions=30):
    episode = collect_episode(seed)
    episode["layout"] = Scenario.from_dict(json.loads(str(episode["scenario"])))
    records = PER_DECISION*decisions+1
    for key in ("time", "uav_pos", "uav_vel", "uav_rot6", "uav_omega", "payload_pos", "payload_vel"):
        episode[key] = episode[key][:records]
    episode["setpoints"], episode["setpoint_time"] = episode["setpoints"][:decisions+1], episode["setpoint_time"][:decisions+1]
    return episode


@pytest.fixture(scope="module")
def episode():
    return recorded()


@pytest.mark.parametrize("latency_ms", [0, 50, 200])
def test_rebuilt_observations_equal_the_live_ones(episode, latency_ms):
    layout = replace(episode["layout"], condition=Condition("static", latency_ms))
    env = ArenaEnv(replace(layout, payload_mass=float(episode["payload_mass"])))
    live = env.reset()
    for k, setpoints in enumerate(episode["setpoints"][1:]):
        rebuilt = observation(episode, k, latency_ms)
        assert rebuilt.keys() == live.keys()
        for key in live:
            np.testing.assert_array_equal(rebuilt[key], live[key], err_msg=key)
        live, *_ = env.step(setpoints)


@pytest.mark.parametrize("latency_ms", [0, 200])
def test_physics_with_the_true_mass_predicts_recorded_episodes_exactly(episode, latency_ms):
    predictor = PhysicsPredictor(workers=2, nominal_mass=float(episode["payload_mass"]))
    try:
        result = evaluate(predictor, [episode], latency_ms, horizon=5, stride=4, dtype=torch.float64)
    finally:
        predictor.close()
    assert result["decisions"] == 7
    assert result["payload"].max() < 1e-6 and result["uav"].max() < 1e-6
