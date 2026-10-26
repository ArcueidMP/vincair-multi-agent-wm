from dataclasses import replace
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.collect import collect_episode
from uav_arena.env import ACTION_LIMIT, ArenaEnv
from uav_arena.scenarios import Scenario


def test_collected_episodes_are_deterministic_and_replay_exactly():
    episode = collect_episode(200_003)
    for key, value in collect_episode(200_003).items():
        np.testing.assert_array_equal(value, episode[key])
    assert np.abs(np.diff(episode["setpoints"], axis=0)).max() <= ACTION_LIMIT+1e-12
    layout = Scenario.from_dict(json.loads(str(episode["scenario"])))
    assert layout.payload_mass is None and layout.pillars == () and layout.movers == ()
    env = ArenaEnv(replace(layout, payload_mass=float(episode["payload_mass"])))
    env.reset()
    np.testing.assert_array_equal(env.actions[0]["setpoints"], episode["setpoints"][0])
    for setpoints in episode["setpoints"][1:]:
        env.step(setpoints)
    assert env.outcome == str(episode["outcome"])
    for key in ("payload_pos", "payload_vel", "uav_pos", "uav_vel"):
        np.testing.assert_array_equal(np.stack([r[key] for r in env.log]), episode[key])
