import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_worldmodel")
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.collect import _write
from uav_arena.env import RECENT_SETPOINTS
from uav_arena.openloop import evaluate
from uav_arena.windows import STATE_KEYS, Windows, decisions, load, window


@pytest.fixture(scope="module")
def episodes(tmp_path_factory):
    out = tmp_path_factory.mktemp("explore")
    for seed in (200_003, 200_004):
        _write((seed, out))
    return load(out)


class Echo:
    """Returns each window's targets, so `openloop.evaluate` scores it zero exactly when the two align."""

    def __init__(self, episode, latency_ms, horizon):
        self.episode, self.latency_ms, self.horizon = episode, latency_ms, horizon

    def reset(self):
        self.k = -1

    def observe(self, obs, layout):
        self.k += 1

    def __call__(self, info, setpoints):
        w = window(self.episode, self.k, self.latency_ms, self.horizon, dtype=np.float64)
        np.testing.assert_array_equal(setpoints[0].numpy(), w["setpoints"])
        return {name: torch.as_tensor(w[f"next_{key}"]).unsqueeze(0)
                for name, key in (("uav", "uav_pos"), ("payload", "payload_pos"), ("payload_vel", "payload_vel"))}


@pytest.mark.parametrize("latency_ms", [0, 200])
def test_window_targets_are_what_openloop_scores(episodes, latency_ms):
    result = evaluate(Echo(episodes[0], latency_ms, 15), episodes[:1], latency_ms, horizon=15, stride=7,
                      dtype=torch.float64)
    assert result["decisions"] == len(range(0, decisions(episodes[0])-15+1, 7))
    assert result["payload"].max() == 0 and result["uav"].max() == 0


def test_windows_hide_the_mass_pad_the_start_and_batch(episodes):
    assert all("payload_mass" not in e for e in episodes)
    data = Windows(episodes, horizon=15, history=4)
    assert len(data) == sum(4*(decisions(e)-15+1) for e in episodes)
    first = data[0]                                   # first episode, decision 0, 0 ms
    assert first["obs_payload_pos"].shape == (4, 3) and first["next_uav_rot6"].shape == (15, 3, 6)
    np.testing.assert_array_equal(first["obs_payload_pos"][0], first["obs_payload_pos"][-1])
    assert first["recent_setpoints"].shape == (RECENT_SETPOINTS, 3, 3)
    np.testing.assert_allclose(first["recent_setpoint_times"], [-0.4, -0.3, -0.2, -0.1], atol=1e-6)
    early = data[3]                                   # decision 0 at 200 ms: only the t = 0 record exists
    assert early["latency"] == pytest.approx(0.2) and early["age"] == pytest.approx(0.0)
    last = data[len(data)-1]                          # last episode, last decision, 200 ms
    assert last["age"] == pytest.approx(0.2)
    np.testing.assert_array_equal(last["next_payload_pos"][-1], episodes[-1]["payload_pos"][-1].astype(np.float32))
    batch = torch.utils.data.default_collate([data[i] for i in range(8)])
    assert set(batch) == {f"obs_{k}" for k in STATE_KEYS} | {f"next_{k}" for k in STATE_KEYS} | {
        "latency", "age", "recent_setpoints", "recent_setpoint_times", "setpoints"}
    assert batch["setpoints"].shape == (8, 15, 3, 3) and batch["setpoints"].dtype == torch.float32
