"""PETS participant contract and delay-alignment tests."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from torch.utils.data import default_collate

from uav_arena.collect import _write
from uav_arena.openloop import load_episode, observation
from uav_arena.participants.pets import train
from uav_arena.participants.pets.model import INPUT_DIM, HISTORY, STATE_DIM, make, pack_state, transition
from uav_arena.windows import STATE_KEYS, window


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("pets")
    _write((200_003, root))
    checkpoint = root / "pets.pt"
    train.main(["--data", str(root), "--episodes", "1", "--members", "2", "--steps", "2",
                "--batch", "4", "--stats-samples", "32", "--width", "16", "--depth", "1",
                "--workers", "0", "--device", "cpu", "--out", str(checkpoint)])
    return root, checkpoint


def test_delay_bridge_and_following_transition_use_the_correct_states_and_actions(trained):
    root, _ = trained
    episode = load_episode(root / "episode_200003.npz")
    batch = default_collate([window(episode, 10, 50, horizon=15, history=HISTORY)])
    observed = {key: batch[f"obs_{key}"][:, -1] for key in STATE_KEYS}
    origin = observed["payload_pos"]
    first = {key: batch[f"next_{key}"][:, 0] for key in STATE_KEYS}
    second = {key: batch[f"next_{key}"][:, 1] for key in STATE_KEYS}
    x0, y0 = transition(batch, torch.tensor([0]))
    x1, y1 = transition(batch, torch.tensor([1]))
    x_last, y_last = transition(batch, torch.tensor([14]))
    assert x0.shape == x1.shape == (1, INPUT_DIM) and y0.shape == y1.shape == (1, STATE_DIM)
    assert x_last.shape == x0.shape and y_last.shape == y0.shape
    torch.testing.assert_close(y0, pack_state(first, origin) - pack_state(observed, origin))
    torch.testing.assert_close(y1, pack_state(second, origin) - pack_state(first, origin))
    torch.testing.assert_close(x0[:, HISTORY * STATE_DIM], batch["age"])
    assert torch.equal(x1[:, HISTORY * STATE_DIM], torch.zeros_like(batch["age"]))
    torch.testing.assert_close(x1[:, HISTORY * STATE_DIM + 28:HISTORY * STATE_DIM + 37],
                               (batch["setpoints"][:, 0] - first["payload_pos"][:, None]).flatten(1))
    torch.testing.assert_close(x1[:, (HISTORY - 1) * STATE_DIM + 45:(HISTORY - 1) * STATE_DIM + 48],
                               torch.zeros((1, 3)))


def test_predictor_contract_reset_and_no_mass_access(trained):
    root, checkpoint = trained
    episode = load_episode(root / "episode_200003.npz")
    predictor = make(str(checkpoint), device="cpu")
    assert predictor.sample_gaussian is False

    class NoMass:
        def __getattribute__(self, key):
            if key == "payload_mass":
                raise AssertionError("the predictor tried to read payload mass")
            return super().__getattribute__(key)

    obs = observation(episode, 10, 50)
    assert "payload_mass" not in obs
    assert predictor.stochastic is True
    predictor.observe(obs, NoMass())
    for size in (1, 512, 555):
        output = predictor(None, torch.zeros((1, size, 15, 3, 3)))
        assert output["uav"].shape == (1, size, 15, 3, 3)
        assert output["payload"].shape == output["payload_vel"].shape == (1, size, 15, 3)
        assert all(torch.isfinite(value).all() and value.dtype == torch.float32 for value in output.values())
    setpoints = torch.zeros((1, 1, 15, 3, 3))
    predictor.reset()
    predictor.observe(obs, NoMass())
    first = predictor(None, setpoints)
    predictor.reset()
    predictor.observe(obs, NoMass())
    second = predictor(None, setpoints)
    assert all(torch.equal(first[key], second[key]) for key in first)
    torch.manual_seed(17)
    first = predictor(None, setpoints)
    torch.manual_seed(17)
    second = predictor(None, setpoints)
    assert all(torch.equal(first[key], second[key]) for key in first)
    predictor.reset()
    predictor.observe(observation(episode, 0, 200), NoMass())
    start = predictor(None, setpoints)
    assert all(torch.isfinite(value).all() for value in start.values())
    predictor.reset()
    with pytest.raises(RuntimeError, match="observe"):
        predictor(None, torch.zeros((1, 512, 15, 3, 3)))
    assert np.isfinite(obs["payload_pos"]).all()
