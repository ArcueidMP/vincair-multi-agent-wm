"""W-ar outer-loop tests with a stand-in for Yihao's shared backbone."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from torch import nn
from torch.utils.data import default_collate

from uav_arena.collect import _write
from uav_arena.openloop import load_episode, observation
from uav_arena.participants.war.model import make
from uav_arena.participants.war import train
from uav_arena.participants.war.objective import EntityDeltaHead, multistep_loss
from uav_arena.windows import window


class FakeBackbone(nn.Module):
    token_dim = 8

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(51, self.token_dim)

    def forward(self, state_history, recent_setpoints, recent_times, setpoint, age):
        assert state_history.ndim == 3 and recent_setpoints.shape[-3:] == (4, 3, 3)
        assert recent_times.shape[-1] == 4 and setpoint.shape[-2:] == (3, 3)
        assert age.ndim == 1
        return self.linear(state_history[:, -1])[:, None].expand(-1, 4, -1)


def fake_factory():
    return FakeBackbone()


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    root = tmp_path_factory.mktemp("war")
    _write((200_003, root))
    return load_episode(root / "episode_200003.npz")


def test_multistep_state_objective_propagates_gradients(episode):
    batch = default_collate([window(episode, 10, 50, horizon=15, history=4)])
    backbone, head = fake_factory(), EntityDeltaHead(FakeBackbone.token_dim)
    stats = dict(state_mean=torch.zeros(51), state_std=torch.ones(51),
                 delta_mean=torch.zeros(51), delta_std=torch.ones(51))
    loss = multistep_loss(backbone, head, batch, stats, steps=3)
    assert torch.isfinite(loss)
    loss.backward()
    assert backbone.linear.weight.grad is not None
    assert head.aircraft[-1].weight.grad is not None


def test_planner_shape_reset_and_no_mass_access(episode, tmp_path):
    backbone, head = fake_factory(), EntityDeltaHead(FakeBackbone.token_dim)
    checkpoint = tmp_path / "war.pt"
    stats = dict(state_mean=torch.zeros(51), state_std=torch.ones(51),
                 delta_mean=torch.zeros(51), delta_std=torch.ones(51))
    torch.save(dict(config=dict(history=4, backbone="test_uav_arena_war:fake_factory", backbone_options={}),
                    backbone_state=backbone.state_dict(), head_state=head.state_dict(), stats=stats), checkpoint)
    predictor = make(str(checkpoint), device="cpu")

    class NoMass:
        def __getattribute__(self, key):
            if key == "payload_mass":
                raise AssertionError("the predictor tried to read payload mass")
            return super().__getattribute__(key)

    predictor.observe(observation(episode, 10, 50), NoMass())
    for size in (1, 512, 555):
        setpoints = torch.zeros((1, size, 15, 3, 3))
        output = predictor(None, setpoints)
        assert output["uav"].shape == (1, size, 15, 3, 3)
        assert output["payload"].shape == output["payload_vel"].shape == (1, size, 15, 3)
        assert all(torch.equal(output[key], predictor(None, setpoints)[key]) for key in output)
    predictor.reset()
    predictor.observe(observation(episode, 0, 200), NoMass())
    assert all(torch.isfinite(value).all() for value in predictor(None, torch.zeros((1, 1, 15, 3, 3))).values())
    predictor.reset()
    with pytest.raises(RuntimeError, match="observe"):
        predictor(None, torch.zeros((1, 512, 15, 3, 3)))


def test_training_entry_point_accepts_the_backbone_contract(tmp_path):
    # Store one standard-format episode locally; windows.load removes the
    # diagnostic mass before W-ar sees a training example.
    source = tmp_path / "episode_200003.npz"
    _write((200_003, tmp_path))
    assert source.exists()
    checkpoint = tmp_path / "trained.pt"
    train.main(["--data", str(tmp_path), "--episodes", "1", "--history", "4",
                "--steps", "1", "--rollout-steps", "2", "--batch", "2",
                "--stats-samples", "16", "--workers", "0", "--device", "cpu",
                "--backbone", "test_uav_arena_war:fake_factory", "--out", str(checkpoint)])
    assert checkpoint.exists()
