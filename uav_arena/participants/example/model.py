"""Example participant: a direct multi-step MLP world model.

One forward pass maps what the predictor sees before a decision (the delayed team state, its age
and the four setpoints in force) and the 15 candidate setpoints to the team state at the end of
each of the 15 decisions. Positions are taken relative to the observed payload position, so the
model does not depend on where the team is. It trains on `windows.py` examples (`train.py`) and
plugs into the shared planner and `openloop.py` through `make`:

    python -m uav_arena.run --predictor uav_arena.participants.example.model:make \
        --predictor-options '{"checkpoint": "data/uav_arena/participants/example/mlp.pt"}' \
        --suite dev --seeds 0:5 --out data/uav_arena/results/example_mlp_dev_s0-5.json

A template for the interface and the data path, kept deliberately simple; it is not a benchmark row.
"""
from __future__ import annotations

import torch
from torch import nn

from ...windows import STATE_KEYS, padded_recent

HORIZON = 15            # the shared planner's horizon (decisions)
PER_STEP = 15           # predicted per decision: aircraft positions (9), payload position (3) and velocity (3)


def features(state, age, recent, setpoints):
    """Network input and its origin (the observed payload position).

    state: STATE_KEYS tensors with a batch dimension B; age: (B,); recent: (B, 4, 3, 3); setpoints: (B, H, 3, 3).
    """
    origin = state["payload_pos"]
    o = origin[:, None, :]
    parts = (state["uav_pos"]-o, state["uav_vel"], state["uav_rot6"], state["uav_omega"], state["payload_vel"],
             age[:, None], recent-o[:, None], setpoints-o[:, None])
    return torch.cat([p.flatten(1) for p in parts], 1), origin


def targets(batch, origin):
    """Training targets in the network's output layout: (B, H * PER_STEP)."""
    o = origin[:, None, :]
    return torch.cat([(batch["next_uav_pos"]-o[:, :, None]).flatten(2), batch["next_payload_pos"]-o,
                      batch["next_payload_vel"]], -1).flatten(1)


def decode(y, origin):
    """Network output to the planner's prediction dict."""
    y = y.unflatten(1, (-1, PER_STEP))
    o = origin[:, None, :]
    return dict(uav=y[..., :9].unflatten(-1, (3, 3))+o[:, :, None], payload=y[..., 9:12]+o, payload_vel=y[..., 12:])


def network(config):
    layers, width = [], config["inputs"]
    for _ in range(config["depth"]):
        layers += [nn.Linear(width, config["width"]), nn.SiLU()]
        width = config["width"]
    return nn.Sequential(*layers, nn.Linear(width, config["outputs"]))


def predict(net, stats, state, age, recent, setpoints):
    x, origin = features(state, age, recent, setpoints)
    y = net((x-stats["x_mean"])/stats["x_std"])*stats["y_std"]+stats["y_mean"]
    return decode(y, origin)


class ExamplePredictor:
    """The participant object the shared planner and `openloop.py` call (PARTICIPANTS.md, Section 2)."""

    def __init__(self, checkpoint, device="cpu"):
        pack = torch.load(checkpoint, map_location=device)
        self.device = torch.device(device)
        self.net = network(pack["config"]).to(self.device).eval()
        self.net.load_state_dict(pack["state"])
        self.stats = {key: value.to(self.device) for key, value in pack["stats"].items()}
        self.obs = None

    def reset(self):
        self.obs = None

    def observe(self, obs, layout):
        recent, _ = padded_recent(obs)
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.obs = dict(state={key: tensor(obs[key])[None] for key in STATE_KEYS},
                        age=tensor([obs["decision_time"]-obs["time"]]), recent=tensor(recent)[None])

    @torch.no_grad()
    def __call__(self, info, setpoints):
        lead, horizon = setpoints.shape[:-3], setpoints.shape[-3]
        candidates = setpoints.to(self.device, torch.float32).reshape(-1, horizon, 3, 3)
        expand = lambda x: x.expand(len(candidates), *x.shape[1:])
        prediction = predict(self.net, self.stats, {key: expand(v) for key, v in self.obs["state"].items()},
                             expand(self.obs["age"]), expand(self.obs["recent"]), candidates)
        return {key: value.reshape(*lead, *value.shape[1:]).to(setpoints.device, setpoints.dtype)
                for key, value in prediction.items()}


def make(checkpoint, device="cpu"):
    """`module:factory` entry point for `run.py` and `openloop.py`."""
    return ExamplePredictor(checkpoint, device)
