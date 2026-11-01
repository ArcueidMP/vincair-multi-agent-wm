"""Arena adapter for W-ar using the shared entity-token backbone.

The checkpoint names a ``module:factory`` published by Yihao. The factory must
return a backbone implementing the contract in ``objective.py``. This module
contains only the state head, rollout and arena adapter; it has no tokenizer or
attention-mask implementation of its own.
"""
from __future__ import annotations

from collections import deque
import importlib

import torch
from torch import nn

from ...windows import STATE_KEYS, padded_recent
from ..state import pack_state, unpack_state
from .objective import EntityDeltaHead, rollout


def build_backbone(factory: str, options: dict) -> nn.Module:
    module_name, separator, factory_name = factory.partition(":")
    if not separator:
        raise ValueError("backbone factory must be module:factory")
    backbone = getattr(importlib.import_module(module_name), factory_name)(**options)
    if not isinstance(backbone, nn.Module) or not hasattr(backbone, "token_dim"):
        raise TypeError("backbone must be an nn.Module with token_dim and the objective.py call contract")
    return backbone


class WARPredictor:
    def __init__(self, checkpoint: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        pack = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = pack["config"]
        self.history = int(config["history"])
        self.backbone = build_backbone(config["backbone"], config["backbone_options"]).to(self.device).eval()
        self.head = EntityDeltaHead(self.backbone.token_dim).to(self.device).eval()
        self.backbone.load_state_dict(pack["backbone_state"])
        self.head.load_state_dict(pack["head_state"])
        self.stats = {key: value.to(self.device) for key, value in pack["stats"].items()}
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=self.history)
        self.current = None

    def observe(self, obs, layout):
        # No layout fields are needed, including its diagnostic payload mass.
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
        state = {key: tensor(obs[key]) for key in STATE_KEYS}
        if not self.frames:
            for _ in range(self.history - 1):
                self.frames.append(state)
        self.frames.append(state)
        recent, times = padded_recent(obs)
        self.current = dict(age=tensor([obs["decision_time"] - obs["time"]]),
                            recent=tensor(recent)[None], recent_times=tensor(times)[None])

    @torch.no_grad()
    def __call__(self, info, setpoints):
        if self.current is None:
            raise RuntimeError("observe must be called before W-ar prediction")
        leading, horizon = setpoints.shape[:-3], setpoints.shape[-3]
        candidates = setpoints.to(self.device, torch.float32).reshape(-1, horizon, 3, 3)
        size = len(candidates)
        origin = self.frames[-1]["payload_pos"][None].expand(size, -1)
        history = {key: torch.stack([frame[key] for frame in self.frames], dim=0)[None]
                   for key in STATE_KEYS}
        history = pack_state({key: value.expand(size, *value.shape[1:]) for key, value in history.items()},
                             origin[:, None])
        predicted = rollout(self.backbone, self.head, history, origin,
                            self.current["age"].expand(size),
                            self.current["recent"].expand(size, -1, -1, -1),
                            self.current["recent_times"].expand(size, -1), candidates, self.stats)
        full = unpack_state(predicted, origin[:, None])
        output = dict(uav=full["uav_pos"], payload=full["payload_pos"], payload_vel=full["payload_vel"])
        return {key: value.reshape(*leading, *value.shape[1:]).to(setpoints.device, setpoints.dtype)
                for key, value in output.items()}


def make(checkpoint: str, device: str | None = None):
    """Arena ``module:factory`` entry point after Yihao's backbone is installed."""
    return WARPredictor(checkpoint, device)
