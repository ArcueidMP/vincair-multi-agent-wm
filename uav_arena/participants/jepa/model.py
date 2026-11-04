"""Arena participant adapter for a trained entity-token JEPA checkpoint."""
from __future__ import annotations

from collections import deque

import torch

from ...windows import STATE_KEYS, padded_recent
from .backbone import pack_state
from .objective import EntityJEPA, cost_features


class JEPAPredictor:
    """Cache only observation encodings; never update history during a rollout."""

    def __init__(self, checkpoint, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        pack = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = EntityJEPA(**pack["config"]).to(self.device).eval()
        self.model.load_state_dict(pack["model"])
        self.stats = {key: value.to(self.device) for key, value in pack["stats"].items()}
        self.history = self.model.backbone.history
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=self.history)
        self.context = None

    @torch.no_grad()
    def observe(self, obs, layout):
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
        state = {key: tensor(obs[key]).clone() for key in STATE_KEYS}
        if not self.frames:
            self.frames.extend([state]*(self.history-1))
        self.frames.append(state)
        origin = state["payload_pos"][None]
        states = {key: torch.stack([frame[key] for frame in self.frames])[None] for key in STATE_KEYS}
        packed = pack_state(states, origin[:, None])
        z = self.model.backbone.encode((packed-self.stats["state_mean"])/self.stats["state_std"])
        recent, times = padded_recent(obs)
        self.context = dict(z=z, origin=origin, recent=tensor(recent)[None]-origin[:, None, None],
                            times=tensor(times)[None], age=tensor([obs["decision_time"]-obs["time"]]))

    @torch.no_grad()
    def __call__(self, info, setpoints):
        if self.context is None:
            raise RuntimeError("observe must be called before JEPA prediction")
        if setpoints.ndim < 4 or setpoints.shape[-2:] != (3, 3) or setpoints.shape[-3] < 1:
            raise ValueError("setpoints must have shape (..., H, 3, 3) with a batch dimension")
        leading, horizon = setpoints.shape[:-3], setpoints.shape[-3]
        candidates = setpoints.to(self.device, torch.float32).reshape(-1, horizon, 3, 3)
        size = candidates.shape[0]
        context = {key: value.expand(size, *value.shape[1:]) for key, value in self.context.items()}
        z = self.model.rollout(context["z"], context["recent"], context["times"],
                               candidates-context["origin"][:, None, None], context["age"])
        state = self.model.readout(z)*cost_features(self.stats["state_std"])+cost_features(self.stats["state_mean"])
        origin = context["origin"][:, None]
        result = dict(uav=state[..., :9].unflatten(-1, (3, 3))+origin[:, :, None],
                      payload=state[..., 9:12]+origin, payload_vel=state[..., 12:15])
        return {key: value.reshape(*leading, *value.shape[1:]).to(setpoints.device, setpoints.dtype)
                for key, value in result.items()}


def make(checkpoint, device=None):
    return JEPAPredictor(checkpoint, device)
