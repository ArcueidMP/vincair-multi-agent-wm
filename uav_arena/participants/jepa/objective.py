"""JEPA latent prediction, SIGReg and physical readout, following VMAS LeWM."""
from __future__ import annotations

import torch
from torch import nn

from module import SIGReg
from ...env import DECISION_DT
from ...windows import STATE_KEYS
from .backbone import UAVBackbone, mlp, pack_state


def window_states(batch):
    """Pack observed and target states about one newest observed origin."""
    origin = batch["obs_payload_pos"][:, -1]
    history = pack_state({key: batch[f"obs_{key}"] for key in STATE_KEYS}, origin[:, None])
    future = pack_state({key: batch[f"next_{key}"] for key in STATE_KEYS}, origin[:, None])
    return history, future, origin


def cost_features(state):
    """The shared planner reads aircraft positions, payload position and velocity."""
    return torch.cat((state[..., :9], state[..., 45:51]), dim=-1)


class StateReadout(nn.Module):
    """Type-shared heads decode the 15 physical features used by the cost."""

    def __init__(self, token_dim, attn):
        super().__init__()
        self.attn = attn
        if attn == "flat":
            self.flat = mlp(token_dim, 15)
        else:
            self.aircraft, self.payload = mlp(token_dim, 3), mlp(token_dim, 6)

    def forward(self, z):
        if self.attn == "flat":
            return self.flat(z[..., 0, :])
        aircraft = self.aircraft(z[..., :3, :])
        return torch.cat((aircraft.flatten(-2), self.payload(z[..., 3, :])), dim=-1)


class EntityJEPA(nn.Module):
    """End-to-end latent prediction and state readout; no simulator access."""

    def __init__(self, backbone_options=None, sigreg_weight=0.09, head_weight=1.0,
                 sigreg_per_token=False, sigreg_projections=1024):
        super().__init__()
        self.backbone = UAVBackbone(**(backbone_options or {}))
        self.readout = StateReadout(self.backbone.token_dim, self.backbone.attn)
        self.sigreg = SIGReg(num_proj=sigreg_projections)
        self.config = dict(backbone_options=self.backbone.config, sigreg_weight=sigreg_weight,
                           head_weight=head_weight, sigreg_per_token=sigreg_per_token,
                           sigreg_projections=sigreg_projections)

    def rollout(self, history, recent, times, setpoints, age):
        """Latent autoregression, retaining gradients to all previous decisions."""
        outputs = []
        for step in range(setpoints.shape[1]):
            z = self.backbone.predict(history, recent, times, setpoints[:, step], age)
            outputs.append(z)
            history = torch.cat((history[:, 1:], z[:, None]), dim=1)
            recent = torch.cat((recent[:, 1:], setpoints[:, step:step+1]), dim=1)
            times = torch.cat((times[:, 1:]-DECISION_DT, torch.full_like(times[:, :1], -DECISION_DT)), dim=1)
            age = torch.zeros_like(age)
        return torch.stack(outputs, dim=1)

    def regularization(self, z):
        if self.config["sigreg_per_token"] and self.backbone.attn != "flat":
            return 0.5*(self.sigreg(z[..., :3, :].reshape(1, -1, z.shape[-1]))
                        + self.sigreg(z[..., 3:, :].reshape(1, -1, z.shape[-1])))
        return self.sigreg(z.reshape(1, -1, z.shape[-1]))

    def loss(self, batch, stats, steps=15):
        history, future, origin = window_states(batch)
        future = future[:, :steps]
        states = torch.cat((history, future), dim=1)
        normalized = (states-stats["state_mean"])/stats["state_std"]
        frames = history.shape[1]
        z = self.backbone.encode(normalized)
        predicted = self.rollout(z[:, :frames], batch["recent_setpoints"]-origin[:, None, None],
                                 batch["recent_setpoint_times"], batch["setpoints"][:, :steps]-origin[:, None, None],
                                 batch["age"])
        latent = (predicted-z[:, frames:]).square().mean()
        sigreg = self.regularization(z)
        physical = cost_features(normalized)
        encoded_state, predicted_state = self.readout(z), self.readout(predicted)
        absolute = ((encoded_state-physical).square().mean()
                    + (predicted_state-physical[:, frames:]).square().mean())
        # Match the upstream readout loss: predicted current state minus the
        # encoded previous true state, not detached autoregressive outputs.
        delta_true = physical[:, frames:]-physical[:, frames-1:-1]
        delta_pred = predicted_state-encoded_state[:, frames-1:-1]
        delta = ((delta_pred-delta_true)/cost_features(stats["delta_std"])).square().mean()
        total = latent+self.config["sigreg_weight"]*sigreg+self.config["head_weight"]*(absolute+delta)
        return total, dict(latent=latent.detach(), sigreg=sigreg.detach(),
                           head_absolute=absolute.detach(), head_delta=delta.detach())
