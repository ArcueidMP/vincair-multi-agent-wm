"""W-ar state head and multistep objective, independent of Yihao's backbone.

The backbone contract is:

    backbone(state_history, recent_setpoints, recent_times, setpoint, age)
        -> token features with shape (B, 4, backbone.token_dim)

``state_history`` is normalized physical state (B, history, 51); setpoints are
relative to the latest observed payload position. The first transition spans
the observation age plus one decision, then each subsequent transition spans
one decision. The backbone owns the entity tokenizer and full attention mask.
"""
from __future__ import annotations

import torch
from torch import nn

from ...env import DECISION_DT
from ...windows import STATE_KEYS
from ..state import STATE_DIM, pack_state


class EntityDeltaHead(nn.Module):
    """Entity heads assembled into the field-major 51-state layout."""

    def __init__(self, token_dim: int):
        super().__init__()
        self.aircraft = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim),
                                      nn.SiLU(), nn.Linear(token_dim, 15))
        self.payload = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim),
                                     nn.SiLU(), nn.Linear(token_dim, 6))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != 4:
            raise ValueError("the backbone must return three aircraft tokens and one payload token")
        aircraft = self.aircraft(tokens[:, :3])
        # Each token predicts its own pos/vel/rot6/omega, but pack_state is
        # field-major rather than aircraft-major. Match that executable layout
        # before adding the increment to the current state.
        return torch.cat((aircraft[..., :3].flatten(1),
                          aircraft[..., 3:6].flatten(1),
                          aircraft[..., 6:12].flatten(1),
                          aircraft[..., 12:15].flatten(1),
                          self.payload(tokens[:, 3])), dim=-1)


def state_history(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Observed state frames from windows.py, relative to the newest payload."""
    origin = batch["obs_payload_pos"][:, -1]
    states = {key: batch[f"obs_{key}"] for key in STATE_KEYS}
    return pack_state(states, origin[:, None]), origin


def future_states(batch: dict[str, torch.Tensor], origin: torch.Tensor) -> torch.Tensor:
    states = {key: batch[f"next_{key}"] for key in STATE_KEYS}
    return pack_state(states, origin[:, None])


def rollout(backbone: nn.Module, head: EntityDeltaHead, history_states: torch.Tensor,
            origin: torch.Tensor, age: torch.Tensor, recent: torch.Tensor,
            recent_times: torch.Tensor, setpoints: torch.Tensor,
            stats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Open-loop physical states, with gradients through every predicted step."""
    history = history_states
    outputs = []
    size = len(history)
    for step in range(setpoints.shape[1]):
        normalized_history = (history - stats["state_mean"]) / stats["state_std"]
        tokens = backbone(normalized_history, recent - origin[:, None, None, :],
                          recent_times, setpoints[:, step] - origin[:, None, :], age)
        normalized_delta = head(tokens)
        if normalized_delta.shape != (size, STATE_DIM):
            raise ValueError(f"state head returned {tuple(normalized_delta.shape)}, expected {(size, STATE_DIM)}")
        next_state = history[:, -1] + normalized_delta * stats["delta_std"] + stats["delta_mean"]
        outputs.append(next_state)
        history = torch.cat((history[:, 1:], next_state[:, None]), dim=1)
        recent = torch.cat((recent[:, 1:], setpoints[:, step:step + 1]), dim=1)
        recent_times = torch.cat((recent_times[:, 1:] - DECISION_DT,
                                  torch.full((size, 1), -DECISION_DT, device=recent_times.device,
                                             dtype=recent_times.dtype)), dim=1)
        age = torch.zeros_like(age)
    return torch.stack(outputs, dim=1)


def multistep_loss(backbone: nn.Module, head: EntityDeltaHead,
                   batch: dict[str, torch.Tensor], stats: dict[str, torch.Tensor],
                   steps: int = 15) -> torch.Tensor:
    """MSE of normalized *states* at every predicted decision, no latent loss."""
    history, origin = state_history(batch)
    truth = future_states(batch, origin)[:, :steps]
    predicted = rollout(backbone, head, history, origin, batch["age"],
                        batch["recent_setpoints"], batch["recent_setpoint_times"],
                        batch["setpoints"][:, :steps], stats)
    return ((predicted - truth) / stats["state_std"]).square().mean()
