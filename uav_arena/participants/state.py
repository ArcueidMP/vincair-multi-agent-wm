"""Shared physical-state vector used by the PETS and W-ar participants."""
from __future__ import annotations

import torch

# Field-major: all aircraft pos, vel, rot6 and omega, then payload pos/vel.
STATE_DIM = 51


def pack_state(state: dict[str, torch.Tensor], origin: torch.Tensor) -> torch.Tensor:
    """Physical state with positions relative to a payload origin."""
    return torch.cat((
        (state["uav_pos"] - origin[..., None, :]).flatten(-2),
        state["uav_vel"].flatten(-2), state["uav_rot6"].flatten(-2),
        state["uav_omega"].flatten(-2), state["payload_pos"] - origin,
        state["payload_vel"],
    ), dim=-1)


def unpack_state(vector: torch.Tensor, origin: torch.Tensor) -> dict[str, torch.Tensor]:
    """Undo ``pack_state``; extra leading dimensions are supported."""
    return dict(
        uav_pos=vector[..., :9].unflatten(-1, (3, 3)) + origin[..., None, :],
        uav_vel=vector[..., 9:18].unflatten(-1, (3, 3)),
        uav_rot6=vector[..., 18:36].unflatten(-1, (3, 6)),
        uav_omega=vector[..., 36:45].unflatten(-1, (3, 3)),
        payload_pos=vector[..., 45:48] + origin,
        payload_vel=vector[..., 48:51],
    )
