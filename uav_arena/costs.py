"""The shared arena objective and constraints on predicted team trajectories (torch, batched).

Every planner scores candidates with these functions, whatever produced the
prediction (the true simulator, a world model or the kinematic reference), so
methods differ only in their predictions. Obstacle forecasts are the same for
everyone: constant-velocity extrapolation from the observed (equally delayed)
obstacle state. Constraints are satisfied when <= 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import PAYLOAD_RADIUS, UAV_RADIUS
from .scenarios import WORKSPACE


CONSTRAINTS = ("pillars", "movers", "team_separation", "floor", "hang")


@dataclass(frozen=True)
class CostWeights:
    goal: float = 1.0                 # mean payload-to-goal distance over the horizon (m)
    terminal: float = 1.0             # payload-to-goal distance at the horizon end (m)
    settle: float = 0.5               # payload speed weighted by proximity to the goal
    effort: float = 0.05              # squared setpoint increments, in (0.1 m)^2
    violation: float = 50.0           # soft penalty on constraint excess (MPPI only)
    obstacle_margin: float = 0.10     # planning margin added to every obstacle clearance (m)
    team_margin: float = 0.18         # minimum planned aircraft separation (m)
    floor_margin: float = 0.30        # minimum planned payload height (m)
    hang_margin: float = 0.30         # minimum planned height of every aircraft above the payload (m); 0.55 in hover


def forecast_movers(position, velocity, offsets):
    """Constant-velocity extrapolation: position/velocity (..., M, 3), offsets (..., H) seconds -> (..., H, M, 3)."""
    return position.unsqueeze(-3)+offsets[..., :, None, None]*velocity.unsqueeze(-3)


def _segment_point_distance(a, b, points):
    """Distance from points (..., P, d) to segments a-b (..., d) broadcast over P."""
    ab = (b-a).unsqueeze(-2)
    ap = points-a.unsqueeze(-2)
    t = ((ap*ab).sum(-1)/(ab*ab).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
    return (ap-t.unsqueeze(-1)*ab).norm(dim=-1)


def clearances(uav, payload, pillars, movers, mover_radius):
    """Per-step minimum clearance to pillars and to movers.

    uav (..., H, 3, 3), payload (..., H, 3); pillars (P, 3) rows x, y, radius;
    movers (..., H, M, 3) forecast centres with mover_radius (M,). Returns two (..., H) tensors.
    """
    lead = payload.shape[:-1]
    inf = torch.full(lead, float("inf"), dtype=payload.dtype, device=payload.device)
    pillar_clear = mover_clear = inf
    if pillars.numel():
        centres, radii = pillars[:, :2], pillars[:, 2]
        uav_d = (uav[..., :, None, :2]-centres).norm(dim=-1)-radii-UAV_RADIUS               # (..., 3, P)
        payload_d = (payload[..., None, :2]-centres).norm(dim=-1)-radii-PAYLOAD_RADIUS       # (..., P)
        cable_d = torch.stack([_segment_point_distance(payload[..., :2], uav[..., i, :2], centres)
                               for i in range(3)], -2)-radii                                  # (..., 3, P)
        pillar_clear = torch.minimum(torch.minimum(uav_d.amin((-1, -2)), payload_d.amin(-1)), cable_d.amin((-1, -2)))
    if movers.numel():
        uav_d = (uav[..., :, None, :]-movers[..., None, :, :]).norm(dim=-1)-mover_radius-UAV_RADIUS
        payload_d = (payload[..., None, :]-movers).norm(dim=-1)-mover_radius-PAYLOAD_RADIUS
        cable_d = torch.stack([_segment_point_distance(payload, uav[..., i, :], movers) for i in range(3)], -2)-mover_radius
        mover_clear = torch.minimum(torch.minimum(uav_d.amin((-1, -2)), payload_d.amin(-1)), cable_d.amin((-1, -2)))
    return pillar_clear, mover_clear


def constraint_values(prediction, info, weights=CostWeights()):
    """Worst-step constraint values (..., 5) in the order of CONSTRAINTS; <= 0 means satisfied."""
    uav, payload = prediction["uav"], prediction["payload"]
    horizon = payload.shape[-2]
    # Obstacles were observed `forecast_offset` seconds before the plan starts; step k ends 0.1 k s later.
    offsets = info["forecast_offset"]+0.1*torch.arange(1, horizon+1, dtype=payload.dtype, device=payload.device)
    if info["mover_pos"].shape[-2]:
        movers = forecast_movers(info["mover_pos"], info["mover_vel"], offsets)
    else:
        movers = payload.new_zeros(*payload.shape[:-1], 0, 3)
    # Obstacle layouts are shared by every candidate of one solve: take them from the first row.
    pillars = info["pillars"][(0,)*(info["pillars"].dim()-2)]
    radius = info["mover_radius"][(0,)*(info["mover_radius"].dim()-1)]
    pillar_clear, mover_clear = clearances(uav, payload, pillars, movers, radius)
    pairs = torch.stack([(uav[..., i, :]-uav[..., j, :]).norm(dim=-1) for i, j in ((0, 1), (0, 2), (1, 2))], -1)
    big = torch.finfo(payload.dtype).max/4
    values = torch.stack([
        (weights.obstacle_margin-pillar_clear).clamp(max=big).amax(-1),
        (weights.obstacle_margin-mover_clear).clamp(max=big).amax(-1),
        (weights.team_margin-pairs.amin(-1)).amax(-1),
        (weights.floor_margin-payload[..., 2]).amax(-1),
        (weights.hang_margin-(uav[..., 2]-payload[..., None, 2]).amin(-1)).amax(-1)], -1)
    return torch.nan_to_num(values, neginf=-1.0)


def objective(prediction, info, increments, weights=CostWeights()):
    """Task cost (..., ) without constraints: reach the goal, arrive slowly, small increments."""
    payload, goal = prediction["payload"], info["goal"].unsqueeze(-2)
    distance = (payload-goal).norm(dim=-1)                                   # (..., H)
    speed = prediction["payload_vel"].norm(dim=-1)
    (x0, x1), (y0, y1), (z0, z1) = WORKSPACE
    outside = (torch.relu(x0+0.3-payload[..., 0])+torch.relu(payload[..., 0]-x1+0.3)
               + torch.relu(y0+0.3-payload[..., 1])+torch.relu(payload[..., 1]-y1+0.3)).sum(-1)
    return (weights.goal*distance.mean(-1)+weights.terminal*distance[..., -1]
            + weights.settle*(speed*torch.exp(-distance/0.3)).mean(-1)
            + weights.effort*(increments/0.1).pow(2).sum((-1, -2))/increments.shape[-2]
            + 10.0*outside)


def soft_cost(prediction, info, increments, weights=CostWeights()):
    """Objective plus quadratic penalties on constraint excess, for sampling planners."""
    excess = torch.relu(constraint_values(prediction, info, weights))
    return objective(prediction, info, increments, weights)+weights.violation*(excess+10.0*excess.pow(2)).sum(-1)
