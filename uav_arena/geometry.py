"""Signed clearances between the team (aircraft, payload, cables) and obstacles.

MuJoCo tendons have no collision geometry, so cable crossings are computed here
as segment distances. Obstacles are non-physical volumes: entering one is a
failure, not a contact force. Positive clearance means free space.
"""
from __future__ import annotations

import numpy as np


UAV_RADIUS = 0.06        # Crazyflie 2 rotor tips around the body centre
PAYLOAD_RADIUS = 0.025   # payload sphere in uav_transport.plant


def point_segment_distance(points, a, b):
    """Distance from points (..., d) to segments a-b (..., d), broadcasting."""
    ab = b-a
    t = np.clip(np.sum((points-a)*ab, axis=-1)/np.maximum(np.sum(ab*ab, axis=-1), 1e-18), 0.0, 1.0)
    return np.linalg.norm(points-(a+t[..., None]*ab), axis=-1)


def team_clearances(uav_pos, payload_pos, pillars, mover_pos, mover_radius):
    """Smallest clearance of aircraft, payload and cables to pillars and movers.

    pillars: (P, 3) rows of x, y, radius (vertical cylinders spanning the workspace height).
    mover_pos: (M, 3) sphere centres with mover_radius (M,).
    Returns a dict of minimum clearances per team part; inf when there are no obstacles.
    """
    uav_pos, payload_pos = np.asarray(uav_pos, dtype=float), np.asarray(payload_pos, dtype=float)
    result = dict(uav=np.inf, payload=np.inf, cable=np.inf)
    pillars = np.asarray(pillars, dtype=float).reshape(-1, 3)
    if len(pillars):
        centres, radii = pillars[:, :2], pillars[:, 2]
        uav = np.linalg.norm(uav_pos[:, None, :2]-centres[None], axis=-1)-radii-UAV_RADIUS
        payload = np.linalg.norm(payload_pos[None, :2]-centres, axis=-1)-radii-PAYLOAD_RADIUS
        cable = point_segment_distance(centres[None], payload_pos[None, None, :2], uav_pos[:, None, :2])-radii
        result.update(uav=float(uav.min()), payload=float(payload.min()), cable=float(cable.min()))
    mover_pos = np.asarray(mover_pos, dtype=float).reshape(-1, 3)
    if len(mover_pos):
        radius = np.asarray(mover_radius, dtype=float).reshape(-1)
        uav = np.linalg.norm(uav_pos[:, None]-mover_pos[None], axis=-1)-radius-UAV_RADIUS
        payload = np.linalg.norm(payload_pos[None]-mover_pos, axis=-1)-radius-PAYLOAD_RADIUS
        cable = point_segment_distance(mover_pos[None], payload_pos[None, None], uav_pos[:, None])-radius
        result = dict(uav=min(result["uav"], float(uav.min())), payload=min(result["payload"], float(payload.min())),
                      cable=min(result["cable"], float(cable.min())))
    return result


def pillar_array(scenario):
    return np.array([[p.x, p.y, p.radius] for p in scenario.pillars], dtype=float).reshape(-1, 3)


def mover_radii(scenario):
    return np.array([m.radius for m in scenario.movers], dtype=float)
