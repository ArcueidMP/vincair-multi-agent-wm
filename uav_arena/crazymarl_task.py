"""CrazyMARL on arena tasks: the task definition shared by training (JAX) and evaluation (NumPy).

The official CrazyMARL method is kept: decentralized IPPO with parameter sharing, the
official networks and hyperparameters, per-motor thrust commands at 250 Hz, and its
domain randomization and reward terms. Only what the arena task needs is added:

- the arena plant: Menagerie Crazyflie 2 mass and inertia, 0.6 m cables, the nominal 25 g
  payload, 30 ms motors, rotors up to 0.16 N;
- goal transit from the arena's settled hover start, and a progress reward;
- obstacles: their positions in the observation, and collisions that end the episode;
- latency: each aircraft's own state is fresh, while the payload, teammates and obstacles
  arrive `latency` late, as in the arena.

Functions take the array module `xp` (numpy or jax.numpy) so that training and evaluation
compute observations and clearances with the same code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


POLICY_HZ = 250.0
N_QUADS, N_PILLARS, N_MOVERS = 3, 4, 2
LATENCIES_S = (0.0, 0.05, 0.1, 0.2)
HISTORY = 51                         # delayed-observable vectors kept: now and 50 policy steps (200 ms) back
DELAYED_SIZE = 6+3*N_QUADS+6*N_MOVERS  # payload position and velocity, aircraft positions, mover states
EVALUATION_MAX_THRUST = 0.15         # N per rotor: centre of the training randomization
UAV_RADIUS, PAYLOAD_RADIUS = 0.06, 0.025
TRAINING_SEEDS = range(100_000, 102_048)   # disjoint from the arena's dev and test suites


def latency_steps(latency_s):
    return int(round(latency_s*POLICY_HZ))


def mover_states(xp, movers, t):
    """movers (M, 9): a (3), b (3), speed, phase, radius; radius 0 marks an absent mover."""
    a, b, speed, phase = movers[:, 0:3], movers[:, 3:6], movers[:, 6], movers[:, 7]
    length = xp.maximum(xp.linalg.norm(b-a, axis=-1), 1e-9)
    period = 2.0*length/xp.maximum(speed, 1e-9)
    s = ((t/period+phase) % 1.0)*2.0
    forward = s < 1.0
    fraction = xp.where(forward, s, 2.0-s)
    direction = (b-a)/length[:, None]
    position = a+fraction[:, None]*(b-a)
    velocity = xp.where(forward, 1.0, -1.0)[:, None]*speed[:, None]*direction
    present = (movers[:, 8] > 0)[:, None]
    return xp.where(present, position, 0.0), xp.where(present, velocity, 0.0)


def _point_segment(xp, points, a, b):
    ab = b-a
    t = xp.clip(xp.sum((points-a)*ab, axis=-1)/xp.maximum(xp.sum(ab*ab, axis=-1), 1e-18), 0.0, 1.0)
    return xp.linalg.norm(points-(a+t[..., None]*ab), axis=-1)


def team_clearance(xp, uav_pos, payload_pos, pillars, mover_pos, mover_radius):
    """Smallest clearance of aircraft, payload and cables to the obstacles (as uav_arena.geometry)."""
    centres, radii = pillars[:, :2], pillars[:, 2]
    uav = xp.linalg.norm(uav_pos[:, None, :2]-centres[None], axis=-1)-radii-UAV_RADIUS
    payload = xp.linalg.norm(payload_pos[None, :2]-centres, axis=-1)-radii-PAYLOAD_RADIUS
    cable = _point_segment(xp, centres[None], payload_pos[None, None, :2], uav_pos[:, None, :2])-radii
    absent = xp.where(mover_radius > 0, 0.0, 1e3)
    uav_m = xp.linalg.norm(uav_pos[:, None]-mover_pos[None], axis=-1)-mover_radius-UAV_RADIUS+absent
    payload_m = xp.linalg.norm(payload_pos[None]-mover_pos, axis=-1)-mover_radius-PAYLOAD_RADIUS+absent
    cable_m = _point_segment(xp, mover_pos[None], payload_pos[None, None], uav_pos[:, None])-mover_radius+absent
    return xp.min(xp.stack([xp.min(uav), xp.min(payload), xp.min(cable), xp.min(uav_m), xp.min(payload_m),
                            xp.min(cable_m)]))


def delayed_vector(xp, payload_pos, payload_vel, uav_pos, mover_pos, mover_vel):
    return xp.concatenate([payload_pos, payload_vel, uav_pos.reshape(-1), mover_pos.reshape(-1),
                           mover_vel.reshape(-1)])


def global_observation(xp, *, goal, delayed, own_pos, rotations, velocities, rates, last_action, pillars,
                       mover_radius, latency):
    """CrazyMARL's observation, with each quantity at the time the aircraft would know it.

    Layout: payload error (capped as upstream) and velocity; per aircraft [position relative to
    the payload (own fresh, payload delayed), rotation (9), velocity, body rates, last action (4)];
    per aircraft delayed position relative to the payload (what teammates see); pillars relative to
    the payload, nearest first (dx, dy, radius); movers (relative position, velocity, radius);
    the latency in seconds.
    """
    payload_pos, payload_vel = delayed[0:3], delayed[3:6]
    seen_uav = delayed[6:6+3*N_QUADS].reshape(N_QUADS, 3)
    mover_pos = delayed[6+3*N_QUADS:6+3*N_QUADS+3*N_MOVERS].reshape(N_MOVERS, 3)
    mover_vel = delayed[6+3*N_QUADS+3*N_MOVERS:].reshape(N_MOVERS, 3)
    error = goal-payload_pos
    error = error/xp.maximum(xp.linalg.norm(error), 1.0)
    blocks = xp.concatenate([own_pos-payload_pos, rotations.reshape(N_QUADS, 9), velocities, rates,
                             xp.clip(last_action.reshape(N_QUADS, 4), -1.0, 1.0)], axis=1)
    relative = pillars[:, :2]-payload_pos[:2]
    order = xp.argsort(xp.linalg.norm(relative, axis=-1)-pillars[:, 2])
    pillar_features = xp.concatenate([relative, pillars[:, 2:3]], axis=1)[order]
    present = (mover_radius > 0)[:, None]
    mover_features = xp.where(present, xp.concatenate([mover_pos-payload_pos, mover_vel, mover_radius[:, None]],
                                                      axis=1), 0.0)
    return xp.concatenate([error, payload_vel, blocks.reshape(-1), (seen_uav-payload_pos).reshape(-1),
                           pillar_features.reshape(-1), mover_features.reshape(-1), xp.reshape(latency, (1,))])


OWN_BLOCK = 22
TEAM_START = 6+OWN_BLOCK*N_QUADS
OBSTACLE_START = TEAM_START+3*N_QUADS
GLOBAL_SIZE = OBSTACLE_START+3*N_PILLARS+7*N_MOVERS+1


def agent_indices(agent):
    """Upstream ix4 mapping (payload terms, own block, teammates' relative positions) plus obstacles and latency."""
    own = list(range(6+OWN_BLOCK*agent, 6+OWN_BLOCK*(agent+1)))
    team = [TEAM_START+3*j+k for j in range(N_QUADS) if j != agent for k in range(3)]
    return np.array(list(range(6))+own+team+list(range(OBSTACLE_START, GLOBAL_SIZE)), dtype=int)


def noise_scale():
    """Per-entry observation noise (upstream scales for the upstream entries) for obs_noise = 1."""
    block = [0.002]*12+[0.004]*6+[0.0]*4
    pillar = [0.002, 0.002, 0.0]*N_PILLARS
    mover = ([0.002]*3+[0.004]*3+[0.0])*N_MOVERS
    return np.array([0.002]*6+block*N_QUADS+[0.002]*3*N_QUADS+pillar+mover+[0.0])


def build_task(path, seeds=TRAINING_SEEDS):
    """Training layouts from the arena generator and the settled-hover constants (arena venv)."""
    from uav_transport.environment import MENAGERIE, make_config
    from uav_transport.plant import formation_offsets, initial_hover
    from scipy.spatial.transform import Rotation
    from .env import NOMINAL_PAYLOAD_MASS
    from .scenarios import Condition, make_scenario

    starts, goals, pillars, movers = [], [], [], []
    for seed in seeds:
        for setting in ("static", "static+moving"):
            scenario = make_scenario(seed, Condition(setting, 0))
            starts.append(scenario.start); goals.append(scenario.goal)
            pillars.append([[p.x, p.y, p.radius] for p in scenario.pillars])
            rows = [[*m.a, *m.b, m.speed, m.phase, m.radius] for m in scenario.movers]
            movers.append(rows+[[0.0]*9]*(N_MOVERS-len(rows)))
    config = make_config(MENAGERIE, payload_mass=NOMINAL_PAYLOAD_MASS)
    thrusts, rotations = initial_hover(config)
    task = dict(start=np.array(starts), goal=np.array(goals), pillars=np.array(pillars), movers=np.array(movers),
                formation=formation_offsets(config), hover_quat=Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]],
                hover_thrust=thrusts, drone_mass=np.array(config.drone_mass), inertia=np.array(config.inertia),
                cable_length=np.array(config.cable_length), payload_mass=np.array(NOMINAL_PAYLOAD_MASS),
                motor_tau=np.array(config.motor_tau), rotor_max=np.array(config.rotor_max))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **task)
    return task


def load_task(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
