"""Gymnasium interface to the arena, for methods that train online (TD-MPC2, DreamerV3).

One episode is one scenario: a training layout and condition drawn at reset, or a given layout seed
and condition for evaluation. The agent acts at the decision rate (10 Hz), as every participant does.

- Action (9,) in [-1, 1]: each aircraft's setpoint increment per axis in units of `ACTION_LIMIT`
  (0.12 m), added to the setpoint in force, as the shared MPPI's candidates are. The onboard
  geofence applies as for every participant.
- Observation (`OBSERVATION_SIZE`,) float32: the decision maker's current observation, flattened by
  `flatten`. The team and movers are seen `latency` late; the payload mass and the undelayed state
  never appear (PARTICIPANTS.md, Section 3). No history beyond the setpoints in force.
- Reward: `ArenaEnv`'s, 10 times the payload's progress to the goal minus 0.01 per decision, and
  +10 for a success or -10 for any other end, the time limit included.
- An episode ends on an arena event or success (`terminated`) or at the 20 s time limit (`truncated`).

Training layouts are drawn from the world-model training seeds and the 8 conditions uniformly:

    env = ArenaGymEnv()
    observation, info = env.reset(seed=0)
    observation, info = env.reset(options=dict(seed=1000, condition="static+moving/100ms"))
"""
from __future__ import annotations

import gymnasium
from gymnasium.spaces import Box
import numpy as np

from .env import ACTION_LIMIT, RECENT_SETPOINTS, ArenaEnv
from .scenarios import CONDITIONS, Condition, make_scenario


TRAINING_SEEDS = range(200_000, 299_000)    # world-model training layouts (PARTICIPANTS.md, Section 4)
N_PILLARS, N_MOVERS = 4, 2
OBSERVATION_SIZE = 6+3*15+3+3*N_PILLARS+7*N_MOVERS+9*RECENT_SETPOINTS+2


def _rows(values, count, width):
    """`values` as (count, width) rows, zero-padded."""
    out = np.zeros((count, width))
    values = np.asarray(values, dtype=float).reshape(-1, width)[:count]
    out[:len(values)] = values
    return out


def flatten(obs, time_limit):
    """The observation vector of an `ArenaEnv` observation, in this order:
    payload position and velocity (6); per aircraft position, velocity, the first two columns of its
    rotation and its angular velocity (3 x 15); the goal (3); pillars as x, y, radius (4 x 3); movers as
    position, velocity, radius, zero when absent (2 x 7); the setpoints in force, oldest first, padded at
    the episode start with the hover setpoint as `windows.padded_recent` pads them (4 x 9); the latency
    (s); and the share of the time limit left."""
    setpoints = np.asarray(obs["recent_setpoints"], dtype=float)
    setpoints = np.concatenate([np.repeat(setpoints[:1], RECENT_SETPOINTS-len(setpoints), axis=0), setpoints])
    team = np.concatenate([obs["uav_pos"], obs["uav_vel"], obs["uav_rot6"], obs["uav_omega"]], axis=1)
    movers = np.concatenate([_rows(obs["mover_pos"], N_MOVERS, 3), _rows(obs["mover_vel"], N_MOVERS, 3),
                             _rows(obs["mover_radius"], N_MOVERS, 1)], axis=1)
    parts = (obs["payload_pos"], obs["payload_vel"], team, obs["goal"], _rows(obs["pillars"], N_PILLARS, 3), movers,
             setpoints, (obs["latency"], 1.0-obs["decision_time"]/time_limit))
    return np.concatenate([np.asarray(p, dtype=float).reshape(-1) for p in parts]).astype(np.float32)


def _condition(condition):
    return condition if isinstance(condition, Condition) else next(c for c in CONDITIONS if c.name == condition)


class ArenaGymEnv(gymnasium.Env):
    """The arena as a Gymnasium environment at the decision rate."""

    metadata = {"render_modes": []}

    def __init__(self, seeds=TRAINING_SEEDS, conditions=CONDITIONS):
        self.seeds, self.conditions = seeds, tuple(_condition(c) for c in conditions)
        # Unbounded: velocities and rates have no natural bounds, and DreamerV3 asserts declared ones.
        self.observation_space = Box(-np.inf, np.inf, (OBSERVATION_SIZE,), np.float32)
        self.action_space = Box(-1.0, 1.0, (9,), np.float32)
        self.scenario = self.arena = self.last = None

    def reset(self, *, seed=None, options=None):
        """A layout and condition drawn with the environment's generator, or `options=dict(seed=, condition=)`."""
        super().reset(seed=seed)
        if options and "seed" in options:
            layout, condition = int(options["seed"]), _condition(options["condition"])
        else:
            layout = int(self.seeds[int(self.np_random.integers(len(self.seeds)))])
            condition = self.conditions[int(self.np_random.integers(len(self.conditions)))]
        self.scenario = make_scenario(layout, condition)
        self.arena = ArenaEnv(self.scenario)            # constructed reset: the settled hover at t = 0
        self.last = self.arena.observation()
        return flatten(self.last, self.scenario.time_limit), self._info(self.arena.info())

    def step(self, action):
        increments = ACTION_LIMIT*np.clip(np.asarray(action, dtype=float).reshape(3, 3), -1.0, 1.0)
        self.last, reward, done, info = self.arena.step(self.last["recent_setpoints"][-1]+increments)
        truncated = done and info["outcome"] == "timeout"
        return (flatten(self.last, self.scenario.time_limit), float(reward), done and not truncated, truncated,
                self._info(info))

    def _info(self, info):
        """`ArenaEnv`'s info with the layout seed and condition; never the payload mass."""
        return dict(info, seed=self.scenario.seed, condition=self.scenario.condition.name,
                    success=info["outcome"] == "success")
