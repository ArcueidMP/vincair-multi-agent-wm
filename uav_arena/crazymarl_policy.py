"""Fly a trained CrazyMARL policy in the arena (arena venv, NumPy only).

The policy runs per aircraft at 250 Hz and writes rotor thrusts directly, replacing the
onboard Lee feedback, as CrazyMARL does onboard a Crazyflie. Its own state is fresh; the
payload, teammates and obstacles reach it `latency` late. Outcomes use the arena's events
and success rule.

    PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.crazymarl_policy \
        --actor data/uav_arena/crazymarl/ippo_s0.npz --suite dev --seeds 0:5 --out <records.json>
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

from . import crazymarl_task as task
from .env import ArenaEnv, quaternion_matrices

# Upstream rotor sites (crazymarl/builders/quad_env_builder.py): thrust1 (+x, -y), thrust2 (-x, -y),
# thrust3 (-x, +y), thrust4 (+x, +y), yaw gear -, +, -, +. The arena's rotors run (+, +), (-, +),
# (-, -), (+, -) with yaw signs +, -, +, -, so arena rotor j is upstream rotor 3 - j.
UPSTREAM_ROTOR_SITES = np.array([[1, -1], [-1, -1], [-1, 1], [1, 1]])*0.032527
UPSTREAM_YAW_SIGNS = np.array([-1, 1, -1, 1])
UPSTREAM_ROTOR_FOR_ARENA = (3, 2, 1, 0)


class CrazyMARLActor:
    """The upstream actor MLP (tanh hidden layers, linear mean); evaluation uses the mean action."""

    def __init__(self, path):
        with np.load(path) as weights:
            count = sum(1 for key in weights.files if key.startswith("kernel"))
            self.layers = [(weights[f"kernel{k}"], weights[f"bias{k}"]) for k in range(count)]

    def __call__(self, observations):
        x = observations
        for kernel, bias in self.layers[:-1]:
            x = np.tanh(x@kernel+bias)
        kernel, bias = self.layers[-1]
        return np.clip(x@kernel+bias, -1.0, 1.0)


class CrazyMARLPilot:
    """`env.step_motors(pilot)` callback: the policy at 250 Hz with the arena's information delays."""

    def __init__(self, actor, task_path, max_thrust=task.EVALUATION_MAX_THRUST):
        self.actor, self.max_thrust = actor, float(max_thrust)
        constants = task.load_task(task_path)
        self.hover_action = 2.0*constants["hover_thrust"].reshape(-1)/self.max_thrust-1.0
        self.indices = np.stack([task.agent_indices(i) for i in range(task.N_QUADS)])

    def reset(self, env):
        self.period = round(1.0/(task.POLICY_HZ*env.config.dt))
        self.lag = round(env.latency/env.config.dt)
        self.history = deque(maxlen=self.lag+1)
        self.last_action = self.hover_action.copy()
        self.pillars = np.asarray(env.pillars).reshape(-1, 3)
        self.mover_radius = np.zeros(task.N_MOVERS)
        self.mover_radius[:len(env.mover_radius)] = env.mover_radius
        self.steps = 0
        self._remember(env)

    def _movers(self, env, t):
        position, velocity = np.zeros((task.N_MOVERS, 3)), np.zeros((task.N_MOVERS, 3))
        p, v = env.scenario.mover_states(t)
        position[:len(p)], velocity[:len(v)] = p, v
        return position, velocity

    def _remember(self, env):
        q, v = env.data.qpos, env.data.qvel
        mover_pos, mover_vel = self._movers(env, float(env.data.time))
        self.history.append(task.delayed_vector(np, q[env._payload_pos], v[env._payload_lin],
                                                q[env._uav_pos].reshape(3, 3), mover_pos, mover_vel))
        while len(self.history) < self.history.maxlen:      # a settled hover before t = 0
            self.history.appendleft(self.history[0])

    def __call__(self, env, k):
        if self.steps > 0:
            self._remember(env)
        if self.steps % self.period == 0:
            q, v = env.data.qpos, env.data.qvel
            obs = task.global_observation(
                np, goal=env.goal, delayed=self.history[0], own_pos=q[env._uav_pos].reshape(3, 3),
                rotations=quaternion_matrices(q[env._uav_quat].reshape(3, 4)), velocities=v[env._uav_lin].reshape(3, 3),
                rates=v[env._uav_ang].reshape(3, 3), last_action=self.last_action, pillars=self.pillars,
                mover_radius=self.mover_radius, latency=env.latency)
            actions = self.actor(obs[self.indices])
            self.last_action = actions.reshape(-1)
            thrust = 0.5*(actions+1.0)*self.max_thrust
            env.data.ctrl[env._ctrl] = thrust[:, list(UPSTREAM_ROTOR_FOR_ARENA)].reshape(-1)
        self.steps += 1


def run_episode(scenario, pilot):
    env = ArenaEnv(scenario)
    env.reset()
    pilot.reset(env)
    while not env.done:
        _, _, _, info = env.step_motors(pilot)
    return dict(seed=scenario.seed, condition=scenario.condition.name, outcome=info["outcome"], time=info["time"],
                decisions=info["decisions"], distance_to_goal=info["distance_to_goal"],
                min_clearance=info["min_clearance"], payload_mass=scenario.payload_mass)


def main():
    from .run import summarize
    from .scenarios import CONDITIONS, SUITE_SEEDS, make_scenario
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--task", type=Path, default=Path("data/uav_arena/crazymarl/task.npz"))
    parser.add_argument("--suite", choices=tuple(SUITE_SEEDS), default="dev")
    parser.add_argument("--seeds", default="0:5")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start, stop = (int(x) if x else None for x in args.seeds.split(":"))
    conditions = CONDITIONS if args.conditions == "all" else [c for c in CONDITIONS if c.name in args.conditions.split(",")]
    pilot = CrazyMARLPilot(CrazyMARLActor(args.actor), args.task)
    records = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        for seed in SUITE_SEEDS[args.suite][start:stop]:
            records.append(run_episode(make_scenario(seed, condition), pilot))
            print(f"{condition.name:>20} seed {seed:>4}: {records[-1]['outcome']:<18} t={records[-1]['time']:5.1f}s",
                  flush=True)
            args.out.write_text(json.dumps(dict(actor=str(args.actor), records=records, summary=summarize(records)),
                                           indent=1)+"\n")
    for row in summarize(records):
        print(row)


if __name__ == "__main__":
    main()
