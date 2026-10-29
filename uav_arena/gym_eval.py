"""Fly a policy that acts through the Gymnasium wrapper on an arena suite and write per-episode records.

    PYTHONPATH=. python -m uav_arena.gym_eval --policy module:factory --policy-options '{"checkpoint": "..."}' \
        --suite dev --seeds 0:5 --out <records.json>

The factory is called with the JSON `--policy-options` as keyword arguments and returns a policy with
`reset(seed)`, called at each episode's start with the layout seed so that a stochastic policy can
reseed and every episode replays alone, and `act(observation)`, from the observation vector of
`gym_env.flatten` to an action in [-1, 1]^9. Records and summaries are run.py's; `plan_seconds` time
`act`. `--policy uav_arena.gym_eval:hold` holds every setpoint: a smoke test, and a floor.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import time

import numpy as np

from .gym_env import ArenaGymEnv
from .run import summarize
from .scenarios import CONDITIONS, SUITE_SEEDS


class Hold:
    def reset(self, seed):
        pass

    def act(self, observation):
        return np.zeros(9, dtype=np.float32)


def hold():
    return Hold()


def make_policy(path, options=None):
    module, _, factory = path.partition(":")
    return getattr(importlib.import_module(module), factory)(**(options or {}))


def run_episode(env, policy, seed, condition):
    observation, info = env.reset(options=dict(seed=seed, condition=condition))
    policy.reset(seed)
    seconds, done = [], False
    while not done:
        start = time.perf_counter()
        action = policy.act(observation)
        seconds.append(time.perf_counter()-start)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return dict(seed=seed, condition=info["condition"], payload_mass=env.scenario.payload_mass,
                outcome=info["outcome"], time=info["time"], decisions=info["decisions"],
                distance_to_goal=info["distance_to_goal"], min_clearance=info["min_clearance"],
                plan_seconds_mean=float(np.mean(seconds)), plan_seconds_max=float(np.max(seconds)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-options", default="{}")
    parser.add_argument("--suite", choices=tuple(SUITE_SEEDS), default="dev")
    parser.add_argument("--seeds", default="0:5")
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    start, stop = (int(x) if x else None for x in args.seeds.split(":"))
    conditions = CONDITIONS if args.conditions == "all" else [c for c in CONDITIONS if c.name in args.conditions.split(",")]
    policy, env, records = make_policy(args.policy, json.loads(args.policy_options)), ArenaGymEnv(), []
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        for seed in SUITE_SEEDS[args.suite][start:stop]:
            records.append(run_episode(env, policy, seed, condition))
            print(f"{condition.name:>20} seed {seed:>4}: {records[-1]['outcome']:<18} t={records[-1]['time']:5.1f}s "
                  f"act={records[-1]['plan_seconds_mean']*1000:.1f}ms", flush=True)
            out.write_text(json.dumps(dict(config=vars(args), records=records, summary=summarize(records)), indent=1)+"\n")
    for row in summarize(records):
        print(row)


if __name__ == "__main__":
    main()
