"""Train the official DreamerV3 online on arena tasks (DreamerV3 venv).

The pinned upstream `dreamerv3/main.py` runs unchanged with its `dmc_proprio` preset, the one for
proprioceptive continuous control (model size 1M, train ratio 1024, 1.1M environment steps, 16
environments). The suite `arena` joins its `make_env` the way it builds every suite: `ArenaTask`
puts `ArenaGymEnv` behind the `embodied.Env` contract, seeded per environment as `use_seed` seeds
them, and upstream's `wrap_env` wraps it. A timeout is the last step of an episode but not a terminal
one. Upstream saves a checkpoint every 15 minutes and resumes from the latest one when the same
command runs again (`--resume`); the launcher adds the final agent.

Writes upstream's logdir `<out>/` (config.yaml, metrics.jsonl, scores.jsonl, replay and ckpt/), the
final agent `<out>/final/` and a report `<out>.json`. The arena trains 2M environment steps with
seeds 42, 67 and 2027 (TASKS.md), for example:

    PYTHONPATH=$PWD data/external/.venv-dreamerv3/bin/python -m uav_arena.dreamerv3_train \
        --checkout data/external/dreamerv3 --out data/uav_arena/dreamerv3/s42 --seed 42 --steps 2000000
"""
from __future__ import annotations

import argparse
import functools
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import gymnasium
import numpy as np

from .gym_env import OBSERVATION_SIZE, ArenaGymEnv
from .upstream import importable, plain, provenance, sha256

UPSTREAM_URL = "https://github.com/danijar/dreamerv3"
UPSTREAM_COMMIT = "e01491fad6434b2245a3b8ca201dd7faedcc458c"
PRESET = "dmc_proprio"
TASK = "arena_train"


class ArenaTask:
    """`embodied.Env` around `ArenaGymEnv`, as `embodied/envs/from_gym.py` wraps Gym tasks: the `reset`
    action starts an episode, and every step reports `is_first`, `is_last` and `is_terminal`."""

    def __init__(self, task, seed=None):
        if task != "train":
            raise ValueError(f"Unknown arena task: {task}")
        self.env, self.seed, self.done = ArenaGymEnv(), seed, True

    @functools.cached_property
    def obs_space(self):
        import elements
        return {"state": elements.Space(np.float32, (OBSERVATION_SIZE,)), "reward": elements.Space(np.float32),
                "is_first": elements.Space(bool), "is_last": elements.Space(bool),
                "is_terminal": elements.Space(bool), "log/success": elements.Space(np.float32)}

    @functools.cached_property
    def act_space(self):
        import elements
        return {"action": elements.Space(np.float32, (9,), -1.0, 1.0), "reset": elements.Space(bool)}

    def step(self, action):
        if action["reset"] or self.done:
            observation, _ = self.env.reset(seed=self.seed)    # seeds the layout stream at the first reset
            self.seed, self.done = None, False
            return self._obs(observation, 0.0, is_first=True)
        observation, reward, terminated, truncated, info = self.env.step(action["action"])
        self.done = terminated or truncated
        return self._obs(observation, reward, is_last=self.done, is_terminal=terminated, success=info["success"])

    def _obs(self, observation, reward, is_first=False, is_last=False, is_terminal=False, success=False):
        return {"state": observation, "reward": np.float32(reward), "is_first": is_first, "is_last": is_last,
                "is_terminal": is_terminal, "log/success": np.float32(success)}

    def close(self):
        pass


def register(upstream_main):
    """Add the suite `arena` to upstream's `make_env`; returns the agents its `make_agent` builds."""
    official_make_env, official_make_agent, agents = upstream_main.make_env, upstream_main.make_agent, []

    def make_env(config, index, **overrides):
        suite, task = config.task.split("_", 1)
        if suite != "arena":
            return official_make_env(config, index, **overrides)
        seed = hash((config.seed, index)) % (2**32-1)          # as main.py's use_seed
        return upstream_main.wrap_env(ArenaTask(task, seed=seed, **overrides), config)

    def make_agent(config):
        agents.append(official_make_agent(config))
        return agents[-1]

    upstream_main.make_env, upstream_main.make_agent = make_env, make_agent
    return agents


def train(checkout, out, *, seed, steps=None, configs=(PRESET,), flags=(), resume=False):
    checkout, out = Path(checkout).resolve(), Path(out).resolve()
    if (out.exists() or out.with_suffix(".json").exists()) and not resume:
        raise FileExistsError(f"Keep earlier runs, or continue this one with --resume: {out}")
    upstream = provenance(checkout, UPSTREAM_URL, UPSTREAM_COMMIT)
    argv = ["--configs", *configs, "--task", TASK, "--logdir", str(out), "--seed", str(int(seed))]
    argv += ["--run.steps", str(int(steps))] if steps else []
    argv += list(flags)
    started = time.perf_counter()
    with importable(checkout):
        import elements
        from dreamerv3 import main as upstream_main
        agents = register(upstream_main)
        upstream_main.main(argv)
        final = elements.Checkpoint()
        final.agent = agents[-1]
        if (out/"final").exists():
            (out/"final").rename(out/f"final-before-{elements.timestamp()}")
        final.save(out/"final", keys=["agent"])
    import jax
    import mujoco
    report = dict(kind="dreamerv3_on_arena_tasks", upstream=upstream, argv=argv,
                  config=dict(elements.Config.load(out/"config.yaml")), logdir=str(out),
                  wall_seconds=time.perf_counter()-started, checkpoint=str(out/"final"),
                  source_sha256=sha256(__file__), versions=dict(python=sys.version.split()[0], jax=jax.__version__,
                  jaxlib=version("jaxlib"), numpy=np.__version__, mujoco=mujoco.__version__,
                  gymnasium=gymnasium.__version__, elements=version("elements"), ninjax=version("ninjax")),
                  devices=[str(d) for d in jax.devices()])
    out.with_suffix(".json").write_text(json.dumps(plain(report), indent=1)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", type=Path, default=Path("data/external/dreamerv3"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=float, default=None, help="environment steps (preset: 1.1M)")
    parser.add_argument("--configs", nargs="+", default=[PRESET])
    parser.add_argument("--resume", action="store_true", help="continue from the logdir's latest checkpoint")
    parser.add_argument("flags", nargs="*", help="further upstream flags after --, e.g. -- --run.envs 8")
    args = parser.parse_args()
    report = train(args.checkout, args.out, seed=args.seed, steps=args.steps, configs=args.configs, flags=args.flags,
                   resume=args.resume)
    print(json.dumps({k: report[k] for k in ("wall_seconds", "checkpoint")}, indent=1))


if __name__ == "__main__":
    main()
