"""Train the official TD-MPC2 online on arena tasks (TD-MPC2 venv).

The pinned upstream trainer, agent and hyperparameters run unchanged through its own Hydra entry
point, `train.py`. The task `uav-arena` joins its `envs.make_env` the way `envs/mujoco.py` adds
Gymnasium tasks: `ArenaGymEnv` behind upstream's old-gym contract and its `Timeout` and
`TensorWrapper`. What the arena adds to `config.yaml`'s defaults:
- `episodic=true`, since arena events end episodes and the default rejects terminations;
- `rho=0.7`, upstream's setting for tasks with terminations (`envs/mujoco.py`, issue #62);
- no wandb and no video. Upstream then saves no agent, so the launcher saves the final one.

Writes `<out>.pt` (the final agent), `<out>.json` (a report with the resolved configuration),
`<out>.log.jsonl` (upstream's train and eval logs) and upstream's work directory `<out>/`. The
arena trains 2M environment steps with seeds 42, 67 and 2027 (TASKS.md), for example:

    PYTHONPATH=$PWD data/external/.venv-tdmpc2/bin/python -m uav_arena.tdmpc2_train \
        --checkout data/external/tdmpc2 --out data/uav_arena/tdmpc2/s42 --seed 42 --steps 2000000
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import json
import os
from pathlib import Path
import runpy
import sys
import time

import gymnasium
import numpy as np

from .env import DECISION_DT
from .gym_env import ArenaGymEnv
from .scenarios import Scenario
from .upstream import importable, plain, provenance, sha256

UPSTREAM_URL = "https://github.com/nicklashansen/tdmpc2"
UPSTREAM_COMMIT = "332f19e4aba4bffe2dc736f45ea7f7bdc585903a"
TASK = "uav-arena"
EPISODE_DECISIONS = round(Scenario.time_limit/DECISION_DT)
RHO = 0.7
OVERRIDES = dict(task=TASK, episodic="true", enable_wandb="false", save_video="false")


class ArenaTask(gymnasium.Wrapper):
    """TD-MPC2's environment contract, as `envs/mujoco.py` wraps Gymnasium tasks: `reset()` returns the
    observation and `step()` returns (observation, reward, done, info) with `info['terminated']`."""

    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed

    def reset(self, **kwargs):
        observation, _ = self.env.reset(seed=self.seed)        # seeds the layout stream at the first reset
        self.seed = None
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not (np.isfinite(observation).all() and np.isfinite(reward)):  # would poison the buffer silently
            raise FloatingPointError(f"Non-finite arena step: {info}")
        return observation, reward, terminated or truncated, dict(info, terminated=terminated)


def make_env(cfg):
    """`envs.make_env` for `task=uav-arena`: upstream's single-task branch around `ArenaTask`."""
    if cfg.task != TASK:
        raise ValueError(f"Unknown task: {cfg.task}")
    from envs.wrappers.tensor import TensorWrapper
    from envs.wrappers.timeout import Timeout
    gym_env = ArenaGymEnv()
    gym_env.action_space.seed(cfg.seed)                        # upstream's random seed actions
    env = TensorWrapper(Timeout(ArenaTask(gym_env, cfg.seed), max_episode_steps=EPISODE_DECISIONS))
    cfg.rho = RHO
    cfg.obs_shape = {cfg.get("obs", "state"): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5*cfg.episode_length)
    return env


def _scalars(values):
    out = {}
    for key, value in values.items():
        if hasattr(value, "item"):                              # NumPy and torch scalars
            try:
                value = value.item()
            except (ValueError, RuntimeError):
                continue
        if isinstance(value, (bool, int, float)):
            out[key] = value
    return out


@contextmanager
def _registered(out):
    """Upstream's modules with the arena task registered and logs and the final agent written beside `out`;
    yields what the run reports."""
    import envs
    from common import logger as upstream_logger
    Logger, run = upstream_logger.Logger, {}
    saved = envs.make_env, Logger.log, Logger.finish

    def log_too(self, d, category="train"):
        with out.with_suffix(".log.jsonl").open("a") as f:
            f.write(json.dumps(dict(_scalars(d), category=category, wall_time=time.time()))+"\n")
        if category == "eval":
            run["final_eval"] = _scalars(d)
        return saved[1](self, d, category)

    def finish_and_save(self, agent=None):
        agent.save(out.with_suffix(".pt"))
        run.update(config=plain(dataclasses.asdict(agent.cfg)), work_dir=str(agent.cfg.work_dir))
        return saved[2](self, agent)

    envs.make_env, Logger.log, Logger.finish = make_env, log_too, finish_and_save
    try:
        yield run
    finally:
        envs.make_env, Logger.log, Logger.finish = saved


def train(checkout, out, *, seed, steps, overrides=()):
    import torch
    checkout, out = Path(checkout).resolve(), Path(out).resolve()
    if any(p.exists() for p in (out, out.with_suffix(".json"), out.with_suffix(".pt"), out.with_suffix(".log.jsonl"))):
        raise FileExistsError(f"Keep earlier runs: {out}")
    upstream = provenance(checkout, UPSTREAM_URL, UPSTREAM_COMMIT)
    out.mkdir(parents=True)
    settings = dict(OVERRIDES, seed=int(seed), steps=int(steps))
    argv = [f"{k}={v}" for k, v in settings.items()]+list(overrides)+[f"hydra.run.dir={out/'hydra'}"]
    started, cwd, saved_argv = time.perf_counter(), os.getcwd(), sys.argv
    with importable(checkout/"tdmpc2"), _registered(out) as run:
        os.chdir(out)                                           # upstream's work_dir: <out>/logs/<task>/<seed>/
        sys.argv = [str(checkout/"tdmpc2"/"train.py"), *argv]
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")    # as `python train.py ...`, for Hydra
        finally:
            os.chdir(cwd)
            sys.argv = saved_argv
    import mujoco
    report = dict(kind="tdmpc2_on_arena_tasks", upstream=upstream, overrides=argv, rho=RHO,
                  episode_decisions=EPISODE_DECISIONS, wall_seconds=time.perf_counter()-started, **run,
                  checkpoint=str(out.with_suffix(".pt")), checkpoint_sha256=sha256(out.with_suffix(".pt")),
                  source_sha256=sha256(__file__), versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                  numpy=np.__version__, mujoco=mujoco.__version__, gymnasium=gymnasium.__version__),
                  device=torch.cuda.get_device_name(0))
    out.with_suffix(".json").write_text(json.dumps(plain(report), indent=1)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", type=Path, default=Path("data/external/tdmpc2"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=float, default=10_000_000, help="environment steps (upstream default 10M)")
    parser.add_argument("--override", action="append", default=[],
                        help="a further Hydra override, e.g. compile=false; recorded in the report")
    args = parser.parse_args()
    report = train(args.checkout, args.out, seed=args.seed, steps=args.steps, overrides=args.override)
    print(json.dumps({k: report.get(k) for k in ("wall_seconds", "final_eval", "checkpoint")}, indent=1))


if __name__ == "__main__":
    main()
