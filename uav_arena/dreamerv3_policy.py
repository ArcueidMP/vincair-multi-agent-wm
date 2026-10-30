"""A trained DreamerV3 agent as a `gym_eval` policy (DreamerV3 venv): upstream's actor in eval mode.

The agent is rebuilt from its logdir's config.yaml, without precompiling the training step, and loads
the final checkpoint. Upstream's eval mode samples from the actor; its sampling seeds count the actions
taken, so each episode restarts that count at the layout seed and replays alone.

    PYTHONPATH=$PWD data/external/.venv-dreamerv3/bin/python -m uav_arena.gym_eval \
        --policy uav_arena.dreamerv3_policy:make --policy-options '{"run": "data/uav_arena/dreamerv3/s0"}' \
        --suite dev --seeds 0:5 --out data/uav_arena/results/dreamerv3_s0_dev_s0-5.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dreamerv3_train import UPSTREAM_COMMIT, UPSTREAM_URL, register
from .upstream import importable, provenance


class DreamerV3Policy:
    def __init__(self, run, checkpoint=None):
        report = json.loads(Path(run).with_suffix(".json").read_text())
        checkout = provenance(report["upstream"]["checkout"], UPSTREAM_URL, UPSTREAM_COMMIT)["checkout"]
        with importable(checkout):
            import elements
            from dreamerv3 import main as upstream_main
            register(upstream_main)
            config = elements.Config.load(Path(report["logdir"])/"config.yaml")
            config = elements.Config({**dict(config), "jax.precompile": False, "jax.prealloc": False})
            self.agent = upstream_main.make_agent(config)
            saved = elements.Checkpoint()
            saved.agent = self.agent
            saved.load(checkpoint or report["checkpoint"], keys=["agent"])
        self.carry, self.first = None, True

    def reset(self, seed):
        self.carry, self.first = self.agent.init_policy(1), True
        with self.agent.n_actions.lock:
            self.agent.n_actions.value = int(seed)

    def act(self, observation):
        obs = {"state": np.asarray(observation, np.float32)[None], "reward": np.zeros(1, np.float32),
               "is_first": np.array([self.first]), "is_last": np.zeros(1, bool), "is_terminal": np.zeros(1, bool)}
        self.carry, acts, _ = self.agent.policy(self.carry, obs, mode="eval")
        self.first = False
        return np.asarray(acts["action"][0], np.float32)


def make(run, checkpoint=None):
    return DreamerV3Policy(run, checkpoint)
