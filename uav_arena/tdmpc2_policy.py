"""A trained TD-MPC2 agent as a `gym_eval` policy (TD-MPC2 venv): upstream's planner in eval mode.

The agent is rebuilt from the configuration its training report recorded and loads the final
weights. Each episode reseeds torch with the layout seed, so the planner's sampling replays alone.

    PYTHONPATH=$PWD data/external/.venv-tdmpc2/bin/python -m uav_arena.gym_eval \
        --policy uav_arena.tdmpc2_policy:make --policy-options '{"run": "data/uav_arena/tdmpc2/s1"}' \
        --suite dev --seeds 0:5 --out data/uav_arena/results/tdmpc2_s1_dev_s0-5.json
"""
from __future__ import annotations

import json
from pathlib import Path

from .tdmpc2_train import UPSTREAM_COMMIT, UPSTREAM_URL
from .upstream import importable, provenance, sha256


class TDMPC2Policy:
    def __init__(self, run, compile=None):
        import torch
        from omegaconf import OmegaConf
        report = json.loads(Path(run).with_suffix(".json").read_text())
        if sha256(report["checkpoint"]) != report["checkpoint_sha256"]:
            raise ValueError(f"{report['checkpoint']} is not the agent this run saved")
        checkout = Path(provenance(report["upstream"]["checkout"], UPSTREAM_URL, UPSTREAM_COMMIT)["checkout"])
        config = dict(report["config"])
        if compile is not None:
            config["compile"] = bool(compile)
        with importable(checkout/"tdmpc2"):
            from common.parser import cfg_to_dataclass
            from tdmpc2 import TDMPC2
            self.cfg = cfg_to_dataclass(OmegaConf.create(config))
            self.agent = TDMPC2(self.cfg)
            self.agent.load(report["checkpoint"])
        self.torch, self.t0 = torch, True

    def reset(self, seed):
        self.torch.manual_seed(int(seed))
        self.t0 = True

    def act(self, observation):
        if self.cfg.compile:
            self.torch.compiler.cudagraph_mark_step_begin()
        action = self.agent.act(self.torch.from_numpy(observation), t0=self.t0, eval_mode=True)
        self.t0 = False
        return action.numpy()


def make(run, compile=None):
    return TDMPC2Policy(run, compile)
