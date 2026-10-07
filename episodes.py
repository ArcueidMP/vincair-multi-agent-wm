"""The eval episodes: LeWM's Push-T eval protocol (50 seeded start states, goal 25 steps later), shipped in
assets/eval_episodes.npz so planning needs no dataset (scripts/export_eval_assets.py made it)."""
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"


def eval_episodes(seed):
    """-> (start states (50, 7), goal states (50, 7)) in the single-agent layout [agent xy, T xy, T angle, agent v]."""
    f = np.load(ASSETS / "eval_episodes.npz")
    if f"state_{seed}" not in f:
        raise KeyError(f"seed {seed} not in assets/eval_episodes.npz (has {sorted(k[6:] for k in f if k.startswith('state_'))})")
    return f[f"state_{seed}"], f[f"goal_state_{seed}"]


def reset(env, seed, e, state, goal_state):
    """Reset episode e as every planner here does: agent 0 and the T from the eval state, the partner at a
    seeded random spot away from the T. Returns (obs, infos, T already placed: the T need not move)."""
    obs, infos = env.reset(seed=seed + e, options={"state": state, "goal_state": goal_state})
    block_err, angle_err, _ = env.core.errors(env.core.goal_state, env.state())
    return obs, infos, block_err < 20 and angle_err < np.pi / 9
