"""Export what planning and training need from the official Push-T dataset (44 GB) into assets/ (a few KB).

    assets/eval_episodes.npz   start / goal states of the eval episodes, per seed (same protocol as LeWM's eval.py)
    assets/pusht_stats.json    mean / std of action, proprio and state (the z-scores every LeWM model was trained with)

Only needed to regenerate the shipped files: python scripts/download.py --pusht-data, then
    python scripts/export_eval_assets.py --seeds 0 1 2 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
from stable_worldmodel.world.world import _extract_init_goal

ROOT = Path(__file__).resolve().parents[1]
NUM_EVAL, GOAL_OFFSET = 50, 25  # LeWM's eval protocol: 50 episodes, goal 25 steps after the start


def sample_eval_starts(dataset, seed):
    """LeWM eval.py: seeded (episode, start step) pairs with the goal frame inside the episode."""
    episode_idx, step_idx = dataset.get_col_data("episode_idx"), dataset.get_col_data("step_idx")
    max_step = np.full(episode_idx.max() + 1, -1)
    np.maximum.at(max_step, episode_idx, step_idx)
    valid = np.nonzero(step_idx <= max_step[episode_idx] - GOAL_OFFSET)[0]
    rows = np.sort(valid[np.random.default_rng(seed).choice(len(valid) - 1, size=NUM_EVAL, replace=False)])
    data = dataset.get_row_data(rows)
    return data["episode_idx"].tolist(), data["step_idx"].tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 42])
    args = p.parse_args()
    ds = swm.data.HDF5Dataset("pusht_expert_train", keys_to_cache=["action", "proprio", "state"],
                              cache_dir=swm.data.utils.get_cache_dir())
    out = {}
    for seed in args.seeds:
        episodes, starts = sample_eval_starts(ds, seed)
        init, goal, _ = _extract_init_goal(ds, episodes, starts, GOAL_OFFSET)
        out[f"state_{seed}"] = np.asarray(init["state"], np.float64)
        out[f"goal_state_{seed}"] = np.asarray(goal["goal_state"], np.float64)
    (ROOT / "assets").mkdir(exist_ok=True)
    np.savez(ROOT / "assets/eval_episodes.npz", **out)

    stats = {}
    for col in ("action", "proprio", "state"):
        x = ds.get_col_data(col)
        x = x[~np.isnan(x).any(1)].astype(np.float64)
        stats[col] = {"mean": x.mean(0).tolist(), "std": x.std(0, ddof=1).tolist()}
    (ROOT / "assets/pusht_stats.json").write_text(json.dumps(stats, indent=1))
    print("wrote assets/eval_episodes.npz (seeds %s) and assets/pusht_stats.json" % args.seeds)


if __name__ == "__main__":
    main()
