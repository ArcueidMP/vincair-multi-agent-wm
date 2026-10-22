"""Data aggregation: append planner-visited episodes (plan.py --record) to the scripted training set.

    python -m vmas_pusht.merge_data --base data/vmas_pusht/train.pt --add data/vmas_pusht/planner_visits_full.pt --out data/vmas_pusht/train_agg.pt

Recorded episodes longer than the base ones are cut into base-length pieces (overlapping by one step, like the base
episodes' own windows) so that train.windows can treat every episode alike.
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/vmas_pusht/train.pt")
    p.add_argument("--add", nargs="+", required=True)
    p.add_argument("--out", default="data/vmas_pusht/train_agg.pt")
    args = p.parse_args()
    base = torch.load(args.base)
    T1 = base["state"].shape[1]
    parts = {k: [v] for k, v in base.items()}
    for f in args.add:
        d = torch.load(f)
        L = d["state"].shape[1]
        starts = list(range(0, L - T1 + 1, T1 - 1)) or [0]
        for s0 in starts:
            parts["state"].append(d["state"][:, s0:s0 + T1]); parts["contact"].append(d["contact"][:, s0:s0 + T1])
            parts["success"].append(d["success"][:, s0:s0 + T1]); parts["action"].append(d["action"][:, s0:s0 + T1 - 1])
            parts["goal"].append(d["goal"])
        print(f, "->", len(starts), "pieces of", T1, "steps")
    out = {k: torch.cat(v) for k, v in parts.items()}
    torch.save(out, args.out)
    print({k: tuple(v.shape) for k, v in out.items()}, "->", args.out)


if __name__ == "__main__":
    main()
