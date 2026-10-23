"""Compare exact simulator replay with reconstruction from the compact model input.

    python -m vmas_pusht.snapshot_audit --out data/novelty/snapshot.json

This is a deterministic physics diagnostic, not a planning benchmark. The legacy
comparison explicitly reproduces the old equal bar/stem velocity reconstruction.
The corrected compact reconstruction still cannot recover soft-joint residuals.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess

import torch
from scipy.stats import rankdata

from vmas_pusht.boundary import place_contact
from vmas_pusht.scenario import make, set_state, state, wrap
from vmas_pusht.snapshot import capture, restore


def physical_difference(reference, other):
    """Coordinate-wise state differences, with rotation differences wrapped."""
    return torch.cat([
        torch.cat([a.state.pos - b.state.pos, a.state.vel - b.state.vel,
                   wrap(a.state.rot - b.state.rot), a.state.ang_vel - b.state.ang_vel], 1)
        for a, b in zip(reference.world.entities, other.world.entities)
    ], 1)


def errors(reference, other):
    pose, rpose = other.scenario.t_pose(), reference.scenario.t_pose()
    return {
        "physical_state_max_abs": float(physical_difference(reference, other).abs().max()),
        "bar_position_error_mean": float((pose[:, :2] - rpose[:, :2]).norm(dim=1).mean()),
        "bar_position_error_max": float((pose[:, :2] - rpose[:, :2]).norm(dim=1).max()),
        "bar_angle_error_max_rad": float(wrap(pose[:, 2] - rpose[:, 2]).abs().max()),
    }


def old_set_state(env, compact, goal):
    set_state(env, compact, goal)
    # The implementation before codex/novelty assigned the bar velocity to both.
    env.scenario.stem.set_vel(compact[:, 11:13].clone(), batch_index=None)


def ranking_diagnostic(true_cost, approximate_cost):
    rows = torch.arange(len(true_cost))
    selected = approximate_cost.argmin(1)
    regret = true_cost[rows, selected] - true_cost.min(1).values
    correlations = []
    for x, y in zip(true_cost.tolist(), approximate_cost.tolist()):
        a, b = torch.tensor(rankdata(x)), torch.tensor(rankdata(y))
        a, b = a - a.mean(), b - b.mean()
        denom = a.norm() * b.norm()
        correlations.append(float((a * b).sum() / denom) if denom > 0 else None)
    return {
        "cost_mae": float((true_cost - approximate_cost).abs().mean()),
        "selected_true_regret_mean": float(regret.mean()),
        "selected_true_regret_max": float(regret.max()),
        "best_candidate_disagreement_fraction": float((selected != true_cost.argmin(1)).float().mean()),
        "spearman_by_context": correlations,
        "cost_matrix": approximate_cost.tolist(),
    }


@torch.no_grad()
def run(contexts=8, warmup=25, horizon=25, candidates=16, seed=0):
    if min(contexts, warmup, horizon, candidates) < 1:
        raise ValueError("contexts, warmup, horizon, and candidates must be positive")
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(seed)
    env = make(num_envs=contexts, seed=seed)
    env.reset()
    angles = torch.arange(contexts) * (2 * math.pi / contexts)
    direction = place_contact(env, angles=angles, layout="wide")
    # Physically evolve asymmetric contact, instead of inventing stem residuals.
    warm_actions = [0.65 * direction, direction]
    for _ in range(warmup):
        env.step([a.clone() for a in warm_actions])
    saved = capture(env)
    compact, goal = state(env), env.scenario.goal_pose.clone()
    copies = {name: make(num_envs=contexts, seed=seed + 1)
              for name in ("exact", "compact_rigid", "legacy_compact")}
    for copy in copies.values():
        copy.reset()
    restore(copies["exact"], saved)
    set_state(copies["compact_rigid"], compact, goal)
    old_set_state(copies["legacy_compact"], compact, goal)
    replay = {name: {"0": errors(env, copy)} for name, copy in copies.items()}
    suffix = torch.rand(horizon, contexts, 2, 2, generator=gen) * 2 - 1
    suffix /= suffix.norm(dim=-1, keepdim=True).clamp(min=1)
    max_exact_reward_error, exact_done_agreement = 0.0, True
    checkpoints = {1, 5, horizon}
    for step, act in enumerate(suffix, 1):
        reference = env.step([act[:, 0].clone(), act[:, 1].clone()])
        for name, copy in copies.items():
            result = copy.step([act[:, 0].clone(), act[:, 1].clone()])
            if name == "exact":
                max_exact_reward_error = max(max_exact_reward_error, max(
                    float((a - b).abs().max()) for a, b in zip(reference[1], result[1])))
                exact_done_agreement &= bool(torch.equal(reference[2], result[2]))
            if step in checkpoints:
                replay[name][str(step)] = errors(env, copy)

    # Same predetermined candidate bank, different initialization only.
    # Constant unit-norm forces keep this audit cheap and easy to reproduce.
    bank = torch.rand(contexts, candidates, 2, 2, generator=gen) * 2 - 1
    bank /= bank.norm(dim=-1, keepdim=True).clamp(min=1)
    actions = bank.reshape(contexts * candidates, 2, 2)
    rows = torch.arange(contexts).repeat_interleave(candidates)
    sims = {name: make(num_envs=contexts * candidates, seed=seed + 2)
            for name in copies}
    for sim in sims.values():
        sim.reset()
    restore(sims["exact"], saved, indices=rows)
    set_state(sims["compact_rigid"], compact[rows], goal[rows])
    old_set_state(sims["legacy_compact"], compact[rows], goal[rows])
    for _ in range(horizon):
        for sim in sims.values():
            sim.step([actions[:, 0].clone(), actions[:, 1].clone()])
    costs = {name: sim.scenario.pose_err().reshape(contexts, candidates) for name, sim in sims.items()}
    if not all(torch.isfinite(cost).all() for cost in costs.values()):
        raise RuntimeError("non-finite simulator outcomes in snapshot audit")
    out = {
        "config": dict(contexts=contexts, warmup=warmup, horizon=horizon, candidates=candidates, seed=seed,
                       device="cpu", substeps=5, friction_ratio=1.5, layout="wide"),
        "versions": {key: importlib.metadata.version(key) for key in ("torch", "vmas", "numpy", "scipy")},
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), *(Path(__file__).with_name(name)
                              for name in ("snapshot.py", "scenario.py", "boundary.py")))},
        "replay": replay,
        "exact_reward_max_abs_error": max_exact_reward_error,
        "exact_done_agreement": exact_done_agreement,
        "true_cost_matrix": costs["exact"].tolist(),
        "constant_action_bank": bank.tolist(),
        "ranking": {name: ranking_diagnostic(costs["exact"], cost) for name, cost in costs.items() if name != "exact"},
        "limitations": ["Fixed noiseless PushT configuration; no random-number-state replay.",
                        "Finite contact contexts and candidate bank; no global optimality claim.",
                        "Compact state is a model observation, not a complete physical state."],
    }
    if (max_exact_reward_error != 0 or not exact_done_agreement
            or any(record["physical_state_max_abs"] != 0 for record in replay["exact"].values())):
        raise AssertionError("Exact CPU snapshot replay must reproduce state, reward, and done bit for bit")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contexts", type=int, default=8)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/novelty/snapshot.json"))
    args = p.parse_args()
    out = run(**{k: v for k, v in vars(args).items() if k != "out"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"exact_reward_max_abs_error": out["exact_reward_max_abs_error"],
                      "exact_done_agreement": out["exact_done_agreement"],
                      "replay": out["replay"],
                      "ranking": {n: {k: v for k, v in d.items() if k != "cost_matrix"}
                                  for n, d in out["ranking"].items()}}, indent=2))
    print("->", args.out)


if __name__ == "__main__":
    main()
