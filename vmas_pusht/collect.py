"""Collect VMAS Push-T episodes with the scripted team -> one .pt file of tensors.

    python -m vmas_pusht.collect --episodes 4096 --batch 1024 --device cuda --out data/vmas_pusht/train.pt
    python -m vmas_pusht.collect --episodes 512 --seed 1 --out data/vmas_pusht/heldout.pt

Saved: state (E, T+1, STATE_DIM), action (E, T, 2, 2), goal (E, 3), contact (E, T+1, 2) bool, success (E, T+1) bool.
Episodes run the full T steps (no reset at success), so every sequence has the same length.
"""
import argparse
import math
import time
from pathlib import Path

import torch

from vmas_pusht.policy import ScriptedTeam, rot
from vmas_pusht.scenario import AGENT_R, BAR, STEM, STEM_OFFSET, make, state


def contacts(env, margin=0.01):
    """(B, n_agents) bool: agent within margin of the T's surface."""
    sc = env.scenario
    pos, r = sc.bar.state.pos, sc.bar.state.rot[:, 0]
    out = []
    for ag in env.world.agents:
        loc = rot(ag.state.pos - pos, -r)
        d = None
        for (cx, cy), (l, w) in (((0.0, 0.0), BAR), ((0.0, STEM_OFFSET), STEM)):
            q = (torch.stack([(loc[:, 0] - cx).abs() - l / 2, (loc[:, 1] - cy).abs() - w / 2], 1)).clamp(min=0).norm(dim=1)
            d = q if d is None else torch.minimum(d, q)
        out.append(d < AGENT_R + margin)
    return torch.stack(out, 1)


def collect(episodes, batch, steps, device, seed, **policy_kwargs):
    S, A, G, C, W = [], [], [], [], []
    for b0 in range(0, episodes, batch):
        n = min(batch, episodes - b0)
        env = make(num_envs=n, device=device, seed=seed * 100003 + b0)
        env.reset()
        pol = ScriptedTeam(env, seed=seed * 100003 + b0, **policy_kwargs)
        s, a, c, w = [state(env)], [], [contacts(env)], [env.scenario.success()]
        for _ in range(steps):
            act = pol()
            env.step(act)
            a.append(torch.stack(act, 1)), s.append(state(env)), c.append(contacts(env)), w.append(env.scenario.success())
        S.append(torch.stack(s, 1).cpu()), A.append(torch.stack(a, 1).cpu()), G.append(env.scenario.goal_pose.cpu().clone())
        C.append(torch.stack(c, 1).cpu()), W.append(torch.stack(w, 1).cpu())
    return dict(state=torch.cat(S), action=torch.cat(A), goal=torch.cat(G), contact=torch.cat(C), success=torch.cat(W))


def report(d):
    c = d["contact"].float()
    moved = (d["state"][:, 1:, 8:10] - d["state"][:, :-1, 8:10]).norm(dim=-1)
    co = c[:, :-1].all(-1)
    return {"episodes": len(d["goal"]), "steps": d["action"].shape[1], "success_ever": float(d["success"].any(1).float().mean()),
            "contact_per_agent": c.mean((0, 1)).tolist(), "co_contact": float(co.float().mean()),
            "T_speed_co": float(moved[co].mean()) if co.any() else 0.0,
            "T_speed_lone": float(moved[(c[:, :-1].sum(-1) == 1)].mean()), "T_speed_none": float(moved[c[:, :-1].sum(-1) == 0].mean())}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p_solo", type=float, default=0.1)
    p.add_argument("--p_random", type=float, default=0.05)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--out", default="data/vmas_pusht/train.pt")
    args = p.parse_args()
    t0 = time.time()
    d = collect(args.episodes, args.batch, args.steps, args.device, args.seed,
                p_solo=args.p_solo, p_random=args.p_random, noise=args.noise)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, out)
    print(report(d), f"{time.time() - t0:.0f}s ->", out, flush=True)
