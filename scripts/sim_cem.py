"""CEM planning with the true simulator as its model: takes model error out of the planner.

Same search as plan.py with planning=centralized (horizon 5 x 5 env steps, replan every 25 steps, CEM 300
samples x 30 iterations, top 30, z-scored actions, warm start from the shifted plan), but every candidate is
rolled out in a copy of the simulator. The cost at the end of the horizon is either

    --cost true          the success test's own quantities: |[agent 0 xy, T xy] - goal| + T angle (20 deg ~ 20 px)
    --cost <checkpoint>  the latent cost ||z - z_goal||^2: agent 0's real view, encoded by that model

So with --cost <checkpoint> the dynamics are perfect and only the cost can mislead; with --cost true neither
can, and what fails is the search. Runs on CPU (true) or CPU + one GPU (latent), one process per episode.

    python scripts/sim_cem.py --cost true --procs 16                  # ~15 min on 16 cores
    python scripts/sim_cem.py --cost lewm_coop2_ftfull --procs 8
-> $STABLEWM_HOME/sim_cem/<cost>_s<seed>/ep<e>.json, summary.json
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data"))

import numpy as np

from episodes import eval_episodes, reset
from utils import pusht_stats

ENV = dict(n_agents=2, others="distinct", agent_force=1e5, solver_iterations=50, jpeg_quality=95)
BUDGET, FRAMESKIP = 150, 5
CEM = dict(H=5, samples=300, iters=30, elites=30, replan=5)
ANGLE_W = 20.0 / np.radians(20)


class Snapshot:
    """Save / restore the bodies of a PushTN core, to roll out candidates from the same state."""

    def __init__(self, core):
        self.core, self.bodies = core, [*core.agents, *core.drives, core.block]

    def load(self, real):
        for b, r in zip(self.bodies, [*real.agents, *real.drives, real.block]):
            b.position, b.velocity, b.angle, b.angular_velocity = r.position, r.velocity, r.angle, r.angular_velocity

    def save(self):
        return [(b.position, b.velocity, b.angle, b.angular_velocity) for b in self.bodies]

    def restore(self, snap):
        for b, (p, v, a, w) in zip(self.bodies, snap):
            b.position, b.velocity, b.angle, b.angular_velocity = p, v, a, w


def true_cost(core, goal):
    s, N = core._get_obs(), core.n_agents
    pos = np.linalg.norm(np.concatenate([goal[:2] - s[:2], goal[2 * N:2 * N + 2] - s[2 * N:2 * N + 2]]))
    ang = abs(goal[2 * N + 2] - s[2 * N + 2])
    return pos + ANGLE_W * min(ang, 2 * np.pi - ang)


def latent_cost(checkpoint, goal_img):
    """-> f(list of agent 0 views) -> ||z - z_goal||^2, with the model's encoder + projector."""
    import stable_worldmodel as swm
    import torch
    model = swm.wm.utils.load_pretrained(checkpoint).to("cuda").eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)  # ImageNet, as in training
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(px):
        x = (torch.from_numpy(np.stack(px)).cuda().permute(0, 3, 1, 2).float().div(255) - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.projector(model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).float()

    z_goal = encode([goal_img])
    return lambda views: (encode(views) - z_goal).pow(2).sum(-1).cpu().numpy()


def plan(sim, real, goal, mean, rng, mu, sd, score):
    """One CEM solve over both agents' actions; mean: (steps, 2 agents, 2) warm start in z-space."""
    sim.load(real)
    snap = sim.save()
    std = np.ones_like(mean)
    lo, hi = (-1 - mu) / sd, (1 - mu) / sd  # the env's action bounds in z-space
    for _ in range(CEM["iters"]):
        cand = np.clip(mean[None] + std[None] * rng.standard_normal((CEM["samples"], *mean.shape)), lo, hi)
        cand[0] = mean
        costs, views = np.empty(len(cand)), []
        for k, z in enumerate(cand):
            sim.restore(snap)
            for a in z:
                sim.core.simulate((a * sd + mu).astype(np.float32))
            if score is None:
                costs[k] = true_cost(sim.core, goal)
            else:
                views.append(sim.core.render_view(0))
        if score is not None:
            costs = score(views)
        top = np.argsort(costs)[: CEM["elites"]]
        mean, std = cand[top].mean(0), np.maximum(cand[top].std(0), 0.05)
    return mean


def run_episode(args):
    cost, seed, e, state, goal_state, out_dir = args
    from world import MultiPushT
    out = out_dir / f"ep{e:02d}.json"
    if out.exists():
        return json.loads(out.read_text())
    mu, sd = pusht_stats("action")
    rng = np.random.default_rng([seed, e])
    real, sim_env = (MultiPushT(max_episode_steps=10 * BUDGET, **ENV) for _ in range(2))
    reset(real, seed, e, state, goal_state)
    reset(sim_env, seed, e, state, goal_state)
    sim, goal = Snapshot(sim_env.core), real.core.goal_state
    score = None if cost == "true" else latent_cost(cost, real.core.goals[0])
    mean = np.zeros((CEM["H"] * FRAMESKIP, 2, 2))
    t, t0, success_step, both = 0, time.time(), -1, 0
    while t < BUDGET and success_step < 0:
        mean = plan(sim, real.core, goal, mean, rng, mu, sd, score)
        n_exec = CEM["replan"] * FRAMESKIP
        for a in mean[:n_exec]:
            _, _, term, _, info = real.step({ag: (a[i] * sd + mu).astype(np.float32) for i, ag in enumerate(real.possible_agents)})
            t += 1
            both += info["agent_0"]["block_contact"] and info["agent_1"]["block_contact"]
            if term["agent_0"]:
                success_step = t
            if success_step > 0 or t >= BUDGET:
                break
        mean = np.concatenate([mean[n_exec:], np.zeros((n_exec, 2, 2))])
    block_err, angle_err, _ = real.core.errors(goal, real.state())
    res = dict(episode=e, success=success_step > 0, success_step=success_step, T_pos_err=block_err,
               T_angle_err=angle_err, both_touch_T_frac=both / t, wall_s=round(time.time() - t0))
    out.write_text(json.dumps(res))
    print(f"ep{e:02d} success={res['success']} T err {block_err:.0f} px, both touch the T {both / t:.0%} ({res['wall_s']} s)", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cost", default="true", help="true, or a checkpoint folder for the latent cost")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=0, help="first n episodes where the T must move (0 = all)")
    p.add_argument("--procs", type=int, default=8)
    args = p.parse_args()
    from world import MultiPushT
    states, goals = eval_episodes(args.seed)
    must_move = [e for e in range(len(states)) if not reset(MultiPushT(**ENV), args.seed, e, states[e], goals[e])[2]]
    must_move = must_move[: args.episodes or None]
    out_dir = Path(os.environ["STABLEWM_HOME"], "sim_cem", f"{args.cost.replace('/', '_')}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"seed {args.seed}: {len(must_move)} episodes where the T must move, cost={args.cost}", flush=True)
    jobs = [(args.cost, args.seed, e, states[e], goals[e], out_dir) for e in must_move]
    with get_context("spawn").Pool(args.procs) as pool:
        res = list(pool.imap_unordered(run_episode, jobs))
    summary = {"cost": args.cost, "seed": args.seed, "solved": f"{sum(r['success'] for r in res)} / {len(res)}",
               "median_T_pos_err": float(np.median([r["T_pos_err"] for r in res])),
               "both_touch_T_frac": float(np.mean([r["both_touch_T_frac"] for r in res]))}
    print(json.dumps(summary, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
