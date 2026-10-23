"""True cost vs readout cost vs latent cost along the scripted team's eval episodes (E0 cost diagnostics).

The scripted team (policies.CoopPolicy, noise-free) plays the eval episodes (plan.py protocol and env, budget 150).
Every step keeps both agents' views and the true state. For each model and agent view, against that agent's goal
image (partner hidden, as plan.py):

    true      pose_cost(true pose, true goal pose)                      readout.pose_cost on the simulator state
    readout   pose_cost(probe(z_t), probe(z_goal))                      the model's probe (readout/cost_probe.pt), as cost=probe
    latent    ||z_t - z_goal||^2                                        the planner's default cost

where a pose is [this agent xy, T xy, T angle] (agent 1's goal spot is its start). Per episode where the T must move
and both agents reach it:
    spearman    rank correlation of the cost with the true cost over the episode
    approach    relative change of the cost from t=0 to the first joint push (< 0: rewards walking to the T)
    window25    pairs (t, t+25) in the approach, the planner's horizon: share where the cost drops
    push        relative change from the first joint push to the episode's end
and the readout's median error along the episodes (self px, T px, T deg). Medians over episodes.

    python scripts/cost_audit.py                          # 3 seeds x 50 episodes, one GPU
    python scripts/cost_audit.py --seeds 0 --episodes 4 --device cpu --procs 2
-> $STABLEWM_HOME/cost_audit/{summary.json, traces.npz}
"""
import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data"))

import numpy as np
import torch

from readout import load_probe, pose_cost, state_pose

ENV = dict(n_agents=2, others="distinct", agent_collisions=True, agent_force=1e5, solver_iterations=50,
           jpeg_quality=95, success="pusht")  # config/plan.yaml
MODELS = ("lewm/pusht", "lewm_coop2_ftfull", "lewm_coop2_joint")
COSTS = ("true", "readout", "latent")


def rollout(job):
    """One scripted episode -> views (T+1, 2, H, W, 3), goal images (2, H, W, 3), global states (T+1, 11), contacts."""
    seed, e, state, goal_state, budget = job
    from episodes import reset
    from policies import CoopPolicy
    from world import MultiPushT
    env = MultiPushT(max_episode_steps=budget, **ENV)
    obs, infos, placed = reset(env, seed, e, state, goal_state)
    ags = env.possible_agents
    team = CoopPolicy(np.random.default_rng(e))
    team.reset(env.core, 0)
    views, states, contact, success = [np.stack([obs[ag]["pixels"] for ag in ags])], [env.state()], [[False, False]], False
    while env.agents:
        obs, _, term, _, infos_t = env.step({ag: team(env.core, i) for i, ag in enumerate(ags)})
        views.append(np.stack([obs[ag]["pixels"] for ag in ags]))
        states.append(env.state())
        contact.append([infos_t[ag]["block_contact"] for ag in ags])
        success = term[ags[0]]
    return dict(seed=seed, ep=e, must_move=not placed, success=bool(success), views=np.stack(views),
                goals=np.stack([infos[ag]["goal"] for ag in ags]), states=np.stack(states),
                goal_state=env.core.goal_state.copy(), contact=np.array(contact))


def rel(c, t0, t1):
    return float((c[t1] - c[t0]) / max(c[t0], 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS), help="checkpoint folders; readout where readout/cost_probe.pt exists")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=50, help="per seed")
    p.add_argument("--budget", type=int, default=150)
    p.add_argument("--procs", type=int, default=4, help="simulator processes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    import stable_worldmodel as swm
    from episodes import eval_episodes
    from fit_readout import encode, metrics
    home = Path(os.environ["STABLEWM_HOME"])
    out = home / "cost_audit"
    out.mkdir(parents=True, exist_ok=True)

    models, probes = {}, {}
    for name in args.models:
        models[name] = swm.wm.utils.load_pretrained(name).to(args.device).eval()
        path = home / "checkpoints" / name / "readout" / "cost_probe.pt"
        if path.exists():
            probes[name] = load_probe(path, args.device)[0]
    print("readout for", list(probes), flush=True)

    jobs = [(s, e, st, g, args.budget) for s in args.seeds for e, (st, g) in enumerate(zip(*eval_episodes(s))) if e < args.episodes]
    traces, meta, err = {}, [], {n: [] for n in probes}
    with get_context("spawn").Pool(args.procs) as pool:
        for r in pool.imap(rollout, jobs):
            key = f"s{r['seed']}_e{r['ep']:02d}"
            both = r["contact"].all(1)
            meta.append(dict(key=key, seed=r["seed"], ep=r["ep"], must_move=r["must_move"], success=r["success"],
                             steps=len(r["states"]) - 1, first_joint_push=int(np.argmax(both)) if both.any() else -1))
            true_pose = np.stack([state_pose(r["states"], i) for i in range(2)], 1)  # (T+1, 2 agents, 6)
            goal_pose = np.stack([state_pose(r["goal_state"], i) for i in range(2)])  # (2 agents, 6)
            traces[f"{key}/true"] = pose_cost(true_pose, goal_pose)                  # (T+1, 2 agents)
            for n, m in models.items():
                zv = encode(m, r["views"].reshape(-1, *r["views"].shape[2:]), args.device).reshape(*r["views"].shape[:2], -1)
                zg = encode(m, r["goals"], args.device)                              # (2 agents, D)
                traces[f"{key}/{n}/latent"] = (zv - zg).pow(2).sum(-1).numpy()
                if n in probes:
                    with torch.no_grad():
                        pv, pg = probes[n](zv.to(args.device)).cpu(), probes[n](zg.to(args.device)).cpu()
                    traces[f"{key}/{n}/readout"] = pose_cost(pv, pg).numpy()
                    err[n].append((pv.reshape(-1, 6).numpy(), true_pose.reshape(-1, 6)))
            print(f"{key} must_move={r['must_move']} success={r['success']} steps={meta[-1]['steps']} "
                  f"first joint push {meta[-1]['first_joint_push']}", flush=True)
    np.savez_compressed(out / "traces.npz", **traces)

    mm = [m for m in meta if m["must_move"] and m["first_joint_push"] > 0]
    summary = {"episodes": meta, "n_must_move": sum(m["must_move"] for m in meta), "n_with_joint_push": len(mm),
               "scripted_success_must_move": sum(m["success"] for m in meta if m["must_move"])}
    rk = lambda x: np.argsort(np.argsort(x))
    for n in ["true"] + list(models):
        for cost in (("true",) if n == "true" else ("readout", "latent") if n in probes else ("latent",)):
            for agent in (0, 1):
                rows = dict(spearman=[], approach=[], window25=[], push=[])
                for m in mm:
                    tc = traces[f"{m['key']}/true"][:, agent]
                    c = tc if n == "true" else traces[f"{m['key']}/{n}/{cost}"][:, agent]
                    tp = m["first_joint_push"]
                    rows["spearman"].append(float(np.corrcoef(rk(c), rk(tc))[0, 1]))
                    rows["approach"].append(rel(c, 0, tp))
                    rows["window25"] += [float(c[t + 25] < c[t]) for t in range(0, tp - 25 + 1)]
                    rows["push"].append(rel(c, tp, len(c) - 1))
                name = f"true | agent {agent} view" if n == "true" else f"{n} | {cost} | agent {agent} view"
                summary[name] = {k: (float(np.mean(v)) if k == "window25" else float(np.nanmedian(v))) if v else None
                                 for k, v in rows.items()}
                summary[name]["n_windows25"] = len(rows["window25"])
                print(name, summary[name], flush=True)
    for n, pairs in err.items():
        P, T = (np.concatenate(x) for x in zip(*pairs))
        summary[f"{n} | readout error along the episodes"] = dict(zip(("r", "median_err"), metrics(P, T)))
        print(n, "readout error", summary[f"{n} | readout error along the episodes"]["median_err"], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
