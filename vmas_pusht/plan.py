"""Plan on VMAS Push-T: CEM over both agents' actions, with the learned dynamics or the true simulator, scored by the
latent cost or the true cost, toward the eval goal, oracle subgoals or the roles coordinator.

    python -m vmas_pusht.plan --method latent --ckpt data/vmas_pusht/wm_full.pt --out data/vmas_pusht/eval/full.json
    python -m vmas_pusht.plan --method true                                 # true simulator + true cost (search ceiling)
    python -m vmas_pusht.plan --method sim_latent --ckpt ... --subgoal scripted --lookahead 15 --horizon 3
                                                                            # true simulator + this model's latent cost + oracle subgoals
    python -m vmas_pusht.plan --method true --subgoal roles --roles_step 0.15   # roles coordinator (subgoal.RoleSchedule)
    python -m vmas_pusht.plan --method scripted / random                     # references

Protocol: `episodes` start/goal pairs from a fixed seed (VMAS reset), `budget` env steps, horizon in model steps of 5 env
steps (LeWM's planner), replan every `replan` env steps (default: the whole horizon, open loop), CEM 300 samples x 30
iterations, top 30. Warm start: zero (default) or the previous plan shifted (--warm shift); noise: white (default) or
colored 1/f^beta along time (iCEM, --noise colored).
Costs: latent = squared distance between the predicted tokens and the target state's tokens (--cost_tokens T: the T token
only, agents free; all: every token); probe = the T pose decoded from the T token by an MLP probe (no agent terms);
decoded = subgoal.state_dist on the planar state decoded from every token by probes (--probe_on encoded: probes fit on
encoder outputs; predicted: on the model's own k-step predictions paired with the true future state); head = the same
planar state read by the model's trained state head; true = the T's pose error (eval goal) or the state distance of
subgoal.py (subgoals).
Coupling (--coupling): joint = one CEM over both agents' actions (centralized); none = each agent plans its own actions
assuming the partner repeats its last action; exchange = one round of plan exchange after that (each re-plans given
the other's announced plan); seq = agent 0 plans first, agent 1 plans given agent 0's plan (SeqWM-style ordering).
An episode counts as solved at its first success. --record saves the visited states and actions in collect.py's format.
--solo i: only agent i is planned, the other stands still (single-agent oracle for the cooperation gap; joint coupling only).
--diag path (latent only): at every replan, score the CEM candidates on the true simulator too and save per episode the
Spearman rank correlation between the model cost and the true cost (random candidates at iteration 0, the final
population and its elites), the chosen plan's model vs true cost (exploitation gap), its regret against the best true
candidate, the predicted vs true T displacement, and the contact class at the replan.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from vmas_pusht.model import EntityJEPA
from vmas_pusht.policy import ScriptedTeam
from vmas_pusht.probe import fit_mlp_probe
from vmas_pusht.scenario import make, set_state, state, wrap
from vmas_pusht.subgoal import RoleSchedule, Schedule, goal_state, scripted_rollout, state_dist

FS = 5
ANGLE_W = 4 * 0.05 / math.radians(20)


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    c = ck["config"]
    m = EntityJEPA(history=c["history"], act_dim=2 * c["frameskip"], attn=c["attn"], head=bool(c.get("head", 0))).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["mean"].to(dev), ck["std"].to(dev)


def train_frames(train_path, dev, n_ep=400, every=4):
    tr = torch.load(train_path)
    g = torch.Generator().manual_seed(5)
    return tr, tr["state"][torch.randint(len(tr["goal"]), (n_ep,), generator=g)].reshape(-1, tr["state"].shape[-1])[::every].to(dev)


@torch.no_grad()
def predicted_tokens(model, norm, tr, dev, K, n=30000):
    """(tokens, true states): the model's own 1..K-block predictions on training windows, paired with the true future
    state, so that a probe fit on them reads predicted latents without the encoder/predictor mismatch."""
    from vmas_pusht.train import windows
    H = model.history
    d = {"state": norm(tr["state"].to(dev)), "action": tr["action"].to(dev)}
    s, a = windows(d, H + K - 1, FS, torch.arange(len(tr["goal"])), torch.Generator().manual_seed(9), n // K, dev)
    z = model.encode(s)
    roll = model.rollout(z[:, :H], a[:, :H - 1], a[:, H - 1:])                    # (n/K, K, n_tok, latent)
    raw = s[:, H:] * (norm.std) + norm.mean                                        # back to raw units
    return roll.reshape(-1, *roll.shape[2:]), raw.reshape(-1, raw.shape[-1])


class Norm:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, s):
        return (s - self.mean) / self.std


def pose_targets(S):
    return torch.cat([S[:, 8:10], torch.sin(S[:, 10:11]), torch.cos(S[:, 10:11])], 1)


def pose_probe(model, norm, train_path, dev, on="encoded", K=3):
    """MLP from the T token to [x, y, sin, cos]."""
    tr, S = train_frames(train_path, dev)
    with torch.no_grad():
        Z = torch.cat([model.encode(norm(S[lo:lo + 8192])) for lo in range(0, len(S), 8192)])
    if on == "predicted":
        Z, S = predicted_tokens(model, norm, tr, dev, K)
    f, stat = fit_mlp_probe(Z[:, -1], pose_targets(S))
    print("pose probe", on, stat, flush=True)
    return f


def state_probe(model, norm, train_path, dev, on="encoded", K=3):
    """Probes from the tokens to the planar state: agent tokens -> [x, y] (one probe shared by both agents), T token ->
    [x, y, sin, cos]; flat models decode all 8 targets from their one token. Returns f: (n, n_tok, latent) -> (n, 8)."""
    tr, S = train_frames(train_path, dev)
    with torch.no_grad():
        Z = torch.cat([model.encode(norm(S[lo:lo + 8192])) for lo in range(0, len(S), 8192)])
    if on == "predicted":
        Z, S = predicted_tokens(model, norm, tr, dev, K)
    tpose = pose_targets(S)
    if model.attn == "flat":
        f, stat = fit_mlp_probe(Z[:, 0], torch.cat([S[:, 0:2], S[:, 4:6], tpose], 1))
        print("state probe (flat)", on, stat, flush=True)
        return lambda z: f(z[:, 0])
    fa, sa = fit_mlp_probe(torch.cat([Z[:, 0], Z[:, 1]]), torch.cat([S[:, 0:2], S[:, 4:6]]))
    ft, st = fit_mlp_probe(Z[:, -1], tpose)
    print("state probe", on, ": agents", sa, "T", st, flush=True)
    return lambda z: torch.cat([fa(z[:, 0]), fa(z[:, 1]), ft(z[:, -1])], 1)


def head_reader(model, mean, std):
    """The model's state head as a planar-state reader: (n, n_tok, latent) -> (n, 8) [a0 xy, a1 xy, T xy, sin, cos], raw units."""
    N = model.N

    def f(z):
        F = model.decode(z)
        a = [F[:, 4 * i:4 * i + 2] * std[4 * i:4 * i + 2] + mean[4 * i:4 * i + 2] for i in range(N)]
        t = F[:, 4 * N:4 * N + 2] * std[8:10] + mean[8:10]
        return torch.cat([*a, t, F[:, 4 * N + 2:4 * N + 4]], 1)
    return f


def planar_cost(P, target):
    """P (n, 8) decoded planar state, target state (n, D) -> subgoal.state_dist on the decoded state."""
    pos = (P[:, 0:2] - target[:, 0:2]).norm(dim=1) + (P[:, 2:4] - target[:, 4:6]).norm(dim=1) + (P[:, 4:6] - target[:, 8:10]).norm(dim=1)
    return pos + ANGLE_W * wrap(torch.atan2(P[:, 6], P[:, 7]) - target[:, 10]).abs()


def probe_cost(f, z_T, target_pose):
    """z_T (n, latent), target_pose (n, 3) -> pose error of the decoded T to the target."""
    P = f(z_T)
    return (P[:, :2] - target_pose[:, :2]).norm(dim=1) + ANGLE_W * wrap(torch.atan2(P[:, 2], P[:, 3]) - target_pose[:, 2]).abs()


def colored_noise(shape, beta, gen, dev):
    """Gaussian noise with a 1 / f^beta power spectrum along dim 2 (time), unit variance per entry (iCEM, Pinneri 2020)."""
    T = shape[2]
    w = torch.randn(*shape, device=dev, generator=gen)
    if beta == 0:
        return w
    W = torch.fft.rfft(w, dim=2)
    f = torch.arange(W.shape[2], device=dev).clamp(min=1).float()
    W = W / f.pow(beta / 2).view(1, 1, -1, *([1] * (len(shape) - 3)))
    c = torch.fft.irfft(W, n=T, dim=2)
    return c / c.std(dim=2, keepdim=True).clamp(min=1e-6)


def spearman_rows(a, b):
    """(E, S) x (E, S) -> (E,) Spearman rank correlation along dim 1."""
    ra = a.argsort(1).argsort(1).float(); rb = b.argsort(1).argsort(1).float()
    ra = ra - ra.mean(1, keepdim=True); rb = rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp(min=1e-8)


@torch.no_grad()
def cem(cost_fn, E, dev, mean_init, samples, iters, elites, gen, free=(0, 1), beta=0.0, probe_cb=None, keep=0, exec_best=False):
    """actions (E, S, T, 2 agents, 2) in [-1, 1]; cost_fn(cands (E*S, T, 2, 2)) -> (E*S,). Returns plan (E, T, 2, 2).
    Only the agents in `free` are sampled; the others keep mean_init (a partner's announced plan or persisted action).
    iCEM options: `keep` elites of one iteration re-enter the next candidate set; `exec_best` returns the lowest-cost elite of
    the last iteration instead of the elite mean."""
    mu, sd = mean_init.clone(), torch.ones_like(mean_init)
    kept, best = None, None
    fixed = torch.ones(mu.shape[2], dtype=torch.bool, device=dev)
    fixed[list(free)] = False
    sd[:, :, fixed] = 0
    for _ in range(iters):
        c = (mu[:, None] + sd[:, None] * colored_noise((E, samples, *mu.shape[1:]), beta, gen, dev)).clamp(-1, 1)
        c[:, 0] = mu
        if kept is not None:
            c[:, 1:1 + kept.shape[1]] = kept
        cost = cost_fn(c.reshape(E * samples, *mu.shape[1:])).reshape(E, samples)
        if probe_cb is not None and _ in (0, iters - 1):
            probe_cb(_, c, cost)
        top = cost.topk(elites, 1, largest=False).indices                # sorted: column 0 is the lowest model cost
        el = c[torch.arange(E, device=dev)[:, None], top]
        if keep:
            kept = el[:, :keep]
        best = el[:, 0]
        mu, sd = el.mean(1), el.std(1).clamp(min=0.05)
        sd[:, :, fixed] = 0
    return best if exec_best else mu


def run(args):
    dev = args.device
    gen = torch.Generator(device=dev).manual_seed(0)
    env = make(num_envs=args.episodes, device=dev, seed=args.seed)
    env.reset()
    E = args.episodes
    goal = env.scenario.goal_pose.clone()
    solved = torch.zeros(E, dtype=torch.bool, device=dev)
    solved_at = torch.full((E,), -1, device=dev)
    err0 = env.scenario.pose_err().clone()
    co, t0 = torch.zeros(E, device=dev), time.time()
    from vmas_pusht.collect import contacts
    H = args.horizon
    PL = H * FS                                                      # plan length in env steps
    replan = args.replan or PL
    assert replan % FS == 0 and replan <= PL, "replan must be a multiple of the block and at most the horizon"
    planning = args.method in ("latent", "true", "sim_latent")
    latent_cost = args.method in ("latent", "sim_latent")
    sched, scripted_solved, scripted_in_budget = None, None, None
    if planning and args.subgoal == "scripted":
        st, sa = scripted_rollout(state(env), goal, steps=args.sub_budget, device=dev, seed=args.seed + 1)
        sched = Schedule(st, sa, lookahead=args.lookahead, window=args.window)
        scripted_solved, scripted_in_budget = int((sa >= 0).sum()), int(((sa >= 0) & (sa <= args.budget)).sum())
        print(f"scripted team solves {scripted_solved}/{E} within {args.sub_budget} steps, {scripted_in_budget} within the budget "
              f"(median {float(sa[sa >= 0].float().median()) if scripted_solved else -1:.0f})", flush=True)
    elif planning and args.subgoal in ("roles", "roles_approach"):
        sched = RoleSchedule(env, goal, step=args.roles_step, step_rad=math.radians(25) * args.roles_step / 0.15,
                             ready=args.roles_ready, patience=args.roles_patience, push=args.subgoal == "roles")
    if latent_cost:
        model, mean, std = load(args.ckpt, dev)
        norm = Norm(mean, std)
        tok = slice(-1, None) if args.cost_tokens == "T" else slice(None)   # T token is last; flat models have one token
        probe = (pose_probe(model, norm, args.train, dev, args.probe_on, H) if args.cost == "probe"
                 else state_probe(model, norm, args.train, dev, args.probe_on, H) if args.cost == "decoded"
                 else head_reader(model, mean, std) if args.cost == "head" else None)
        hist_s, hist_a = [state(env)], []
    if args.method in ("true", "sim_latent"):
        sim = make(num_envs=E * args.samples, device=dev, seed=args.seed + 2)
        sim.reset()
    if args.method == "scripted":
        pol = ScriptedTeam(env, p_solo=0.0, p_random=0.0, noise=0.0)
    diag = None
    if args.diag:
        assert args.method == "latent", "--diag scores a learned model's candidates on the true simulator"
        dsim = make(num_envs=E * args.samples, device=dev, seed=args.seed + 3)
        dsim.reset()
        diag = []
    if args.solo >= 0:
        assert args.coupling == "joint", "--solo plans one agent with a joint CEM"
    plan = torch.zeros(E, PL, 2, 2, device=dev)
    last_act = torch.zeros(E, 2, 2, device=dev)
    rec_s, rec_a, rec_c, rec_w = [state(env)], [], [contacts(env)], [env.scenario.success()]
    for t in range(args.budget):
        if planning and t % replan == 0:
            s = state(env)
            target = sched.update(s) if sched is not None else goal_state(s, goal)      # (E, D)
            if latent_cost:
                zg = model.encode(norm(target))[:, tok]                                   # (E, n_tok, latent)
                tpose = target[:, 8:11]
            if args.method == "latent":
                hs = torch.stack(hist_s[-model.history:], 1)
                zh = model.encode(norm(hs))
                ha = torch.stack(hist_a[-(hs.shape[1] - 1):], 1) if hs.shape[1] > 1 else torch.zeros(E, 0, 2, 2 * FS, device=dev)

                def roll_fn(c, zh=zh, ha=ha):
                    n = c.shape[0]; S = n // E
                    blocks = c.reshape(n, H, FS, 2, 2).permute(0, 1, 3, 2, 4).reshape(n, H, 2, 2 * FS)
                    return model.rollout(zh.repeat_interleave(S, 0), ha.repeat_interleave(S, 0), blocks)[:, -1]

                def cost_fn(c, zg=zg, tpose=tpose, target=target):
                    S = c.shape[0] // E
                    last = roll_fn(c)
                    if args.cost in ("decoded", "head"):
                        return planar_cost(probe(last), target.repeat_interleave(S, 0))
                    if probe is not None:
                        return probe_cost(probe, last[:, -1], tpose.repeat_interleave(S, 0))
                    return (last[:, tok] - zg.repeat_interleave(S, 0)).pow(2).sum((-1, -2))

                if diag is not None:
                    def true_fn(c, s=s, target=target):
                        """true cost of candidates c (E*S, PL, 2, 2) from state s, and the T position they reach."""
                        S = c.shape[0] // E
                        rep = args.samples // S                                        # the sim batch is E * samples
                        c = c.repeat_interleave(rep, 0)
                        set_state(dsim, s.repeat_interleave(args.samples, 0), goal.repeat_interleave(args.samples, 0))
                        for k in range(c.shape[1]):
                            dsim.step([c[:, k, 0], c[:, k, 1]])
                        st = state(dsim)
                        tc = state_dist(st, target.repeat_interleave(args.samples, 0)) if sched is not None else dsim.scenario.pose_err()
                        return tc[::rep], st[::rep, 8:10]
                    drec = {"t": t, "cls": contacts(env).sum(1).tolist()}

                    def probe_cb(it, c, mcost, drec=drec):
                        S = c.shape[1]
                        tcost, _ = true_fn(c.reshape(E * S, *c.shape[2:]))
                        tcost = tcost.reshape(E, S)
                        if it == 0:
                            drec["spearman_iter0"] = spearman_rows(mcost, tcost).tolist()
                        if it == args.iters - 1:                                   # with --iters 1 this is also iteration 0
                            drec["spearman_final"] = spearman_rows(mcost, tcost).tolist()
                            top = mcost.topk(args.elites, 1, largest=False).indices
                            ar = torch.arange(E, device=dev)[:, None]
                            drec["spearman_elites"] = spearman_rows(mcost[ar, top], tcost[ar, top]).tolist()
                            drec["min_true_final"] = tcost.min(1).values.tolist()
                            drec["min_model_final"] = mcost.min(1).values.tolist()
            else:
                def cost_fn(c, s=s, target=target):
                    set_state(sim, s.repeat_interleave(args.samples, 0), goal.repeat_interleave(args.samples, 0))
                    for k in range(c.shape[1]):
                        sim.step([c[:, k, 0], c[:, k, 1]])
                    if latent_cost:
                        z = model.encode(norm(state(sim)))
                        if args.cost in ("decoded", "head"):
                            return planar_cost(probe(z), target.repeat_interleave(args.samples, 0))
                        if probe is not None:
                            return probe_cost(probe, z[:, -1], tpose.repeat_interleave(args.samples, 0))
                        return (z[:, tok] - zg.repeat_interleave(args.samples, 0)).pow(2).sum((-1, -2))
                    if sched is not None:
                        return state_dist(state(sim), target.repeat_interleave(args.samples, 0))
                    return sim.scenario.pose_err()
            if args.warm == "shift" and t > 0:
                warm = torch.cat([plan[:, replan:], torch.zeros(E, replan, 2, 2, device=dev)], 1)
            else:
                warm = torch.zeros(E, PL, 2, 2, device=dev)
            cb = probe_cb if diag is not None else None
            run = lambda init, free, cb=None: cem(cost_fn, E, dev, init, args.samples, args.iters, args.elites, gen, free=free, beta=args.beta, probe_cb=cb,
                                                  keep=args.keep_elites, exec_best=args.exec == "best")
            if args.coupling == "joint":
                plan = run(warm, (0, 1) if args.solo < 0 else (args.solo,), cb)
                if diag is not None:
                    tc, tT = true_fn(plan)
                    drec["model_cost_chosen"] = cost_fn(plan).tolist(); drec["true_cost_chosen"] = tc.tolist()
                    drec["true_dT"] = (tT - s[:, 8:10]).norm(dim=1).tolist()
                    if args.cost in ("decoded", "head"):
                        P = probe(roll_fn(plan)); P0 = probe(zh[:, -1])
                        drec["pred_dT"] = (P[:, 4:6] - P0[:, 4:6]).norm(dim=1).tolist()
                    diag.append(drec)
            else:
                persist = last_act[:, None].expand(E, PL, 2, 2).clone()           # the partner repeats its last action
                plans = [run(persist, (i,)) for i in (0, 1)]                        # round 1: no message
                if args.coupling == "seq":                                          # agent 1 plans given agent 0's plan
                    plans[1] = run(plans[0], (1,))
                elif args.coupling == "exchange":                                   # both re-plan given the other's plan
                    plans = [run(plans[1 - i], (i,)) for i in (0, 1)]
                plan = torch.stack([plans[0][:, :, 0], plans[1][:, :, 1]], 2)
        if planning:
            a = plan[:, t % replan]
            act = [a[:, 0], a[:, 1]]
        elif args.method == "scripted":
            act = pol()
        else:
            act = [torch.rand(E, 2, device=dev, generator=gen) * 2 - 1 for _ in range(2)]
        env.step(act)
        last_act = torch.stack(act, 1)
        if args.method == "latent" and (t + 1) % FS == 0:
            lo = (t + 1 - FS) % replan
            hist_a.append(plan[:, lo:lo + FS].permute(0, 2, 1, 3).reshape(E, 2, 2 * FS))
            hist_s.append(state(env))
        if args.record:
            rec_a.append(last_act); rec_s.append(state(env)); rec_c.append(contacts(env)); rec_w.append(env.scenario.success())
        co += contacts(env).all(1).float() * (~solved)
        ok = env.scenario.success() & ~solved
        solved_at[ok] = t + 1
        solved |= ok
    res = {"method": args.method, "ckpt": args.ckpt, "subgoal": args.subgoal, "lookahead": args.lookahead, "window": args.window,
           "horizon": H, "replan": replan, "warm": args.warm, "beta": args.beta, "roles_step": args.roles_step,
           "cost_tokens": args.cost_tokens, "cost": args.cost, "probe_on": args.probe_on, "coupling": args.coupling,
           "episodes": E, "budget": args.budget, "solved": int(solved.sum()), "success_rate": float(solved.float().mean()),
           "scripted_solved": scripted_solved, "scripted_solved_within_budget": scripted_in_budget,
           "n_push_subgoals_mean": float(sched.n_push.float().mean()) if isinstance(sched, RoleSchedule) else None,
           "pose_err_start": float(err0.mean()), "pose_err_end": float(env.scenario.pose_err()[~solved].mean()) if (~solved).any() else 0.0,
           "co_contact": float((co / args.budget).mean()), "solved_at": solved_at.tolist(), "minutes": round((time.time() - t0) / 60, 1)}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "solved_at"}, "->", out, flush=True)
    if diag is not None:
        T_ = lambda k: torch.tensor([r[k] for r in diag])                        # (replans, E)
        C_ = T_("cls")
        summ = {}
        for k in ("spearman_iter0", "spearman_final", "spearman_elites", "pred_dT", "true_dT"):
            if k in diag[0]:
                v = T_(k)
                summ[k] = {"all": float(v.nanmedian())} | {n: float(v[C_ == i].nanmedian()) for i, n in enumerate(("none", "lone", "both")) if (C_ == i).any()}
        gap = T_("true_cost_chosen") - T_("model_cost_chosen"); regret = T_("true_cost_chosen") - T_("min_true_final")
        summ["gap_true_minus_model_median"] = float(gap.median()); summ["regret_vs_best_true_median"] = float(regret.median())
        summ["regret_vs_best_true_mean"] = float(regret.mean())
        Path(args.diag).parent.mkdir(parents=True, exist_ok=True)
        Path(args.diag).write_text(json.dumps({"summary": summ, "args": vars(args), "replans": diag}, indent=1))
        print("diag", json.dumps(summ, indent=1), "->", args.diag, flush=True)
    if args.record:
        d = dict(state=torch.stack(rec_s, 1).cpu(), action=torch.stack(rec_a, 1).cpu(), goal=goal.cpu(),
                 contact=torch.stack(rec_c, 1).cpu(), success=torch.stack(rec_w, 1).cpu())
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        torch.save(d, args.record)
        print("recorded", {k: tuple(v.shape) for k, v in d.items()}, "->", args.record, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="latent", choices=["latent", "true", "sim_latent", "scripted", "random"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--horizon", type=int, default=5, help="model steps of 5 env steps")
    p.add_argument("--replan", type=int, default=0, help="replan every N env steps (multiple of 5; default: the whole horizon)")
    p.add_argument("--warm", default="zero", choices=["zero", "shift"], help="CEM initial mean: zeros or the previous plan shifted")
    p.add_argument("--beta", type=float, default=0.0, help="colored-noise exponent for CEM samples (0 = white, iCEM uses 2)")
    p.add_argument("--subgoal", default="none", choices=["none", "scripted", "roles", "roles_approach"])
    p.add_argument("--lookahead", type=int, default=25)
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--sub_budget", type=int, default=600, help="scripted replay length for the subgoal schedule")
    p.add_argument("--roles_step", type=float, default=0.15, help="roles: how far the push subgoal moves the T (angle 25 deg x step / 0.15)")
    p.add_argument("--roles_ready", type=float, default=0.05)
    p.add_argument("--roles_patience", type=int, default=2)
    p.add_argument("--cost_tokens", default="T", choices=["T", "all"])
    p.add_argument("--cost", default="latent", choices=["latent", "probe", "decoded", "head"],
                   help="probe: T pose decoded from the T token; decoded: state_dist on the planar state decoded by probes; head: by the model's state head")
    p.add_argument("--probe_on", default="encoded", choices=["encoded", "predicted"], help="fit the probes on encoder outputs or on the model's predictions")
    p.add_argument("--coupling", default="joint", choices=["joint", "none", "exchange", "seq"],
                   help="joint CEM over both agents, or per-agent CEM with no message / one plan exchange / sequential")
    p.add_argument("--train", default="data/vmas_pusht/train.pt", help="training data for the probe")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--elites", type=int, default=30)
    p.add_argument("--keep_elites", type=int, default=0, help="iCEM: elites carried into the next iteration's candidates (0 = off)")
    p.add_argument("--exec", default="mean", choices=["mean", "best"], help="execute the elite mean (CEM) or the best elite (iCEM)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--record", default=None, help="save the visited states / actions (collect.py format) to this .pt")
    p.add_argument("--solo", type=int, default=-1, help="plan only this agent, the other stands still (single-agent oracle)")
    p.add_argument("--diag", default=None, help="latent only: save rankability / exploitation diagnostics per replan to this .json")
    p.add_argument("--out", default="data/vmas_pusht/eval/out.json")
    run(p.parse_args())
