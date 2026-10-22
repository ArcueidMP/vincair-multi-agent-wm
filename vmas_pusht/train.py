"""Train the entity-token JEPA on VMAS Push-T data (vmas_pusht/collect.py).

    python -m vmas_pusht.train --data data/vmas_pusht/train.pt --attn full --out data/vmas_pusht/wm_full.pt
    python -m vmas_pusht.train --attn full --head 1 --train_k 3 --sigreg per_token --out data/vmas_pusht/wm_full_h3p.pt

Windows: history H frames frameskip env steps apart plus the next k frames; each frame's action block is the next
frameskip actions of each agent (act_dim = 2 x frameskip). Episodes are split 90 / 10 (validation holds out whole
episodes). The data lives on the training device (the earlier CPU-side sampling made training 10x slower than the
model). Validation reports the one-step prediction loss against persistence, the K-step rollout loss in latent space,
and, with a head, the T position error decoded from the k-step predicted T token (arena units), the quantity the
planner's cost reads; model selection goes by that, not by the one-step loss (2610.19375).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from vmas_pusht.model import EntityJEPA, feats_from_state


def windows(d, H, fs, idx, gen, n, dev="cpu"):
    """Sample n windows from episodes idx -> states (n, H+1, D), actions (n, H, N, 2 fs), on d's device."""
    E, T1, D = d["state"].shape
    e = idx[torch.randint(len(idx), (n,), generator=gen)]
    t0 = torch.randint(0, T1 - H * fs - 1, (n,), generator=gen)
    rows = (t0[:, None] + torch.arange(H + 1)[None] * fs).to(dev)                 # (n, H+1)
    e = e.to(dev)
    s = d["state"][e[:, None], rows]
    arow = (t0[:, None] + torch.arange(H * fs)[None]).to(dev)                     # (n, H fs)
    a = d["action"][e[:, None], arow]                                             # (n, H fs, N, 2)
    a = a.reshape(n, H, fs, a.shape[2], 2).permute(0, 1, 3, 2, 4).reshape(n, H, a.shape[2], 2 * fs)
    return s, a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/vmas_pusht/train.pt")
    p.add_argument("--attn", default="full")
    p.add_argument("--out", default="data/vmas_pusht/wm_full.pt")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--rollout_k", type=int, default=5, help="validation rollout length")
    p.add_argument("--train_k", type=int, default=1, help="training rollout length (1 = teacher-forced one step)")
    p.add_argument("--head", type=int, default=0, help="1: state readout head on encoded + predicted tokens (delta loss)")
    p.add_argument("--head_weight", type=float, default=1.0)
    p.add_argument("--sigreg", default="pooled", choices=["pooled", "per_token"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    d = torch.load(args.data)
    E = len(d["goal"])
    perm = torch.randperm(E, generator=torch.Generator().manual_seed(0))
    tr, va = perm[: int(0.9 * E)], perm[int(0.9 * E):]
    S = d["state"][tr].reshape(-1, d["state"].shape[-1])
    mean, std = S.mean(0), S.std(0).clamp(min=1e-3)
    mean[10], std[10] = 0.0, 1.0                                               # T angle: sin / cos are taken later
    mean[14], std[14] = 0.0, 1.0                                               # padding
    d = {"state": ((d["state"] - mean) / std).to(dev), "action": d["action"].to(dev), "goal": d["goal"]}
    H, fs, K = args.history, args.frameskip, args.train_k
    model = EntityJEPA(history=H, act_dim=2 * fs, attn=args.attn, head=bool(args.head), sigreg_per_token=args.sigreg == "per_token").to(dev)
    g = torch.Generator().manual_seed(args.seed)
    # per-feature std of the one-block change of the token features (normalised units): the head's delta targets
    with torch.no_grad():
        s0, _ = windows(d, 1, fs, tr, g, 20000, dev)
        f0 = feats_from_state(s0, model.N)
        delta_std = (f0[:, 1] - f0[:, 0]).std(0).clamp(min=1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    log, t0 = [], time.time()
    mean_d, std_d = mean.to(dev), std.to(dev)
    # resume: every 1000 steps the optimiser state goes to <out>.ckpt, so a cancelled job continues where it stopped
    ck_path, start = Path(args.out + ".ckpt"), 1
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        g.set_state(ck["gen"]); log, start = ck["log"], ck["step"] + 1
        t0 -= ck["elapsed"]
        print(f"resumed from {ck_path} at step {ck['step']}", flush=True)

    @torch.no_grad()
    def validate():
        model.eval()
        gv = torch.Generator().manual_seed(123)
        s, a = windows(d, H, fs, va, gv, 4096, dev)
        z = model.encode(s)
        pred = model.predict(z[:, :-1], a)
        one = (pred - z[:, -1]).pow(2).mean().item()
        persist = (z[:, -2] - z[:, -1]).pow(2).mean().item()
        R = args.rollout_k
        s2, a2 = windows(d, H + R - 1, fs, va, gv, 2048, dev)
        z2 = model.encode(s2)
        roll = model.rollout(z2[:, :H], a2[:, :H - 1], a2[:, H - 1:])
        k_err = (roll - z2[:, H:]).pow(2).mean((0, 2, 3)).tolist()
        out = {"val_pred": one, "val_persistence": persist, "val_ratio": one / persist, "val_rollout_by_k": k_err}
        if model.state_head is not None:
            fT = model.decode(roll)[..., model.N * 4:model.N * 4 + 2] * std_d[8:10] + mean_d[8:10]   # (n, R, 2) raw units
            tT = s2[:, H:, 8:10] * std_d[8:10] + mean_d[8:10]
            out["val_head_Tpos_err_by_k"] = (fT - tT).norm(dim=-1).median(0).values.tolist()
            fE = model.decode(z2[:, H:])[..., model.N * 4:model.N * 4 + 2] * std_d[8:10] + mean_d[8:10]
            out["val_head_Tpos_err_encoded"] = float((fE - tT).norm(dim=-1).median())
            moved = (tT - (s2[:, H - 1:H, 8:10] * std_d[8:10] + mean_d[8:10])).norm(dim=-1)
            out["val_true_Tdisp_by_k_median"] = moved.median(0).values.tolist()
        model.train()
        return out

    for step in range(start, args.steps + 1):
        s, a = windows(d, H + K - 1, fs, tr, g, args.batch, dev)
        loss, parts = model.loss(s, a, k=K, delta_std=delta_std, head_weight=args.head_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0 or step == args.steps:
            v = validate()
            rec = {"step": step, "loss": loss.item(), **parts, **v, "min": round((time.time() - t0) / 60, 1)}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            ck_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "gen": g.get_state(),
                        "log": log, "step": step, "elapsed": time.time() - t0}, ck_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {**vars(args)}, "mean": mean, "std": std, "delta_std": delta_std.cpu(), "log": log}, args.out)
    print("->", args.out)
    ck_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
