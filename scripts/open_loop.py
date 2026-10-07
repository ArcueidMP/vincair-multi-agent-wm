"""Does the plan that wins in the world model also win in reality? (held-out scenes, one GPU)

For windows of held-out coop episodes (frame t, goal = the real frame 25 env steps later, as in planning):
  1. score the recorded actions in the model: roll them out from z_t and measure ||z_hat - z_goal||^2
  2. run CEM in the model (300 x 30, top 30, horizon 5 x 5 env steps, as plan.py) to find its best plan
  3. execute both in the simulator from the window's true state (the partner replays its recorded actions),
     encode the real final frame, and measure the same cost there

If the model were right, CEM's plan would beat the recorded actions in reality about as often as in the model.

    python scripts/open_loop.py lewm_coop2_ftfull                       # own-action model
    python scripts/open_loop.py lewm_coop2_joint --data multipusht_2a_coop_heldout_joint.h5
-> $STABLEWM_HOME/open_loop/<checkpoint>.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data"))

import h5py
import hdf5plugin  # noqa: F401  (Blosc filter for the pixels)
import numpy as np
import stable_worldmodel as swm
import torch

from utils import pusht_stats
from world import MultiPushT

ENV = dict(n_agents=2, others="distinct", agent_force=1e5, solver_iterations=50, jpeg_quality=95)  # config/collect.yaml
K, BLOCK, S, ITERS, TOP = 5, 5, 300, 30, 30  # model steps, env steps per model step, CEM samples / iterations / elites
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet, as in training
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def encode(model, px):
    """(n, H, W, 3) uint8 -> (n, D) latents (encoder CLS token + projector), as the planner sees frames."""
    out = []
    for i in range(0, len(px), 256):
        x = (torch.from_numpy(px[i:i + 256]).cuda().permute(0, 3, 1, 2).float().div(255) - MEAN.cuda()) / STD.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model.projector(model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).float())
    return torch.cat(out)


@torch.no_grad()
def model_cost(model, z0, zg, acts, chunk=8192):
    """Roll action sequences (B, K, BLOCK * d) out from z0 (B, D) with one history frame, as planning does,
    -> ||z_hat_K - z_goal||^2 (B,)."""
    costs = []
    for i in range(0, len(z0), chunk):
        emb, a = z0[i:i + chunk, None], acts[i:i + chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(K):
                ae = model.action_encoder(a[:, :t + 1])
                emb = torch.cat([emb, model.predict(emb[:, -3:], ae[:, -3:])[:, -1:].float()], 1)
        costs.append((emb[:, -1].float() - zg[i:i + chunk]).pow(2).sum(-1))
    return torch.cat(costs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--data", default="multipusht_2a_coop_heldout.h5")
    p.add_argument("--windows", type=int, default=300)
    args = p.parse_args()
    torch.manual_seed(0)
    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval().requires_grad_(False)
    d = model.action_encoder.patch_embed.in_channels // BLOCK  # 2 = own action, 4 = [self, partner]

    with h5py.File(Path(os.environ["STABLEWM_HOME"], "datasets", args.data), "r") as f:
        off, length = f["ep_offset"][:], f["ep_len"][:]
        rng = np.random.default_rng(0)
        W = np.array([o + s for o, n in zip(off, length) for s in range(0, n - K * BLOCK - 1, K * BLOCK)])
        W = np.sort(rng.choice(W, min(args.windows, len(W)), replace=False))
        G = W + K * BLOCK
        px0, pxg = f["pixels"][W], f["pixels"][G]
        rows = W[:, None] + np.arange(K * BLOCK)
        joint = f["joint_action"][:][rows]                     # (n, 25, 4), agent-index order
        agent = f["agent_idx"][:][W]
        gstate = f["global_state"][:][W].astype(np.float64)
        T0, TG = f["global_state"][:][W][:, 4:7], f["global_state"][:][G][:, 4:7]
    n = len(W)
    own, partner = np.where(agent[:, None, None] == 0, joint[..., :2], joint[..., 2:]), np.where(agent[:, None, None] == 0, joint[..., 2:], joint[..., :2])
    ego = np.concatenate([own, partner], -1)[..., :d]        # what the model takes as action
    mu, sd = pusht_stats("action", d // 2)
    true_z = torch.from_numpy(((ego - mu) / sd).reshape(n, K, BLOCK * d).astype(np.float32)).cuda()

    z0, zg = encode(model, px0), encode(model, pxg)
    c_true = model_cost(model, z0, zg, true_z)

    # CEM in the model, batched over windows
    g = torch.Generator(device="cuda").manual_seed(0)
    m, s = torch.zeros(n, K, BLOCK * d, device="cuda"), torch.ones(n, K, BLOCK * d, device="cuda")
    for _ in range(ITERS):
        c = torch.randn(n, S, K, BLOCK * d, device="cuda", generator=g) * s[:, None] + m[:, None]
        c[:, 0] = m
        cost = model_cost(model, z0.repeat_interleave(S, 0), zg.repeat_interleave(S, 0), c.reshape(n * S, K, -1)).view(n, S)
        elite = c[torch.arange(n, device="cuda")[:, None], cost.topk(TOP, 1, largest=False).indices]
        m, s = elite.mean(1), elite.std(1)
    c_cem = model_cost(model, z0, zg, m)
    plan = np.clip(m.cpu().numpy().reshape(n, K * BLOCK, d) * sd + mu, -1, 1)

    # execute in the simulator: own half of the plan (or the recorded own action), partner replays its record
    env = MultiPushT(max_episode_steps=10 ** 6, **ENV)
    real = {}
    for name, own_actions in (("true", own), ("cem", plan[..., :2])):
        views, T_end = [], []
        for i in range(n):
            env.reset(seed=0, options={"state": gstate[i]})
            for t in range(K * BLOCK):
                pair = [own_actions[i, t], partner[i, t]] if agent[i] == 0 else [partner[i, t], own_actions[i, t]]
                env.core.simulate(np.stack(pair).astype(np.float32))
            views.append(env.core.render_view(int(agent[i])))
            T_end.append(env.core._get_obs()[4:6])
        real[name] = ((encode(model, np.stack(views)) - zg).pow(2).sum(-1), np.array(T_end))

    T_moves = (np.linalg.norm(TG[:, :2] - T0[:, :2], axis=1) >= 10) | (np.degrees(np.abs(np.angle(np.exp(1j * (TG[:, 2] - T0[:, 2]))))) >= 5)
    res = {}
    for cat, mask in (("all windows", np.ones(n, bool)), ("T moves", T_moves)):
        mk = torch.from_numpy(mask).cuda()
        res[cat] = {
            "n": int(mask.sum()),
            "model: CEM plan beats recorded actions": float((c_cem < c_true)[mk].float().mean()),
            "reality: CEM plan beats recorded actions": float((real["cem"][0] < real["true"][0])[mk].float().mean()),
            "median cost, recorded actions (model / reality)": [float(c_true[mk].median()), float(real["true"][0][mk].median())],
            "median cost, CEM plan (model / reality)": [float(c_cem[mk].median()), float(real["cem"][0][mk].median())],
            "median T error at the goal frame, px (recorded / CEM)": [
                float(np.median(np.linalg.norm(real["true"][1] - TG[:, :2], axis=1)[mask])),
                float(np.median(np.linalg.norm(real["cem"][1] - TG[:, :2], axis=1)[mask]))],
        }
    print(json.dumps(res, indent=1))
    out = Path(os.environ["STABLEWM_HOME"], "open_loop", args.checkpoint.replace("/", "_") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
