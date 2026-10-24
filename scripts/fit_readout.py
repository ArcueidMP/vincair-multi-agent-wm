"""Fit the state readout (readout.py) of a MultiPush-T model on its encoder latents of the held-out coop scenes.

Scenes are split 70/10/20 (train / early-stopping val / test, seed 0), so both agents' episodes of a scene land in
the same split. Targets are z-scored on train; MLP 2 x 512, AdamW (lr 1e-3, wd 1e-4), batch 256, early stopping on
val (patience 20, at most 300 epochs). The shipped cost_probe.pt files were fit this way.

    python collect.py seed=1 num_episodes=100 output.name=multipusht_2a_coop_heldout.h5   # or download.py --coop-heldout
    python scripts/fit_readout.py lewm_coop2_ftfull       # -> $STABLEWM_HOME/checkpoints/lewm_coop2_ftfull/readout/cost_probe.pt
    python scripts/fit_readout.py lewm_coop2_joint
"""
import argparse
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

from readout import TARGETS, make_probe, state_pose

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet, as in training and planning
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def scene_split(scene, seed=0):
    """Row indices per split, by scene: 70 / 10 / 20 of the shuffled scene ids."""
    scenes = np.random.default_rng(seed).permutation(np.unique(scene))
    a, b = int(0.7 * len(scenes)), int(0.8 * len(scenes))
    cut = {"train": scenes[:a], "val": scenes[a:b], "test": scenes[b:]}
    return {k: np.nonzero(np.isin(scene, v))[0] for k, v in cut.items()}


@torch.no_grad()
def encode(model, pixels, device, batch=256):
    """(n, 224, 224, 3) uint8 -> (n, D) latents (encoder CLS token + projector), as the planner sees frames."""
    out = []
    for lo in range(0, len(pixels), batch):
        x = torch.from_numpy(pixels[lo:lo + batch]).to(device).permute(0, 3, 1, 2).float().div(255)
        x = (x - MEAN.to(device)) / STD.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model.projector(model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).float().cpu())
    return torch.cat(out)


def fit(X, Y, device, seed=0, max_epochs=300, patience=20, batch=256):
    torch.manual_seed(seed)
    probe = make_probe(X["train"].shape[1], Y["train"].shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr, Xva, Yva = X["train"].to(device), Y["train"].to(device), X["val"].to(device), Y["val"].to(device)
    best, best_state, wait = np.inf, None, 0
    for _ in range(max_epochs):
        probe.train()
        for idx in torch.randperm(len(Xtr), device=device).split(batch):
            loss = (probe(Xtr[idx]) - Ytr[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            val = (probe(Xva) - Yva).pow(2).mean().item()
        if val < best - 1e-5:
            best, best_state, wait = val, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    probe.load_state_dict(best_state)
    return probe.eval()


def metrics(P, T):
    """Per-target Pearson r and median errors (self px, T px, T deg) of decoded poses P vs true poses T (n, 6)."""
    r = {n: float(np.corrcoef(P[:, j], T[:, j])[0, 1]) for j, n in enumerate(TARGETS)}
    dang = np.arctan2(P[:, 4], P[:, 5]) - np.arctan2(T[:, 4], T[:, 5])
    err = {"self px": float(np.median(np.linalg.norm(P[:, :2] - T[:, :2], axis=1))),
           "T px": float(np.median(np.linalg.norm(P[:, 2:4] - T[:, 2:4], axis=1))),
           "T deg": float(np.median(np.degrees(np.abs(np.arctan2(np.sin(dang), np.cos(dang))))))}
    return r, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="checkpoint folder under $STABLEWM_HOME/checkpoints")
    p.add_argument("--data", default="multipusht_2a_coop_heldout.h5", help="under $STABLEWM_HOME/datasets")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    home = Path(os.environ["STABLEWM_HOME"])

    h = h5py.File(home / "datasets" / args.data, "r")
    scene, agent = h["scene_idx"][:], h["agent_idx"][:]
    gs = h["global_state"][:]
    Y = np.where(agent[:, None] == 0, state_pose(gs, 0), state_pose(gs, 1))  # each row seen from its agent's view
    split = scene_split(scene)
    print({k: len(v) for k, v in split.items()}, "frames", flush=True)

    model = swm.wm.utils.load_pretrained(args.model).to(args.device).eval()
    Z = encode(model, h["pixels"], args.device)
    m, s = Y[split["train"]].mean(0), Y[split["train"]].std(0)
    Ys = torch.from_numpy((Y - m) / s)
    probe = fit({k: Z[v] for k, v in split.items()}, {k: Ys[v] for k, v in split.items()}, args.device)
    with torch.no_grad():
        P = probe(Z[split["test"]].to(args.device)).cpu().numpy() * s + m
    r, err = metrics(P, Y[split["test"]])
    print(f"{args.model}: test r {r}\n   median error {err}", flush=True)

    out = home / "checkpoints" / args.model / "readout" / "cost_probe.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"in_dim": Z.shape[1], "out_dim": Y.shape[1], "state_dict": probe.cpu().state_dict(),
                "mean": torch.from_numpy(m), "std": torch.from_numpy(s), "targets": TARGETS,
                "test_r": r, "test_median_err": err, "model": args.model,
                "data": f"{args.data}: encoder latents, scenes 70/10/20 (seed 0), metrics on the test scenes"}, out)
    print("->", out)


if __name__ == "__main__":
    main()
