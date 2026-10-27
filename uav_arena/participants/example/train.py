"""Train the example participant on `windows.py` examples.

    python -m uav_arena.participants.example.train --data data/uav_arena/datasets/explore_v2 \
        --episodes 2000 --steps 20000 --out data/uav_arena/participants/example/mlp.pt

Every window of the first `--episodes` episodes (all decisions, all four latencies) is a training
example. Inputs and targets are normalized with statistics from a sample of the training windows;
the loss is the mean squared error of the normalized 15-decision prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from ...windows import STATE_KEYS, Windows, load
from .model import HORIZON, features, network, targets


def tensors(batch, device):
    batch = {key: value.to(device) for key, value in batch.items()}
    x, origin = features({key: batch[f"obs_{key}"][:, -1] for key in STATE_KEYS}, batch["age"],
                         batch["recent_setpoints"], batch["setpoints"])
    return x, targets(batch, origin)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="a collect.py directory")
    parser.add_argument("--episodes", type=int, default=2000, help="the first n episodes by seed")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    data = Windows(load(args.data)[:args.episodes], horizon=HORIZON)
    sample = np.random.default_rng(args.seed).choice(len(data), size=min(20_000, len(data)), replace=False)
    x, y = tensors(default_collate([data[int(i)] for i in sample]), "cpu")
    stats = dict(x_mean=x.mean(0), x_std=x.std(0).clamp_min(1e-6), y_mean=y.mean(0), y_std=y.std(0).clamp_min(1e-6))
    config = dict(inputs=x.shape[1], outputs=y.shape[1], width=args.width, depth=args.depth)
    net = network(config).to(args.device)
    device_stats = {key: value.to(args.device) for key, value in stats.items()}
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                        persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(args.seed))
    step, start = 0, time.perf_counter()
    while step < args.steps:
        for batch in loader:
            x, y = tensors(batch, args.device)
            loss = ((net((x-device_stats["x_mean"])/device_stats["x_std"])
                     -(y-device_stats["y_mean"])/device_stats["y_std"])**2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            step += 1
            if step % 1000 == 0 or step == args.steps:
                print(f"step {step:>6}: loss {loss.item():.4f} ({time.perf_counter()-start:.0f} s)", flush=True)
            if step == args.steps:
                break
    info = dict(vars(args), data=str(args.data), out=str(args.out), windows=len(data), seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in net.parameters()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=config, state=net.cpu().state_dict(), stats=stats), args.out)
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=1)+"\n")
    print(json.dumps(dict(parameters=info["parameters"], windows=info["windows"], seconds=round(info["seconds"]))))


if __name__ == "__main__":
    main()
