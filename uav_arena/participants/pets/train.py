"""Train the five-member PETS ensemble on arena windows.

Phase A example::

    python -m uav_arena.participants.pets.train \
      --data data/uav_arena/datasets/explore_v2 --episodes 250 --steps 5000 \
      --out data/uav_arena/participants/pets/phase_a.pt

``--steps`` is the number of optimizer steps *per ensemble member*. Each member
gets its own bootstrap resample of window/transition pairs. Statistics are
computed once from training data and shared by the five members.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from ...windows import Windows, load
from .model import HISTORY, HORIZON, ProbabilisticMLP, transition


class BootstrapTransitions(Dataset):
    """One fixed, independent bootstrap resample for an ensemble member."""

    def __init__(self, windows: Windows, seed: int):
        self.windows = windows
        rng = np.random.default_rng(seed)
        self.indices = rng.integers(0, len(windows), size=len(windows))
        # The delayed first transition is only 1/15 of a uniformly sampled
        # rollout, yet all later predictions depend on bridging it correctly.
        self.steps = rng.integers(1, windows.horizon, size=len(windows))
        self.steps[rng.random(len(windows)) < 0.25] = 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = self.windows[int(self.indices[index])]
        item["step"] = np.int64(self.steps[index])
        return item


def statistics(dataset: BootstrapTransitions, seed: int, count: int = 20_000,
               center_current: bool = True):
    """Input/increment normalization from a bounded sample of training transitions."""
    sample = np.random.default_rng(seed).integers(0, len(dataset), size=count)
    x_sum = y_sum = x_sq = y_sq = None
    for start in range(0, count, 256):
        batch = default_collate([dataset[int(i)] for i in sample[start:start + 256]])
        x, y = transition(batch, batch.pop("step"), center_current)
        a, b = x.double().sum(0), y.double().sum(0)
        aa, bb = x.double().square().sum(0), y.double().square().sum(0)
        if x_sum is None:
            x_sum, y_sum, x_sq, y_sq = a, b, aa, bb
        else:
            x_sum += a; y_sum += b; x_sq += aa; y_sq += bb
    x_mean, y_mean = x_sum / count, y_sum / count
    x_std = (x_sq / count - x_mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
    y_std = (y_sq / count - y_mean.square()).clamp_min(0).sqrt().clamp_min(1e-5)
    return dict(x_mean=x_mean.float(), x_std=x_std.float(),
                y_mean=y_mean.float(), y_std=y_std.float())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--steps", type=int, default=1000, help="optimizer steps per ensemble member")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats-samples", type=int, default=20_000)
    parser.add_argument("--center-current", action=argparse.BooleanOptionalAction, default=True,
                        help="center each transition on its current payload position")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 2000 or args.steps < 1 or args.members < 1:
        parser.error("episodes must be 1..2000; steps and members must be positive")
    torch.manual_seed(args.seed)
    windows = Windows(load(args.data)[:args.episodes], horizon=HORIZON, history=HISTORY)
    if not len(windows):
        parser.error("the selected data contains no complete training windows")
    start = time.perf_counter()
    stats = statistics(BootstrapTransitions(windows, args.seed), args.seed,
                       args.stats_samples, args.center_current)
    device_stats = {key: value.to(args.device) for key, value in stats.items()}
    config = dict(members=args.members, width=args.width, depth=args.depth,
                  history=HISTORY, center_current=args.center_current)
    members = []
    member_seconds = []
    for member_id in range(args.members):
        net = ProbabilisticMLP(args.width, args.depth).to(args.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
        dataset = BootstrapTransitions(windows, args.seed + 10_000 * member_id)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True,
                            num_workers=args.workers, persistent_workers=args.workers > 0,
                            pin_memory=str(args.device).startswith("cuda"),
                            generator=torch.Generator().manual_seed(args.seed + member_id))
        iterator = iter(loader)
        member_start = time.perf_counter()
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            x, y = transition(batch, batch.pop("step"), args.center_current)
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            x = (x - device_stats["x_mean"]) / device_stats["x_std"]
            y = (y - device_stats["y_mean"]) / device_stats["y_std"]
            loss = net.nll(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 250 == 0 or step == args.steps:
                print(f"member {member_id + 1}/{args.members} step {step}/{args.steps}: "
                      f"nll {loss.item():.3f}", flush=True)
        member_seconds.append(time.perf_counter() - member_start)
        members.append(net.cpu())
        del iterator, loader
    state = torch.nn.ModuleList(members).state_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=config, state=state, stats=stats), args.out)
    info = dict(config=config, data=str(args.data), episodes=args.episodes, windows=len(windows),
                steps=args.steps, batch=args.batch, seed=args.seed, seconds=time.perf_counter() - start,
                member_seconds=member_seconds,
                parameters=sum(p.numel() for member in members for p in member.parameters()))
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({key: info[key] for key in ("episodes", "windows", "parameters", "seconds")}))


if __name__ == "__main__":
    main()
