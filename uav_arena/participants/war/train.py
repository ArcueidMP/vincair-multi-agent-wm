"""Train W-ar once Yihao's importable tokenizer/transformer backbone is available.

The backbone factory contract is documented in ``objective.py``. The same
factory and its options are saved in the checkpoint so the arena can rebuild
it for open-loop and closed-loop evaluation. W-ar optimizes only a multistep
physical-state loss; it has no JEPA latent prediction or SIGReg term.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from ...windows import Windows, load
from .model import build_backbone
from .objective import EntityDeltaHead, future_states, multistep_loss, state_history


def statistics(data: Windows, seed: int, count: int = 5000):
    """Physical-state and increment statistics from training windows only."""
    sample = np.random.default_rng(seed).integers(0, len(data), size=count)
    state_sum = state_sq = delta_sum = delta_sq = None
    state_count = delta_count = 0
    for start in range(0, count, 128):
        batch = default_collate([data[int(i)] for i in sample[start:start + 128]])
        history, origin = state_history(batch)
        future = future_states(batch, origin)
        states = torch.cat((history, future), dim=1).reshape(-1, history.shape[-1]).double()
        deltas = torch.diff(torch.cat((history[:, -1:], future), dim=1), dim=1)
        deltas = deltas.reshape(-1, history.shape[-1]).double()
        a, aa = states.sum(0), states.square().sum(0)
        b, bb = deltas.sum(0), deltas.square().sum(0)
        if state_sum is None:
            state_sum, state_sq, delta_sum, delta_sq = a, aa, b, bb
        else:
            state_sum += a; state_sq += aa; delta_sum += b; delta_sq += bb
        state_count += len(states)
        delta_count += len(deltas)
    state_mean, delta_mean = state_sum / state_count, delta_sum / delta_count
    state_std = (state_sq / state_count - state_mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
    delta_std = (delta_sq / delta_count - delta_mean.square()).clamp_min(0).sqrt().clamp_min(1e-5)
    return dict(state_mean=state_mean.float(), state_std=state_std.float(),
                delta_mean=delta_mean.float(), delta_std=delta_std.float())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path,
                        default=Path("data/uav_arena/datasets/explore_v2"))
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stats-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", required=True, help="Yihao's module:factory")
    parser.add_argument("--backbone-options", default="{}", help="JSON options agreed with Yihao")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 2000 or args.history < 1 or not 1 <= args.rollout_steps <= 15:
        parser.error("episodes must be 1..2000, history positive, rollout-steps 1..15")
    options = json.loads(args.backbone_options)
    if options.get("attn", "full") != "full":
        parser.error("W-ar must use the JEPA full-attention backbone")
    options.setdefault("history", args.history)
    if int(options["history"]) != args.history:
        parser.error("--history and backbone-options history must match")
    torch.manual_seed(args.seed)
    data = Windows(load(args.data)[:args.episodes], horizon=15, history=args.history)
    if not len(data):
        parser.error("the selected data contains no complete training windows")
    start = time.perf_counter()
    stats = statistics(data, args.seed, args.stats_samples)
    device_stats = {key: value.to(args.device) for key, value in stats.items()}
    backbone = build_backbone(args.backbone, options).to(args.device)
    head = EntityDeltaHead(backbone.token_dim).to(args.device)
    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=str(args.device).startswith("cuda"),
                        generator=torch.Generator().manual_seed(args.seed))
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
        loss = multistep_loss(backbone, head, batch, device_stats, args.rollout_steps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 250 == 0 or step == args.steps:
            print(f"step {step}/{args.steps}: multistep state loss {loss.item():.5f}", flush=True)
    config = dict(history=args.history, backbone=args.backbone, backbone_options=options)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=config, backbone_state=backbone.cpu().state_dict(),
                    head_state=head.cpu().state_dict(), stats=stats), args.out)
    info = dict(config=config, data=str(args.data), episodes=args.episodes, windows=len(data),
                steps=args.steps, rollout_steps=args.rollout_steps, batch=args.batch, seed=args.seed,
                seconds=time.perf_counter() - start,
                parameters=sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in head.parameters()))
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({key: info[key] for key in ("episodes", "windows", "parameters", "seconds")}))


if __name__ == "__main__":
    main()
