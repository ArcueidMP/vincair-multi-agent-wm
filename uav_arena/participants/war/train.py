"""Train W-ar on the shared entity-token backbone with JEPA's C/D optimizer recipe.

The backbone factory contract is documented in ``objective.py``. The same
factory and its options are saved in the checkpoint so the arena can rebuild
it for open-loop and closed-loop evaluation. W-ar optimizes only a multistep
physical-state loss; it has no JEPA latent prediction or SIGReg term. The
optimizer, schedule and clipping are those of ``participants/jepa/train.py``
(PR #3), so W-ar and JEPA ``full`` differ only in the objective and the head.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
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
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="peak of the one-cycle schedule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stats-samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", required=True, help="Yihao's module:factory")
    parser.add_argument("--backbone-options", default="{}", help="JSON options agreed with Yihao")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 2000 or args.history < 1 or not 1 <= args.rollout_steps <= 15 or args.steps < 4:
        parser.error("episodes must be 1..2000, history positive, rollout-steps 1..15, steps >= 4")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError(f"use a fresh output prefix: {args.out}")
    options = json.loads(args.backbone_options)
    if options.get("attn", "full") != "full":
        parser.error("W-ar must use the JEPA full-attention backbone")
    options.setdefault("history", args.history)
    if int(options["history"]) != args.history:
        parser.error("--history and backbone-options history must match")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None and manifest["policy"] != "explore-v2":
        parser.error("W-ar trains on explore-v2; do not mix behavior-policy versions")
    torch.manual_seed(args.seed)
    episodes = load(args.data)[:args.episodes]
    data = Windows(episodes, horizon=15, history=args.history)
    if not len(data):
        parser.error("the selected data contains no complete training windows")
    start = time.perf_counter()
    cuda = str(args.device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(args.device)
    stats = statistics(data, args.seed, args.stats_samples)
    device_stats = {key: value.to(args.device) for key, value in stats.items()}
    backbone = build_backbone(args.backbone, options).to(args.device)
    head = EntityDeltaHead(backbone.token_dim).to(args.device)
    parameters = list(backbone.parameters()) + list(head.parameters())
    # JEPA's C/D optimizer and schedule (participants/jepa/train.py on PR #3).
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps,
                                                    pct_start=min(.5, max(.05, 2 / args.steps)))
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=cuda, generator=torch.Generator().manual_seed(args.seed))
    iterator = iter(loader)
    log = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
        loss = multistep_loss(backbone, head, batch, device_stats, args.rollout_steps)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == args.steps:
            record = dict(step=step, loss=float(loss.detach()), grad_norm=float(grad), lr=lr,
                          seconds=time.perf_counter() - start)
            log.append(record)
            print(json.dumps(record), flush=True)
    seconds = time.perf_counter() - start
    config = dict(history=args.history, backbone=args.backbone, backbone_options=options)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=config, backbone_state=backbone.cpu().state_dict(),
                    head_state=head.cpu().state_dict(), stats=stats), args.out)
    here = Path(__file__).parent
    sources = [here / "train.py", here / "objective.py", here / "model.py",
               here.parent / "jepa" / "backbone.py", here.parent / "state.py"]
    info = dict(config=config, data=str(args.data), episodes=args.episodes, windows=len(data),
                selected_seeds=[int(episodes[0]["layout"].seed), int(episodes[-1]["layout"].seed)],
                steps=args.steps, rollout_steps=args.rollout_steps, batch=args.batch, lr=args.lr,
                seed=args.seed, workers=args.workers, stats_samples=args.stats_samples,
                optimizer="AdamW (weight decay 1e-3), OneCycleLR, gradient norm clipped at 1",
                seconds=seconds, parameters=sum(p.numel() for p in parameters),
                peak_gpu_allocated_mib=torch.cuda.max_memory_allocated(args.device) / 1024**2 if cuda else 0,
                device=torch.cuda.get_device_name(args.device) if cuda else "cpu",
                torch=torch.__version__, host=platform.node(),
                dataset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest else None,
                source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                log=log)
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({key: info[key] for key in ("episodes", "windows", "parameters", "seconds")}))


if __name__ == "__main__":
    main()
