"""Train JEPA on arena training windows; select configurations on validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, default_collate

from ...windows import Windows, load
from .objective import EntityJEPA, window_states
from .training_checkpoint import CheckpointBatches, atomic_save


def statistics(data, seed=0, count=5000):
    """Training-only state moments and normalized physical-increment scales."""
    indices = np.random.default_rng(seed).integers(0, len(data), size=count)
    sums = [torch.zeros(51, dtype=torch.float64) for _ in range(4)]
    state_count = delta_count = 0
    for start in range(0, count, 128):
        batch = default_collate([data[int(i)] for i in indices[start:start+128]])
        history, future, _ = window_states(batch)
        state = torch.cat((history, future), dim=1).reshape(-1, 51).double()
        delta = torch.diff(torch.cat((history[:, -1:], future), dim=1), dim=1).reshape(-1, 51).double()
        if not torch.isfinite(state).all() or not torch.isfinite(delta).all():
            raise ValueError("nonfinite training states")
        for total, value in zip(sums, (state.sum(0), state.square().sum(0), delta.sum(0), delta.square().sum(0))):
            total += value
        state_count += len(state)
        delta_count += len(delta)
    mean = sums[0]/state_count
    std = (sums[1]/state_count-mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
    delta_mean = sums[2]/delta_count
    delta_std = (sums[3]/delta_count-delta_mean.square()).clamp_min(0).sqrt()/std
    return dict(state_mean=mean.float(), state_std=std.float(), delta_std=delta_std.clamp_min(1e-3).float())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/uav_arena/datasets/explore_v2"))
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--rollout-steps", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--pred-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attn", choices=("full", "no_cross", "flat"), default="full")
    parser.add_argument("--sigreg-weight", type=float, default=.09)
    parser.add_argument("--sigreg-per-token", action="store_true")
    parser.add_argument("--sigreg-projections", type=int, default=1024)
    parser.add_argument("--head-weight", type=float, default=1.)
    parser.add_argument("--stats-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Restore a complete training checkpoint")
    parser.add_argument("--stop-after", type=int, help="Save and stop after this global step")
    args = parser.parse_args(argv)
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("kind") != "jepa_training_resume_v1":
            raise ValueError("expected a complete training checkpoint")
        overrides = dict(out=args.out, resume=args.resume, stop_after=args.stop_after,
                         device=args.device, workers=args.workers)
        if args.workers != saved["training"]["workers"] or args.device != saved["training"]["device"]:
            raise ValueError("resume with the same workers and device")
        args = argparse.Namespace(**{**saved["training"], **overrides})
        args.data = Path(args.data)
    progress_path = args.out.with_suffix(".resume.pt")
    if progress_path.exists() and saved is None:
        raise FileExistsError(f"use a fresh output prefix: {progress_path}")
    if not 1 <= args.episodes <= 2000 or args.steps < 4 or not 1 <= args.rollout_steps <= 15:
        parser.error("episodes must be 1..2000, steps >= 4, rollout-steps 1..15")
    if min(args.batch, args.history, args.stats_samples, args.sigreg_projections) < 1:
        parser.error("batch/history/statistics/projection counts must be positive")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError(f"use a fresh output prefix: {args.out}")
    manifest_path = args.data/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["policy"] != "explore-v2":
        raise ValueError("this experiment uses explore-v2; do not mix behavior-policy versions")
    episodes = load(args.data, seeds=(200000, 200000+args.episodes))
    if len(episodes) != args.episodes:
        raise ValueError("the data must contain every selected standard training episode")
    torch.manual_seed(args.seed)
    data = Windows(episodes, horizon=15, history=args.history)
    if len(data) < args.batch:
        raise ValueError("batch exceeds the number of training windows")
    stats = saved["stats"] if saved else statistics(data, args.seed, args.stats_samples)
    device_stats = {key: value.to(args.device) for key, value in stats.items()}
    options = dict(history=args.history, dim=args.dim, latent=args.latent, enc_layers=args.enc_layers,
                   pred_layers=args.pred_layers, heads=args.heads, attn=args.attn)
    model = EntityJEPA(options, sigreg_weight=args.sigreg_weight, head_weight=args.head_weight,
                       sigreg_per_token=args.sigreg_per_token, sigreg_projections=args.sigreg_projections).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps,
                                                   pct_start=min(.5, max(.05, 2/args.steps)))
    generator = torch.Generator().manual_seed(args.seed)
    sampler = CheckpointBatches(BatchSampler(RandomSampler(data, generator=generator), args.batch, True),
                                saved["sampler"] if saved else None)
    loader = DataLoader(data, batch_sampler=sampler, num_workers=args.workers,
                        persistent_workers=args.workers > 0, generator=generator,
                        pin_memory=str(args.device).startswith("cuda"))
    iterator = iter(loader)
    log = saved["log"] if saved else []
    sources = [Path(__file__), Path(__file__).with_name("objective.py"),
               Path(__file__).with_name("model.py"), Path(__file__).with_name("backbone.py"),
               Path(__file__).with_name("training_checkpoint.py")]
    source_sha256 = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    training_args = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items() if key not in ("resume", "stop_after")}
    initial_step, elapsed = 0, 0.
    if saved:
        if saved["source_sha256"] != source_sha256 or saved["dataset_manifest_sha256"] != manifest_hash:
            raise ValueError("training source or data manifest changed since checkpoint")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        # A new DataLoader iterator draws a worker base seed. Undo that extra
        # draw: deterministic Windows data does not use worker randomness.
        generator.set_state(saved["loader_rng"])
        torch.set_rng_state(saved["cpu_rng"])
        if str(args.device).startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        initial_step, elapsed = saved["step"], saved["seconds"]
    start = time.perf_counter()
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    def save_resume(step):
        atomic_save(dict(kind="jepa_training_resume_v1", training=training_args, config=model.config,
                         model=model.state_dict(), stats=stats, optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), sampler=sampler.state_dict(),
                         loader_rng=generator.get_state(), cpu_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith("cuda") else [],
                         step=step, seconds=elapsed+time.perf_counter()-start, log=log,
                         source_sha256=source_sha256, dataset_manifest_sha256=manifest_hash), progress_path)
    step = initial_step
    try:
        for step in range(initial_step+1, args.steps+1):
            if stop_requested:
                step -= 1
                break
            try:
                batch = next(iterator)
            except StopIteration:
                sampler.next_epoch()
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(batch, device_stats, args.rollout_steps)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            sampler.commit()
            if step % 100 == 0 or step == args.steps or step == args.stop_after or stop_requested:
                record = dict(step=step, loss=float(loss.detach()), grad_norm=float(grad),
                              seconds=elapsed+time.perf_counter()-start, **{key: float(value) for key, value in parts.items()})
                log.append(record)
                print(json.dumps(record), flush=True)
                save_resume(step)
            if step % 1000 == 0 and step < args.steps:
                progress = args.out.with_suffix(f".step{step}.pt")
                progress.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(config=model.config, model={key: value.cpu() for key, value in model.state_dict().items()},
                                stats=stats, training_info=dict(step=step, planned_steps=args.steps,
                                                               incomplete=True, source_sha256=source_sha256)), progress)
            if step == args.stop_after or stop_requested:
                save_resume(step)
                print(json.dumps(dict(stage="paused", step=step, resume=str(progress_path))), flush=True)
                return
        if stop_requested:
            save_resume(step)
            print(json.dumps(dict(stage="paused", step=step, resume=str(progress_path))), flush=True)
            return
        save_resume(step)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    seconds = elapsed+time.perf_counter()-start
    info = dict(config=model.config, training={key: str(value) if isinstance(value, Path) else value
                                             for key, value in vars(args).items()},
                windows=len(data), parameters=sum(p.numel() for p in model.parameters()), seconds=seconds,
                dataset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                selected_seeds=[200000, 200000+args.episodes-1],
                source_sha256=source_sha256,
                peak_gpu_allocated_mib=(torch.cuda.max_memory_allocated()/1024**2
                                        if str(args.device).startswith("cuda") else 0), log=log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=model.config, model={key: value.cpu() for key, value in model.state_dict().items()},
                    stats=stats, training_info=info), args.out)
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps({key: info[key] for key in ("parameters", "windows", "seconds", "peak_gpu_allocated_mib")}), flush=True)


if __name__ == "__main__":
    main()
