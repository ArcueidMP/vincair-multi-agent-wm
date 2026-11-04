"""Weight-only warm start as a new experiment, with persistent exact step resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ...windows import Windows, load
from .objective import EntityJEPA


class CommittedBatches:
    """Prefetch cannot advance the cursor; only a completed optimizer step commits."""
    def __init__(self, size, batch, seed, epoch=0, cursor=0):
        self.size, self.batch, self.seed = size, batch, seed
        self.epoch, self.cursor = epoch, cursor

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed+self.epoch))
        for i in range(self.cursor, len(self)):
            yield order[i*self.batch:(i+1)*self.batch].tolist()

    def __len__(self):
        return self.size//self.batch

    def commit(self):
        self.cursor += 1

    def next_epoch(self):
        assert self.cursor == len(self)
        self.epoch += 1
        self.cursor = 0

    def state_dict(self):
        return dict(epoch=self.epoch, cursor=self.cursor)


def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=13000, help="Additional steps in the new experiment")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stop-after", type=int, help="Stop after this local step, saving resumable state")
    args = parser.parse_args(argv)
    if args.steps < 4 or args.workers < 0:
        parser.error("steps >= 4 and workers >= 0 required")
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError(args.out)
    latest = args.out.with_suffix(".resume.pt")
    if latest.exists() and not args.resume:
        raise FileExistsError(latest)
    if args.resume:
        pack = torch.load(args.resume, map_location="cpu", weights_only=True)
        if pack.get("kind") != "jepa_continuation_resume_v1":
            raise ValueError("resume requires a complete continuation checkpoint")
        setup = pack["setup"]
    else:
        if args.checkpoint is None or args.training_config is None:
            parser.error("warm start requires --checkpoint and --training-config")
        pack = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        reference = json.loads(args.training_config.read_text())
        if reference["config"] != pack["config"]:
            raise ValueError("parent architecture/objective differs from training configuration")
        setup = dict(training=reference["training"], steps=args.steps,
                     parent_checkpoint=str(args.checkpoint.resolve()), parent_sha256=digest(args.checkpoint),
                     parent_step=pack.get("training_info", {}).get("step", 0),
                     parent_optimizer_restored=False, experiment="weight_warm_start_optimizer_reset",
                     batch_order="seed_plus_epoch_committed_batches_v1")
    training = setup["training"]
    data_path = Path(training["data"])
    manifest_hash = digest(data_path/"manifest.json")
    manifest = json.loads((data_path/"manifest.json").read_text())
    if manifest["policy"] != "explore-v2":
        raise ValueError("expected explore-v2")
    if not 1 <= training["episodes"] <= 250:
        raise ValueError("this continuation is restricted to Phase A's first 250 episodes")
    sources = [Path(__file__), Path(__file__).with_name("objective.py"),
               Path(__file__).with_name("backbone.py"), Path(__file__).with_name("model.py")]
    source_hashes = {p.name: digest(p) for p in sources}
    if args.resume:
        if pack["dataset_manifest_sha256"] != manifest_hash or pack["source_sha256"] != source_hashes:
            raise ValueError("data manifest or training source changed since checkpoint")
    elif reference["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("training data manifest differs from original experiment")
    episodes = load(data_path, seeds=(200000, 200000+training["episodes"]))
    if len(episodes) != training["episodes"]:
        raise ValueError("missing selected training episodes")
    torch.manual_seed(training["seed"])
    model = EntityJEPA(**pack["config"]).to(args.device)
    model.load_state_dict(pack["model"], strict=True)
    stats = pack["stats"]
    device_stats = {k: v.to(args.device) for k, v in stats.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, training["lr"], total_steps=setup["steps"],
                                                   pct_start=min(.5, max(.05, 2/setup["steps"])))
    data = Windows(episodes, horizon=15, history=training["history"])
    if len(data) < training["batch"]:
        raise ValueError("batch exceeds training windows")
    sampler = CommittedBatches(len(data), training["batch"], training["seed"],
                               **(pack["sampler"] if args.resume else {}))
    loader = DataLoader(data, batch_sampler=sampler, num_workers=args.workers,
                        persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(training["seed"]),
                        pin_memory=str(args.device).startswith("cuda"))
    iterator = iter(loader)
    step, elapsed, log = 0, 0., []
    if args.resume:
        optimizer.load_state_dict(pack["optimizer"])
        scheduler.load_state_dict(pack["scheduler"])
        step, elapsed, log = pack["step"], pack["seconds"], pack["log"]
        torch.set_rng_state(pack["cpu_rng"])
        if str(args.device).startswith("cuda"):
            torch.cuda.set_rng_state_all(pack["cuda_rng"])
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    start = time.perf_counter()
    def save_state():
        state = dict(kind="jepa_continuation_resume_v1", config=model.config,
                     model=model.state_dict(), stats=stats, setup=setup, optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), sampler=sampler.state_dict(), step=step,
                     cpu_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith("cuda") else [],
                     seconds=elapsed+time.perf_counter()-start, log=log,
                     dataset_manifest_sha256=manifest_hash, source_sha256=source_hashes)
        atomic_save(state, latest)
    try:
        while step < setup["steps"] and not stop:
            try:
                batch = next(iterator)
            except StopIteration:
                sampler.next_epoch()
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(batch, device_stats, training["rollout_steps"])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            sampler.commit()
            step += 1
            if step % 100 == 0 or step == setup["steps"] or step == args.stop_after or stop:
                row = dict(step=step, total_weight_updates=setup["parent_step"]+step,
                           loss=float(loss.detach()), grad_norm=float(grad), lr=optimizer.param_groups[0]["lr"],
                           seconds=elapsed+time.perf_counter()-start,
                           **{k: float(v) for k, v in parts.items()})
                log.append(row)
                print(json.dumps(row), flush=True)
                save_state()
            if step == args.stop_after:
                stop = True
        save_state()
        if step < setup["steps"]:
            print(json.dumps(dict(stage="paused", step=step, resume=str(latest))), flush=True)
            return
        info = dict(config=model.config, training={**training, "steps": setup["steps"],
                    "out": str(args.out), "device": args.device, "workers": args.workers},
                    continuation=setup, step=step,
                    seconds=elapsed+time.perf_counter()-start, windows=len(data),
                    parameters=sum(p.numel() for p in model.parameters()), log=log,
                    dataset_manifest_sha256=manifest_hash, source_sha256=source_hashes,
                    selected_seeds=[200000, 200000+training["episodes"]-1])
        atomic_save(dict(config=model.config, model={k: v.cpu() for k, v in model.state_dict().items()},
                         stats=stats, training_info=info), args.out)
        args.out.with_suffix(".json").write_text(json.dumps(info, indent=2)+"\n")
        print(json.dumps(dict(stage="complete", checkpoint=str(args.out))), flush=True)
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    main()
