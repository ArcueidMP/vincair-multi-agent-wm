"""Freeze a JEPA checkpoint and fit a fresh physical readout for diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from ...windows import Windows, load
from .objective import EntityJEPA, StateReadout, cost_features, window_states


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_frozen(checkpoint, device):
    pack = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = EntityJEPA(**pack["config"]).to(device).eval()
    model.load_state_dict(pack["model"])
    model.requires_grad_(False)
    stats = {key: value.to(device) for key, value in pack["stats"].items()}
    return model, stats, pack


def evaluation_indices(data, stride):
    """Every stride-th decision of each episode, at every latency."""
    result = []
    latencies = len(data.latencies)
    for begin, end in zip(data.offsets[:-1], data.offsets[1:]):
        for k in range(0, int(end-begin)//latencies, stride):
            result.extend(int(begin)+k*latencies+j for j in range(latencies))
    return result


@torch.no_grad()
def encode_cache(model, stats, data, batch_size, device):
    """Cache all true-frame encodings on CPU; future frames are probe targets."""
    cached_z = cached_target = None
    offset = 0
    for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        history, future, _ = window_states(batch)
        state = torch.cat((history, future), dim=1)
        normalized = (state-stats["state_mean"])/stats["state_std"]
        z = model.backbone.encode(normalized).cpu()
        target = cost_features(normalized).cpu()
        if cached_z is None:
            cached_z = torch.empty((len(data), *z.shape[1:]), dtype=z.dtype)
            cached_target = torch.empty((len(data), *target.shape[1:]), dtype=target.dtype)
        cached_z[offset:offset+len(z)].copy_(z)
        cached_target[offset:offset+len(z)].copy_(target)
        offset += len(z)
        if offset % (batch_size*16) == 0 or offset == len(data):
            print(json.dumps(dict(cache_windows=offset, total_windows=len(data))), flush=True)
    if cached_z is None:
        raise ValueError("no complete training windows")
    return cached_z, cached_target


def physical_errors(prediction, target, stats):
    """Euclidean errors per window/frame, with position in cm and velocity in m/s."""
    error = (prediction-target)*cost_features(stats["state_std"])
    return dict(payload_cm=100*error[..., 9:12].norm(dim=-1),
                uav_cm=100*error[..., :9].unflatten(-1, (3, 3)).norm(dim=-1).mean(-1),
                payload_velocity_m_s=error[..., 12:15].norm(dim=-1))


@torch.no_grad()
def evaluate(head, cache, stats, history, device, batch_size=128, groups=1):
    errors = {key: [] for key in ("payload_cm", "uav_cm", "payload_velocity_m_s")}
    for z, target in DataLoader(TensorDataset(*cache), batch_size=batch_size):
        parts = physical_errors(head(z.to(device)), target.to(device), stats)
        for key, value in parts.items():
            errors[key].append(value.cpu())
    all_errors = {key: torch.cat(values) for key, values in errors.items()}

    def summarize(parts):
        report = {}
        for key, value in parts.items():
            future = value[:, history:]
            report[key] = dict(observed=float(value[:, history-1].mean()),
                               future_steps=future.mean(0).tolist(), future_mean=float(future.mean()),
                               steps_1_5=float(future[:, :5].mean()),
                               steps_6_10=float(future[:, 5:10].mean()),
                               steps_11_15=float(future[:, 10:15].mean()))
        return report

    result = dict(windows=len(cache[0]), overall=summarize(all_errors))
    if groups > 1:
        result["by_latency_ms"] = {str(ms): summarize({key: value[i::groups] for key, value in all_errors.items()})
                                   for i, ms in enumerate((0, 50, 100, 200))}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/uav_arena/datasets/explore_v2"))
    parser.add_argument("--val-data", type=Path, default=Path("data/uav_arena/datasets/explore_v2_val"))
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--val-episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True, help="fresh output prefix, without a suffix")
    args = parser.parse_args(argv)
    if not 1 <= args.episodes <= 250 or not 1 <= args.val_episodes <= 100 or args.steps < 4 or args.batch < 1:
        parser.error("episodes 1..250, val-episodes 1..100, steps >=4 and batch >=1 required")
    report_path, weights_path = Path(str(args.out)+".json"), Path(str(args.out)+".probe.pt")
    if report_path.exists() or weights_path.exists():
        raise FileExistsError("use a fresh diagnostic output prefix")
    start = time.perf_counter()
    manifests = {}
    for key, path in (("training", args.data), ("validation", args.val_data)):
        content = (path/"manifest.json").read_bytes()
        if json.loads(content)["policy"] != "explore-v2":
            raise ValueError("probe requires explore-v2")
        manifests[key] = hashlib.sha256(content).hexdigest()
    episodes = load(args.data, seeds=(200000, 200000+args.episodes))
    val_episodes = load(args.val_data, seeds=(299000, 299000+args.val_episodes))
    if len(episodes) != args.episodes or len(val_episodes) != args.val_episodes:
        raise ValueError("missing selected training or validation episodes")
    parent_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model, stats, pack = load_frozen(args.checkpoint, args.device)
    frozen_before = state_digest(model)
    history = model.backbone.history
    data = Windows(episodes, history=history)
    validation = Windows(val_episodes, history=history)
    indices = evaluation_indices(validation, 10)
    training_cache = encode_cache(model, stats, data, args.batch, args.device)
    validation_cache = encode_cache(model, stats, Subset(validation, indices), args.batch, args.device)
    torch.manual_seed(args.seed)
    head = StateReadout(model.backbone.token_dim, model.backbone.attn).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps,
                                                   pct_start=min(.5, max(.05, 2/args.steps)))
    loader = DataLoader(TensorDataset(*training_cache), batch_size=args.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    iterator, log = iter(loader), []
    for step in range(1, args.steps+1):
        try:
            z, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            z, target = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = (head(z.to(args.device))-target.to(args.device)).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite probe loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == args.steps:
            row = dict(step=step, loss=float(loss.detach()), seconds=time.perf_counter()-start)
            log.append(row)
            print(json.dumps(row), flush=True)
    head.eval()
    measurements = {}
    for split, cache, groups in (("training", training_cache, 1), ("validation", validation_cache, 4)):
        measurements[split] = {name: evaluate(readout, cache, stats, history, args.device, args.batch, groups)
                               for name, readout in (("original", model.readout), ("probe", head))}
    if state_digest(model) != frozen_before:
        raise AssertionError("frozen checkpoint parameters changed")
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != parent_hash:
        raise AssertionError("parent checkpoint file changed")
    result = dict(kind="frozen_encoder_readout_probe", parent_checkpoint_sha256=parent_hash,
                  model_config=pack["config"], training=dict(episodes=args.episodes,
                  seeds_inclusive=[200000, 200000+args.episodes-1], steps=args.steps, batch=args.batch,
                  lr=args.lr, weight_decay=.001, seed=args.seed, objective="encoded_readout_mse_only"),
                  validation=dict(episodes=args.val_episodes, seeds_inclusive=[299000, 299000+args.val_episodes-1],
                                  stride=10, latencies_ms=[0, 50, 100, 200]),
                  manifest_sha256=manifests, frozen_state_sha256=frozen_before,
                  frozen_parameters_unchanged=True,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  seconds=time.perf_counter()-start, measurements=measurements, log=log)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind=result["kind"], parent_checkpoint_sha256=parent_hash,
                    model_config=pack["config"], stats=pack["stats"],
                    readout={key: value.cpu() for key, value in head.state_dict().items()}), weights_path)
    result["probe_checkpoint_sha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(report=str(report_path), weights=str(weights_path), seconds=result["seconds"])), flush=True)
    return result


if __name__ == "__main__":
    main()
