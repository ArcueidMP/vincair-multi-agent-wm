"""Train CrazyMARL's official IPPO on arena tasks (CrazyMARL GPU venv).

The pinned upstream trainer, networks and hyperparameters run unchanged; only the
environment id and its arena arguments differ (see `crazymarl_env`). Writes the full
checkpoint (`.msgpack`), the actor weights for NumPy evaluation (`.npz`), per-update
training metrics (`_metrics.npz`, plus a live `.log.jsonl`) and a JSON report.

    PYTHONPATH=$PWD data/external/.venv-crazymarl-gpu/bin/python -m uav_arena.crazymarl_train \
        --checkout <CrazyMARL checkout> --task data/uav_arena/crazymarl/task.npz \
        --out data/uav_arena/crazymarl/ippo_seed0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from uav_transport.crazymarl_baseline import (_plain, load_upstream_ippo, official_config, sha, upstream_context,
                                              upstream_provenance)

EPISODE_SECONDS = 20.0


class MetricLog:
    """Stands in for the trainer's `wandb` module: per-update scalars as JSON lines."""

    def __init__(self, path):
        self.path = Path(path)

    def log(self, metric, step=None):
        values = {key: float(np.asarray(value)) for key, value in metric.items() if np.asarray(value).size == 1}
        with self.path.open("a") as f:
            f.write(json.dumps(dict(values, time=time.time()))+"\n")


def actor_weights(params):
    actor = params["params"]["actor_module"]
    layers = sorted(actor, key=lambda name: int(name.split("_")[1]))
    weights = {f"kernel{k}": np.asarray(actor[name]["kernel"]) for k, name in enumerate(layers)}
    weights.update({f"bias{k}": np.asarray(actor[name]["bias"]) for k, name in enumerate(layers)})
    weights["log_std"] = np.asarray(params["params"]["log_std"])
    return weights


def train(checkout, task_path, out, *, timesteps, num_envs, seed, progress_coef=0.25, smoke=False):
    import jax
    from flax import serialization
    checkout, out = Path(checkout).resolve(), Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    outputs = [out.with_suffix(s) for s in (".json", ".msgpack", ".npz", ".log.jsonl")]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Keep earlier runs: {out}")
    provenance = upstream_provenance(checkout)
    config = official_config(checkout)
    policy_hz = config["ENV_KWARGS"]["policy_freq"]
    config["ENV_NAME"] = "arenaquad_ix4"
    config["ENV_KWARGS"].update(task_path=str(Path(task_path).resolve()), progress_coef=float(progress_coef),
                                episode_length=int(EPISODE_SECONDS*policy_hz))
    config.update(TOTAL_TIMESTEPS=int(timesteps), NUM_ENVS=int(num_envs), SEED=int(seed), WANDB_MODE="disabled")
    if smoke:
        config.update(NUM_STEPS=16, NUM_MINIBATCHES=4, UPDATE_EPOCHS=1)
    started = time.perf_counter()
    with upstream_context(checkout):
        ippo, adapter = load_upstream_ippo(checkout)
        import jaxmarl
        from .crazymarl_env import make
        jaxmarl.make = make(jaxmarl.make)
        ippo.wandb = MetricLog(out.with_suffix(".log.jsonl"))
        rng = jax.random.PRNGKey(int(seed))
        rng, rng_init = jax.random.split(rng)
        result = jax.jit(ippo.make_train(config, rng_init))(rng)
        jax.block_until_ready(result)
        if hasattr(jax, "effects_barrier"):
            jax.effects_barrier()
    seconds = time.perf_counter()-started
    state = result["runner_state"][0]
    checkpoint = serialization.to_bytes(dict(params=state.params, opt_state=state.opt_state, step=state.step))
    out.with_suffix(".msgpack").write_bytes(checkpoint)
    np.savez(out.with_suffix(".npz"), **actor_weights(state.params))
    metrics = {key: np.asarray(value) for key, value in result["metrics"].items()}
    np.savez(out.with_name(out.name+"_metrics.npz"), **metrics)
    returns = metrics.get("returned_episode_returns")
    report = dict(kind="crazymarl_ippo_on_arena_tasks", smoke=bool(smoke), config=config, upstream=provenance,
                  adapter=adapter, wall_seconds=seconds, optimizer_steps=int(state.step),
                  environment_steps=int(config["NUM_UPDATES"]*config["NUM_STEPS"]*config["NUM_ENVS"]),
                  final_returned_episode_return=None if returns is None else float(returns[-1]),
                  checkpoint=str(out.with_suffix(".msgpack")), actor=str(out.with_suffix(".npz")),
                  actor_sha256=sha(out.with_suffix(".npz")), task_sha256=sha(task_path), source_sha256=sha(__file__),
                  jax_backend=jax.default_backend(), devices=[str(d) for d in jax.devices()])
    out.with_suffix(".json").write_text(json.dumps(_plain(report), indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", type=Path, default=Path("data/external/CrazyMARL"))
    parser.add_argument("--task", type=Path, default=Path("data/uav_arena/crazymarl/task.npz"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timesteps", type=float, default=2e9)
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-coef", type=float, default=0.25)
    parser.add_argument("--smoke", action="store_true", help="tiny rollouts and one epoch: an integration check")
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    report = train(args.checkout, args.task, args.out, timesteps=args.timesteps, num_envs=args.num_envs,
                   seed=args.seed, progress_coef=args.progress_coef, smoke=args.smoke)
    print(json.dumps({k: report[k] for k in ("wall_seconds", "optimizer_steps", "environment_steps",
                                              "final_returned_episode_return")}, indent=1))


if __name__ == "__main__":
    main()
