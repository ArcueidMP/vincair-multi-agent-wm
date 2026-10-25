"""Independent goal-free data and frozen models of joint physical options."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import torch

from vmas_pusht.abstraction_models import ResponseMLP, parameter_count
from vmas_pusht.action_interface import model_inputs
from vmas_pusht.boundary import _at_wall, _finite, _plain, _rotate
from vmas_pusht.joint_interface import HELPERS, INPUT_DIM, JointActionInterface, features
from vmas_pusht.option_models import local_contexts, local_programs, prepare_local_faces, normalization, fit_model, metrics, sha
from vmas_pusht.physical_abstraction import physical_state
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.scenario import wrap
from vmas_pusht.snapshot import capture, restore
from vmas_pusht.stable_composition import snapshot_from_record


SUPPORTS = ("passive", "world_anchor", "preloaded_anchor")
COUNTS = dict(train=32, val=8, test=8)


@torch.no_grad()
def execute(env, interface, options, enabled=None):
    """Run one complete joint option; save its real endpoint before padding."""
    n = env.num_envs
    enabled = torch.ones(n, dtype=torch.bool) if enabled is None else enabled
    initial = env.scenario.t_pose().clone()
    terminal = initial.clone()
    start_state = physical_state(env).clone()
    end_state = start_state.clone()
    clock = env.steps.clone()
    finite, wall = _finite(env), _at_wall(env)
    maximum_norm = torch.zeros(n, 2)
    effort = torch.zeros(n, 2)
    interface.start(options, enabled)
    maximum = interface.state.config.acquisition_limit + int(options[:, 3].max()) + 1
    integrated = 0
    for _ in range(maximum):
        active = interface.busy & enabled
        if not active.any():
            break
        command = interface.command(active)
        env.step([command[:, 0], command[:, 1]])
        integrated += 1
        status = interface.observe(active)
        norms = command.norm(dim=-1)
        maximum_norm = torch.maximum(maximum_norm, norms)
        effort += norms*env.world.dt
        finite &= ~active | _finite(env)
        wall |= active & _at_wall(env)
        done = status["newly_done"]
        terminal[done] = env.scenario.t_pose()[done]
        end_state[done] = physical_state(env)[done]
        clock[done] = env.steps[done]
    if (interface.busy & enabled).any() or maximum_norm.max() > 1.000001:
        raise AssertionError("Joint options must terminate under the unit force bound")
    response = terminal-initial
    response[:, 2] = wrap(response[:, 2])
    complete = interface.completed & finite & ~wall & enabled
    y = torch.cat((response, interface.elapsed[:, None]*env.world.dt, complete[:, None].float()), 1)
    return dict(Y=y, physical_start=start_state, physical_end=end_state,
        finite=finite, wall=wall, maximum_norm=maximum_norm, force_integral=effort,
        active_steps=interface.elapsed.clone(), padded_steps=n*integrated,
        failed=interface.failed.clone(), environment_end_step=clock)


@torch.no_grad()
def collect_support(support, out, counts=None, chunk_size=3072):
    torch.set_num_threads(1)
    started = time.perf_counter()
    contexts = local_contexts(COUNTS if counts is None else counts, seed=92617)
    prepared = prepare_local_faces(support, contexts, substeps=160)
    record = prepared["record"]
    rows, indices, options = [], [], []
    for index, source in enumerate(record["rows"]):
        if not source["preparation_completed"]:
            continue
        for route in ((0, 1, 2) if source["current_mode"] == 3 else (0,)):
            for program in local_programs(source["program_split"]):
                for helper in range(len(HELPERS)):
                    rows.append(dict(context_id=source["context_id"], split=source["split"],
                        support=support, source_index=index, current_mode=source["current_mode"],
                        source_origin=source["entry_origin"], helper_id=helper, route=route,
                        **program))
                    indices.append(index)
                    options.append([route, program["direction_deg"], program["strength"], program["push_steps"], helper])
    tensors = {}
    # Every second source gets an independently prescribed real helper action
    # before collection, covering helper-induced continuation states goal-free.
    source_env = make_profile("dry_light", num_envs=len(record["rows"]), substeps=160, seed=92617)
    restore(source_env, snapshot_from_record(record))
    source_interface = JointActionInterface(source_env, support, record["anchor"])
    source_modes = torch.tensor([r["current_mode"] if r["preparation_completed"] else 3 for r in record["rows"]])
    source_interface.reset_contact(torch.ones(source_env.num_envs, dtype=torch.bool), mode=source_modes)
    warm = torch.tensor([[0., 0., .3 if i % 3 else 0., 5., float(i % len(HELPERS))]
                         for i in range(source_env.num_envs)])
    warm_mask = torch.tensor([i % 2 == 1 and r["preparation_completed"] for i, r in enumerate(record["rows"])])
    # Preserve unselected physical starts exactly; batched zero force still
    # integrates free dynamics, so select original snapshot rows afterward.
    execute(source_env, source_interface, warm, warm_mask)
    continued = capture(source_env)
    original = snapshot_from_record(record)
    for old, new in zip(original.entities, continued.entities):
        for key, value in new.state.items():
            if value is not None:
                value[~warm_mask] = old.state[key][~warm_mask]
        if new.action is not None:
            for key, value in new.action.items():
                if value is not None and old.action[key] is not None:
                    value[~warm_mask] = old.action[key][~warm_mask]
    # Task/reward fields are not queried in local dynamics; restore their rows
    # too so the stored branching snapshot remains internally exact.
    for key, value in continued.scenario.items():
        if value is not None and original.scenario[key] is not None:
            value[~warm_mask] = original.scenario[key][~warm_mask]
    continued.steps[~warm_mask] = original.steps[~warm_mask]
    for begin in range(0, len(rows), chunk_size):
        end = min(begin+chunk_size, len(rows))
        idx = torch.tensor(indices[begin:end])
        option = torch.tensor(options[begin:end], dtype=torch.float32)
        env = make_profile("dry_light", num_envs=len(idx), substeps=160, seed=92617)
        restore(env, continued, idx)
        interface = JointActionInterface(env, support, record["anchor"][idx])
        interface.reset_contact(torch.ones(len(idx), dtype=torch.bool), mode=source_modes[idx])
        x = features(env, interface, option)
        angle = env.scenario.bar.state.rot[:, 0].clone()
        result = execute(env, interface, option)
        body_y = result["Y"].clone()
        body_y[:, :2] = _rotate(body_y[:, :2], -angle)
        values = dict(X=x, Y=body_y, world_Y=result["Y"], options=option, angle=angle,
            physical_start=result["physical_start"], physical_end=result["physical_end"])
        for key, value in values.items():
            tensors.setdefault(key, []).append(value)
        for j, row in enumerate(rows[begin:end]):
            row.update(helper_continuation=bool(warm_mask[idx[j]]),
                completed=bool(result["Y"][j, 4]), failed=bool(result["failed"][j]),
                finite=bool(result["finite"][j]), wall=bool(result["wall"][j]),
                elapsed_steps=int(result["active_steps"][j]),
                maximum_norm=result["maximum_norm"][j].tolist())
        if begin == 0 or end == len(rows) or end % (8*chunk_size) == 0:
            print(json.dumps(dict(support=support, collected=end, total=len(rows), seconds=time.perf_counter()-started)), flush=True)
    data = {key: torch.cat(value) for key, value in tensors.items()}
    data.update(rows=rows, prepared=prepared, source_snapshot=continued,
                helper_continuation_options=warm, helper_continuation_mask=warm_mask)
    validate(data)
    torch.save(data, out)
    return dict(path=str(out), sha256=sha(out), samples=len(rows), support=support,
        splits=dict(Counter(r["split"] for r in rows)), failures=sum(not r["completed"] for r in rows),
        seconds=time.perf_counter()-started)


def validate(data):
    n = len(data["rows"])
    if data["X"].shape != (n, INPUT_DIM) or data["Y"].shape != (n, 5):
        raise ValueError("Joint dataset schema mismatch")
    if not torch.isfinite(data["X"]).all() or not torch.isfinite(data["Y"]).all():
        raise ValueError("Nonfinite samples must not enter model training")
    groups = {}
    for row in data["rows"]:
        partition = "test" if row["split"].startswith("test") else row["split"]
        previous = groups.setdefault(row["context_id"], partition)
        if previous != partition:
            raise ValueError("Context crosses training/validation/test partitions")
        if max(row["maximum_norm"]) > 1.000001:
            raise ValueError("Training violated the physical force budget")


class FrozenJointPredictor:
    def __init__(self, entry):
        self.entry = entry
        self.model = ResponseMLP(INPUT_DIM, entry["width"], outputs=5).eval()
        self.model.load_state_dict(entry["state_dict"])
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x):
        norm = self.entry["normalization"]
        result = self.model((x-norm["x_mean"])/norm["x_scale"])*norm["y_scale"]+norm["y_mean"]
        result[..., 2] = wrap(result[..., 2])
        result[..., 3] = result[..., 3].clamp_min(0.)
        result[..., 4] = result[..., 4].clamp(0., 1.)
        return result


def train(paths, out, *, steps=2000, seeds=(0, 1, 2), width=128):
    torch.set_num_threads(1)
    parts = [torch.load(p, weights_only=False) for p in paths]
    data = dict(X=torch.cat([p["X"] for p in parts]), Y=torch.cat([p["Y"] for p in parts]),
                rows=[r for p in parts for r in p["rows"]])
    validate(data)
    splits = {name: torch.tensor([r["split"] == name for r in data["rows"]])
              for name in ("train", "val", "test_action", "test_context")}
    x, y = data["X"], data["Y"]
    norm = normalization(x[splits["train"]], y[splits["train"]])
    pack = dict(format_version=1, helpers=HELPERS, input_dim=INPUT_DIM,
        data_sha256={str(p): sha(p) for p in paths}, checkpoints={},
        training=dict(steps=steps, batch_size=512, lr=.003, width=width, seeds=list(seeds),
                      parameters=parameter_count(INPUT_DIM, width, 5), final_checkpoint=True,
                      normalization="training only", goal_free=True), metrics={})
    for seed in seeds:
        model = fit_model(x[splits["train"]], y[splits["train"]], norm, width,
                          steps=steps, batch_size=512, seed=seed)
        entry = dict(width=width, normalization=norm, state_dict=model.state_dict(), seed=seed)
        predict = FrozenJointPredictor(entry)
        validation = predict(x[splits["val"]])
        entry["duration_margin"] = float((validation[:, 3]-y[splits["val"], 3]).abs().quantile(.95))
        pack["checkpoints"][seed] = entry
        pack["metrics"][seed] = {name: metrics(y[mask], predict(x[mask])) for name, mask in splits.items()}
        print(json.dumps(_plain(dict(seed=seed, metrics=pack["metrics"][seed]))), flush=True)
    torch.save(pack, out)
    summary = {key: value for key, value in pack.items() if key != "checkpoints"}
    summary.update(model_path=str(out), model_sha256=sha(out),
        samples=len(x), splits={name: int(mask.sum()) for name, mask in splits.items()},
        duration_margins={seed: pack["checkpoints"][seed]["duration_margin"] for seed in seeds})
    Path(out).with_suffix(".json").write_text(json.dumps(_plain(summary), indent=2)+"\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--train-only", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output/(support+".pt") for support in SUPPORTS]
    if not args.train_only:
        if any(path.exists() for path in paths):
            raise FileExistsError("Preserve existing local data")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(collect_support, support, path) for support, path in zip(SUPPORTS, paths)]
            collection = [future.result() for future in futures]
        (args.output/"collection.json").write_text(json.dumps(collection, indent=2)+"\n")
    output = args.output/"models.pt"
    if output.exists():
        raise FileExistsError("Preserve frozen models")
    train(paths, output, steps=args.steps)


if __name__ == "__main__":
    main()
