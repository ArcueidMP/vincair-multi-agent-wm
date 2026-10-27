"""Evaluation layers 1-3 of a predictor on U's recorded decision states (HARNESS.md).

    python -m uav_arena.harness truth --states data/uav_arena/states/dev
    python -m uav_arena.harness score --predictor physics --states data/uav_arena/states/dev \
        --out data/uav_arena/results/harness/physics_dev.json

`truth` rolls the true simulator out from every question's snapshot, under U's plan and under 42
changes of it, and writes the trajectories to `<states>/truth/`. The true costs of the candidate
pool were recorded with the questions (`record.py`).

`score` replays every recorded episode to the predictor through the participant interface:
`reset()`, then `observe(obs, layout)` at every decision, with the observation the planner saw.
Only the privileged oracle receives the true state instead, as in `ArenaPlanner.act`. At a question
the predictor is called once, as the shared planner calls it, on U's plan, the 42 changes and the
512 candidates U scored: a leading shape of (1, 555). A stochastic predictor (`stochastic = True`)
is instead called once per sequence for layer 2, each call after the same seed, and 5 times with
different seeds for layers 1 and 3, whose metrics are averaged.
- Layer 1: the error of the predicted trajectory under U's plan.
- Layer 2: the error of each change's predicted effect, f(changed) - f(plan), against its true
  effect, on the payload, on the two aircraft that kept their increments and on the changed one.
- Layer 3: the regret of the plan MPPI executes on the predictor's costs of the same candidates, its
  true cost minus that of U's plan. The layer-3 number is the share of costly decisions, regrets
  above 1 cost unit, among questions whose reference is reliable: U's plan costs at most 1 more
  than the best candidate. Also elite regret, and the Spearman correlation and pairwise accuracy of
  the predicted costs over the candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import rankdata

from .collect import source_version
from .costs import soft_cost
from .env import ACTION_LIMIT, ArenaEnv
from .openloop import observation
from .planning import _rollouts, info_from_observation, setpoints_from
from .record import load, physics_state, regenerate_pool, true_scenario
from .scenarios import CONDITIONS, OBSTACLE_SETTINGS

A_CHANGES, B_CHANGES = 8, 6           # per aircraft
CHANGE_STD = 0.04                     # m per decision and axis: MPPI's sampling noise (swm multiplies it by var_scale)
OFFSET, RAMP = 0.10, 3                # B changes: a setpoint offset reached over 3 decisions, then held
DIRECTIONS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
CHANGE_KIND = np.repeat([0, 1], [3*A_CHANGES, 3*B_CHANGES])          # 0: A, 1: B
CHANGE_AIRCRAFT = np.concatenate([np.repeat(np.arange(3), A_CHANGES), np.repeat(np.arange(3), B_CHANGES)])
CHANGES = len(CHANGE_KIND)            # 42
PARTS = ("payload", "others", "own")
EFFECT_FLOOR = 0.005                  # m: relative errors only where the true effect reaches this
REPEATS = 5                           # calls of a stochastic predictor for layers 1 and 3
ELITES = (1, 8)
COSTLY = 1.0                          # cost units: a costly decision's regret, and an unreliable reference's excess
SEED = 7
DETERMINISM_ATOL = 1e-5               # m: repeated calls of a deterministic predictor may differ by this
TRUTH = "truth"


def changes(plan, layout_seed, obstacles, decision):
    """The 42 changed versions of U's plan (H, 9) -> (42, H, 9). Each aircraft in turn gets 8 A changes,
    Gaussian noise on its increments, and 6 B changes, a 0.10 m offset along +-x, +-y and +-z ramped in
    over the first 3 decisions; the other two keep their increments. Seeded by layout, obstacle setting
    and decision: the four latencies of a static layout, whose U episodes coincide, share them."""
    horizon = plan.shape[0]
    rng = np.random.default_rng([SEED, 0, int(layout_seed), OBSTACLE_SETTINGS.index(obstacles), int(decision)])
    noise = CHANGE_STD*rng.standard_normal((3*A_CHANGES, horizon, 3))
    ramp = np.diff(np.minimum(np.arange(1, horizon+1), RAMP)/RAMP, prepend=0.0)    # 1/3 for 3 decisions, then 0
    offsets = OFFSET*ramp[None, :, None]*np.tile(DIRECTIONS, (3, 1))[:, None, :]
    delta = np.zeros((CHANGES, horizon, 3, 3))
    delta[np.arange(CHANGES), :, CHANGE_AIRCRAFT] = np.concatenate([noise, offsets])
    changed = plan.unsqueeze(0)+torch.as_tensor(delta.reshape(CHANGES, horizon, 9), dtype=plan.dtype)
    return changed.clamp(-ACTION_LIMIT, ACTION_LIMIT)


def expand(info, n):
    """Batch-of-one info as the solver hands it to the cost: (1, n, ...)."""
    return {key: value.unsqueeze(1).expand(1, n, *value.shape[1:]) for key, value in info.items()}


def executed_plan(costs, candidates, temperature):
    """The frozen planner's update on costs (1, S) of candidates (1, S, H, 9): swm 0.1.1's MPPI
    softmax-weighted mean (one iteration, all samples), clipped as `ArenaPlanner.act` clips it."""
    weights = torch.softmax(-(costs-costs.min(dim=1, keepdim=True)[0])/temperature, dim=1)
    return (weights.unsqueeze(-1).unsqueeze(-1)*candidates).sum(dim=1)[0].clamp(-ACTION_LIMIT, ACTION_LIMIT)


def as_prediction(rollouts, dtype=torch.float32):
    """`planning._rollouts` output as the oracle returns it: (1, n, ...) tensors in the planner's dtype."""
    return {key: torch.as_tensor(value, dtype=dtype).unsqueeze(0) for key, value in rollouts.items()}


def true_costs(rollouts, info, increments):
    """The shared cost of exact rollouts (n sequences), computed as the oracle's planner computes it."""
    return soft_cost(as_prediction(rollouts), expand(info, increments.shape[1]), increments)[0]


def question_info(episode, decision):
    obs = observation(episode, decision, episode["layout"].condition.latency_ms)
    return obs, info_from_observation(obs, obs["recent_setpoints"][-1])


def sequences(episode, q):
    """U's plan, its 42 changes and the pool at question q, stacked: (1 + 42 + S, H, 9)."""
    k, layout = int(episode["question"][q]), episode["layout"]
    plan = torch.from_numpy(episode["plan"][k])
    return torch.cat([plan.unsqueeze(0), changes(plan, layout.seed, layout.condition.obstacles, k),
                      regenerate_pool(episode, q)])


# ---------- ground truth ----------
def episode_truth(pool, episode):
    scenario = true_scenario(episode)
    template = ArenaEnv(scenario)
    tasks, inputs = [], []
    for q, k in enumerate(episode["question"]):
        _, info = question_info(episode, k)
        stack = sequences(episode, q)[:1+CHANGES]
        increments, setpoints = setpoints_from(expand(info, len(stack)), stack.unsqueeze(0))
        tasks.append((scenario, physics_state(template, episode, q), setpoints[0].numpy().astype(float)))
        inputs.append((info, increments[:, :1]))
    results = pool.starmap(_rollouts, tasks)
    f32 = lambda key, rows: np.stack([r[key][rows] for r in results]).astype(np.float32)
    return dict(question=episode["question"].copy(), states_sha256=np.str_(episode["content_sha256"]),
                plan_uav=f32("uav", 0), plan_payload=f32("payload", 0), plan_payload_vel=f32("payload_vel", 0),
                plan_cost=np.array([float(true_costs({key: v[:1] for key, v in r.items()}, info, increments)[0])
                                    for r, (info, increments) in zip(results, inputs)], dtype=np.float32),
                change_uav=f32("uav", slice(1, None)), change_payload=f32("payload", slice(1, None)))


def compute_truth(states, workers):
    paths = episode_paths(states)
    (states/TRUTH).mkdir(exist_ok=True)
    with mp.get_context("fork").Pool(workers) as pool:
        for path in paths:
            start = time.perf_counter()
            np.savez_compressed(states/TRUTH/path.name, **episode_truth(pool, load(path)))
            print(f"{path.name}: {time.perf_counter()-start:.1f}s", flush=True)


# ---------- scoring ----------
def call(predictor, info, setpoints):
    with torch.inference_mode():
        prediction = predictor(info, setpoints)
    lead = tuple(setpoints.shape[:-2])
    for key, shape in dict(uav=(*lead, 3, 3), payload=(*lead, 3), payload_vel=(*lead, 3)).items():
        value = prediction[key]
        if tuple(value.shape) != shape:
            raise ValueError(f"prediction[{key!r}] has shape {tuple(value.shape)}, expected {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"prediction[{key!r}] is not finite")
    return {key: prediction[key].detach().to("cpu") for key in ("uav", "payload", "payload_vel")}


def call_seed(episode, decision, purpose, repeat=0):
    layout = episode["layout"]
    entropy = [SEED, purpose, layout.seed, CONDITIONS.index(layout.condition), int(decision), repeat]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def layer1(prediction, truth, q):
    """Position errors per step under U's plan: payload (H,) and the aircraft mean (H,)."""
    payload = np.linalg.norm(prediction["payload"].double().numpy()-truth["plan_payload"][q], axis=-1)
    uav = np.linalg.norm(prediction["uav"].double().numpy()-truth["plan_uav"][q], axis=-1).mean(-1)
    return payload, uav


def layer2(base, changed, truth, q):
    """Effect errors and true effect sizes per change, step and part (payload, others, own): two (42, H, 3)."""
    own = np.eye(3, dtype=bool)[CHANGE_AIRCRAFT][:, None, :]
    truth_base = dict(payload=truth["plan_payload"][q].astype(float), uav=truth["plan_uav"][q].astype(float))
    truth_changed = dict(payload=truth["change_payload"][q].astype(float), uav=truth["change_uav"][q].astype(float))
    true = {key: truth_changed[key]-truth_base[key] for key in truth_base}
    predicted = {key: changed[key].double().numpy()-base[key].double().numpy() for key in truth_base}

    def parts(payload, uav):
        return np.stack([np.linalg.norm(payload, axis=-1), (np.linalg.norm(uav, axis=-1)*~own).sum(-1)/2,
                         (np.linalg.norm(uav, axis=-1)*own).sum(-1)], -1)
    error = parts(predicted["payload"]-true["payload"], predicted["uav"]-true["uav"])
    return error, parts(true["payload"], true["uav"])


def ranking(predicted, true, pairs):
    """Elite regret (the true cost of the k candidates predicted best, minus that of the k truly best, for
    k = 1 and 8), Spearman correlation and pairwise accuracy of predicted costs over the candidates."""
    order, best = np.argsort(predicted, kind="stable"), np.sort(true)
    elites = [float(true[order[:k]].mean()-best[:k].mean()) for k in ELITES]
    a, b = rankdata(predicted), rankdata(true)
    spearman = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    i, j = pairs
    pairwise = float(np.mean(np.sign(predicted[i]-predicted[j]) == np.sign(true[i]-true[j])))
    return (*elites, spearman, pairwise)


def score_episode(predictor, pool, episode, truth, temperature):
    layout, scenario = episode["layout"], true_scenario(episode)
    template = ArenaEnv(scenario)
    stochastic, privileged = getattr(predictor, "stochastic", False), getattr(predictor, "privileged", False)
    questions = {int(k): q for q, k in enumerate(episode["question"])}
    samples = int(episode["config"]["samples"])
    pairs = np.triu_indices(samples, 1)
    true_pool = episode["pool_cost"].astype(float)
    rows, plans = [], []
    if hasattr(predictor, "reset"):
        predictor.reset()
    for k in range(len(episode["plan"])):
        obs, info = question_info(episode, k)
        if hasattr(predictor, "observe"):
            predictor.observe(obs, layout)
        if k not in questions:
            continue
        q = questions[k]
        if not hasattr(predictor, "observe") and privileged:
            predictor.set_state(scenario, physics_state(template, episode, q))
        stack = sequences(episode, q)
        big = expand(info, len(stack))
        increments, setpoints = setpoints_from(big, stack.unsqueeze(0))
        start = time.perf_counter()
        if stochastic:
            single = []
            for j in range(1+CHANGES):
                torch.manual_seed(call_seed(episode, k, 1))
                single.append(call(predictor, expand(info, 1), setpoints[:, j:j+1]))
            l2 = {key: torch.cat([s[key] for s in single], 1)[0] for key in single[0]}
            draws = []
            for r in range(REPEATS):
                torch.manual_seed(call_seed(episode, k, 2, r))
                draws.append(call(predictor, big, setpoints))
        else:
            draws = [call(predictor, big, setpoints)]
            l2 = {key: value[0, :1+CHANGES] for key, value in draws[0].items()}
        seconds = time.perf_counter()-start
        if not stochastic and q == 0:
            again = call(predictor, big, setpoints)
            if any(not torch.allclose(again[key], draws[0][key], rtol=0, atol=DETERMINISM_ATOL) for key in again):
                raise ValueError("The predictor returned different outputs for identical calls; a stochastic "
                                 "predictor must set stochastic = True (PARTICIPANTS.md, Section 2)")
        error, effect = layer2({key: v[:1] for key, v in l2.items()}, {key: v[1:] for key, v in l2.items()}, truth, q)
        l1, l3 = [], []
        pool_info = expand(info, samples)
        for prediction in draws:
            l1.append(layer1({key: v[0, 0] for key, v in prediction.items()}, truth, q))
            predicted_pool = {key: v[:, 1+CHANGES:].contiguous() for key, v in prediction.items()}
            costs = soft_cost(predicted_pool, pool_info, increments[:, 1+CHANGES:].contiguous())
            plans.append((q, executed_plan(costs, stack[1+CHANGES:].unsqueeze(0), temperature)))
            l3.append(ranking(costs[0].double().numpy(), true_pool[q], pairs))
        rows.append(dict(decision=k, l1_payload=np.mean([a for a, _ in l1], 0), l1_uav=np.mean([b for _, b in l1], 0),
                         l2_error=error.mean(1), l2_effect=effect.mean(1),
                         l2_steps=np.stack([error[CHANGE_KIND == kind].mean(0).T for kind in (0, 1)]),
                         l3=np.mean(l3, 0), seconds=seconds))
    # True costs of the executed plans, rolled out from each question's snapshot in parallel.
    by_question = {}
    for q, plan in plans:
        by_question.setdefault(q, []).append(plan)
    tasks, inputs = [], []
    for q, executed in by_question.items():
        _, info = question_info(episode, int(episode["question"][q]))
        increments, setpoints = setpoints_from(expand(info, len(executed)), torch.stack(executed).unsqueeze(0))
        tasks.append((scenario, physics_state(template, episode, q), setpoints[0].numpy().astype(float)))
        inputs.append((info, increments))
    for row, rollouts, (info, increments), q in zip(rows, pool.starmap(_rollouts, tasks), inputs, by_question):
        regret = true_costs(rollouts, info, increments).double().numpy()-float(truth["plan_cost"][q])
        row.update(l3_regret=float(regret.mean()), l3_costly=float(np.mean(regret > COSTLY)),
                   l3_reliable=bool(float(truth["plan_cost"][q]) <= true_pool[q].min()+COSTLY))
    return rows


def summary(rows, mask):
    """The quantities of ANALYSIS.md, Section 3, and the reported diagnostics, over the masked questions."""
    error, effect = rows["l2_error"][mask], rows["l2_effect"][mask]
    cross = lambda e: 0.5*e[..., 0]+0.5*e[..., 1]
    a, b = CHANGE_KIND == 0, CHANGE_KIND == 1
    costly, reliable, regret = rows["l3_costly"][mask], rows["l3_reliable"][mask], rows["l3_regret"][mask]
    out = dict(questions=int(mask.sum()), L1=float(rows["l1_payload"][mask].mean()), L1_uav=float(rows["l1_uav"][mask].mean()),
               L2=float(cross(error[:, a]).mean()), L2_B=float(cross(error[:, b]).mean()),
               L3=float(costly[reliable].mean()) if reliable.any() else float("nan"), L3_all=float(costly.mean()),
               regret_mean=float(regret.mean()), regret_median=float(np.median(regret)),
               unreliable=float(1-reliable.mean()))
    for kind, selected in (("A", a), ("B", b)):
        for p, part in enumerate(PARTS):
            out[f"L2_{kind}_{part}"] = float(error[:, selected, p].mean())
            out[f"effect_{kind}_{part}"] = float(effect[:, selected, p].mean())
    for p, part in enumerate(PARTS):
        e, s = error[:, a, p], effect[:, a, p]
        resolved = s >= EFFECT_FLOOR
        out[f"relative_A_{part}"] = float((e[resolved]/s[resolved]).mean()) if resolved.any() else float("nan")
        out[f"unresolved_A_{part}"] = float(1-resolved.mean())
    for name, column in zip(("top1", "top8", "spearman", "pairwise"), rows["l3"][mask].T):
        out[name] = float(np.nanmean(column))
    out["L1_steps"] = rows["l1_payload"][mask].mean(0).tolist()
    out["L2_steps"] = cross(np.moveaxis(rows["l2_steps"][mask][:, 0], 1, 2)).mean(0).tolist()
    out["seconds_per_question"] = float(rows["seconds"][mask].mean())
    return out


CONDITION_ORDER = {f"{c.obstacles}_{c.latency_ms}ms": i for i, c in enumerate(CONDITIONS)}


def episode_paths(states):
    """A record.py directory's episode files in the order they were flown: by condition, then seed."""
    return sorted(Path(states).glob("episode_*.npz"), key=lambda p: (CONDITION_ORDER[p.stem.split("_", 2)[2]],
                                                                    int(p.stem.split("_")[1])))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(args):
    from .run import make_predictor
    paths = episode_paths(args.states)
    if args.episodes:
        start, stop = (int(x) if x else None for x in args.episodes.split(":"))
        paths = paths[start:stop]
    options = json.loads(args.predictor_options)
    pool = mp.get_context("fork").Pool(args.workers)     # before the predictor, which may start CUDA
    predictor = make_predictor(args.predictor, args.workers, options)
    rows, labels = [], []
    try:
        for path in paths:
            episode = load(path)
            with np.load(args.states/TRUTH/path.name) as f:
                truth = {key: f[key] for key in f.files}
            if str(truth["states_sha256"]) != episode["content_sha256"]:
                raise ValueError(f"{path.name}: the ground truth was computed for other questions; rerun `truth`")
            start = time.perf_counter()
            episode_rows = score_episode(predictor, pool, episode, truth, float(episode["config"]["temperature"]))
            rows += episode_rows
            labels += [(episode["layout"].seed, episode["layout"].condition.name)]*len(episode_rows)
            l1 = np.mean([r["l1_payload"].mean() for r in episode_rows])
            l2 = np.mean([(0.5*r["l2_error"][CHANGE_KIND == 0, 0]+0.5*r["l2_error"][CHANGE_KIND == 0, 1]).mean()
                          for r in episode_rows])
            costly = [r["l3_costly"] for r in episode_rows if r["l3_reliable"]]
            l3 = np.mean(costly) if costly else float("nan")
            print(f"{episode['layout'].condition.name:>20} seed {episode['layout'].seed:>4}: {len(episode_rows):>3} questions "
                  f"L1 {l1:.4f} m  L2 {l2:.4f} m  L3 {l3:.3f}  {time.perf_counter()-start:.1f}s", flush=True)
    finally:
        pool.terminate()
        if hasattr(predictor, "close"):
            predictor.close()
    arrays = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    arrays.update(seed=np.array([s for s, _ in labels]), condition=np.array([c for _, c in labels]))
    conditions = [c.name for c in CONDITIONS if c.name in set(arrays["condition"])]
    summaries = dict(all=summary(arrays, np.ones(len(rows), dtype=bool)),
                     **{c: summary(arrays, arrays["condition"] == c) for c in conditions})
    manifest = args.states/"manifest.json"
    provenance = dict(source=source_version(), states=str(args.states), states_manifest_sha256=file_sha256(manifest),
                      checkpoint_sha256=file_sha256(options["checkpoint"]) if "checkpoint" in options else None,
                      stochastic=bool(getattr(predictor, "stochastic", False)),
                      privileged=bool(getattr(predictor, "privileged", False)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    args.out.write_text(json.dumps(dict(config={k: str(v) for k, v in vars(args).items()}, provenance=provenance,
                                        summary=summaries), indent=1)+"\n")
    row = summaries["all"]
    print(f"L1 {row['L1']:.4f} m  L2 {row['L2']:.4f} m  L3 {row['L3']:.3f} (mean regret {row['regret_mean']:+.3f})  "
          f"top-1 {row['top1']:.4f}  Spearman {row['spearman']:.3f}  pairwise {row['pairwise']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    truth = commands.add_parser("truth", help="true rollouts of every question's plan and changes")
    truth.add_argument("--states", type=Path, required=True, help="a record.py directory")
    truth.add_argument("--workers", type=int, default=None)
    scoring = commands.add_parser("score", help="layers 1-3 of one predictor")
    scoring.add_argument("--predictor", required=True, help="oracle, physics, adaptive or module:factory")
    scoring.add_argument("--predictor-options", default="{}", help="JSON keyword arguments for a module:factory predictor")
    scoring.add_argument("--states", type=Path, required=True, help="a record.py directory with its ground truth")
    scoring.add_argument("--episodes", default=None, help="start:stop slice of the episode files, for quick checks")
    scoring.add_argument("--workers", type=int, default=None)
    scoring.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.workers = int(args.workers or max(1, (os.cpu_count() or 2)-2))
    if args.command == "truth":
        compute_truth(args.states, args.workers)
    else:
        score(args)


if __name__ == "__main__":
    main()
