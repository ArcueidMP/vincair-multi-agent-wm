"""Run planners on arena suites and write per-episode records.

    python -m uav_arena.run --predictor oracle --suite dev --seeds 0:5 --out data/uav_arena/results/oracle_dev.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .env import ArenaEnv
from .scenarios import CONDITIONS, SUITE_SEEDS, make_scenario


def run_episode(scenario, planner):
    env = ArenaEnv(scenario)
    obs = env.reset()
    planner.reset()
    plan_seconds = []
    while not env.done:
        start = time.perf_counter()
        setpoints = planner.act(obs, env)
        plan_seconds.append(time.perf_counter()-start)
        obs, reward, done, info = env.step(setpoints)
    return dict(seed=scenario.seed, condition=scenario.condition.name, outcome=info["outcome"], time=info["time"],
                decisions=info["decisions"], distance_to_goal=info["distance_to_goal"],
                min_clearance=info["min_clearance"], plan_seconds_mean=float(np.mean(plan_seconds)),
                plan_seconds_max=float(np.max(plan_seconds)))


def summarize(records):
    by_condition = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)
    rows = []
    for condition in [c.name for c in CONDITIONS if c.name in by_condition]:
        items = by_condition[condition]
        successes = [r for r in items if r["outcome"] == "success"]
        rows.append(dict(condition=condition, episodes=len(items), success=len(successes),
                         outcomes=dict(Counter(r["outcome"] for r in items)),
                         success_time_mean=float(np.mean([r["time"] for r in successes])) if successes else None))
    return rows


def make_planner(args):
    from .planning import ArenaCost, ArenaPlanner, ConstrainedArenaCost, PhysicsPredictor, TrueSimPredictor
    predictor = TrueSimPredictor(args.workers) if args.predictor == "oracle" else PhysicsPredictor(args.workers)
    cost = (ConstrainedArenaCost if args.solver == "lagrangian" else ArenaCost)(predictor)
    return ArenaPlanner(cost, solver=args.solver, horizon=args.horizon, samples=args.samples,
                        var_scale=args.var_scale, temperature=args.temperature, seed=args.seed), predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor", choices=("oracle", "physics"), default="oracle")
    parser.add_argument("--solver", choices=("mppi", "lagrangian"), default="mppi")
    parser.add_argument("--suite", choices=tuple(SUITE_SEEDS), default="dev")
    parser.add_argument("--seeds", default="0:5", help="slice into the suite's seed list, e.g. 0:5")
    parser.add_argument("--conditions", default="all", help="'all' or comma-separated names like static/0ms")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--var-scale", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.solver == "lagrangian":
        parser.error("the augmented-Lagrangian solver needs a differentiable predictor (a world model)")
    start, stop = (int(x) if x else None for x in args.seeds.split(":"))
    seeds = SUITE_SEEDS[args.suite][start:stop]
    conditions = CONDITIONS if args.conditions == "all" else [c for c in CONDITIONS if c.name in args.conditions.split(",")]
    planner, predictor = make_planner(args)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for condition in conditions:
            for seed in seeds:
                record = run_episode(make_scenario(seed, condition), planner)
                records.append(record)
                print(f"{condition.name:>20} seed {seed:>4}: {record['outcome']:<18} t={record['time']:5.1f}s "
                      f"plan={record['plan_seconds_mean']:.3f}s", flush=True)
                out.write_text(json.dumps(dict(config=vars(args), records=records, summary=summarize(records)),
                                          indent=1)+"\n")
    finally:
        if hasattr(predictor, "close"):
            predictor.close()
    for row in summarize(records):
        print(row)


if __name__ == "__main__":
    main()
