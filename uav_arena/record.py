"""Record the oracle U's closed-loop episodes: the questions of the evaluation harness (HARNESS.md).

    python -m uav_arena.record --suite dev --seeds 0:5 --out data/uav_arena/states/dev \
        --results data/uav_arena/results/oracle_mppi_dev_s0-5_recorded.json

U flies with the frozen planner through `run.run_episode`, unchanged; wrappers around the planner's
MPPI solver and cost only read what each solve uses and returns. A file per episode extends
`collect.py`'s format (the true 10 ms state log and every setpoint sent, from which
`openloop.observation` rebuilds what a predictor sees at any decision) with U's clipped plan at
every decision and, at every 5th decision (a question):
- the physics snapshot U planned from: the plant's integration state, the onboard controller memory
  and the setpoints in force, enough for exact rollouts (`physics_state`);
- the state of the solver's random generator before the solve, from which `regenerate_pool`
  rebuilds the candidates U scored, bit for bit;
- U's cost of each of those candidates, which is their true cost.
The records written to `--results` are those `run.py` writes, so the recording is also U's
reference row. Predictors never read these files: the harness hands them observations only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from uav_transport import snapshot as plant_snapshot

from .collect import content_sha256, source_version
from .env import ACTION_LIMIT, quaternion_matrices
from .scenarios import CONDITIONS, SUITE_SEEDS, Scenario, make_scenario

QUESTION_EVERY = 5
POLICY = "oracle-mppi"
LOG_KEYS = ("time", "uav_pos", "uav_quat", "uav_vel", "uav_omega", "payload_pos", "payload_vel")


class _CostTap:
    """Stands in for the solver's cost and keeps what its last call scored."""

    def __init__(self, cost):
        self.cost = cost

    def get_cost(self, info, candidates):
        self.candidates, self.costs = candidates, self.cost.get_cost(info, candidates)
        return self.costs


class _SolverTap:
    """Stands in for the planner's MPPI solver and keeps its generator state before each solve and the result."""

    def __init__(self, solver):
        self.solver = solver
        self.cost = solver.model = _CostTap(solver.model)

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def solve(self, info, init_action=None):
        self.generator = self.solver.torch_gen.get_state()
        outputs = self.solver.solve(info, init_action=init_action)
        self.actions = outputs["actions"]
        return outputs


class RecordingPlanner:
    """Flies exactly as the wrapped `ArenaPlanner` and records every decision of the episode."""

    def __init__(self, planner, every=QUESTION_EVERY):
        self.planner, self.every = planner, int(every)
        self.cost = planner.cost                    # run_episode reads planner.cost.predictor
        self.tap = planner.solver = _SolverTap(planner.solver)

    def reset(self, episode):
        self.planner.reset(episode)
        self.env, self.plans, self.questions = None, [], []

    def act(self, obs, env):
        self.env, k = env, len(self.plans)
        state = env.physics_state() if k % self.every == 0 else None
        setpoints = self.planner.act(obs, env)
        self.plans.append(self.tap.actions[0].detach().clamp(-ACTION_LIMIT, ACTION_LIMIT).numpy())  # as act() clips it
        if state is not None:
            self.questions.append(dict(decision=k, state=state, generator=self.tap.generator.numpy().copy(),
                                       costs=self.tap.cost.costs[0].numpy().copy()))
        return setpoints

    def arrays(self):
        """The episode file's contents, once `run_episode` has finished."""
        env, planner = self.env, self.planner
        log = {key: np.stack([r[key] for r in env.log]) for key in LOG_KEYS}
        rotation = quaternion_matrices(log.pop("uav_quat"))
        log["uav_rot6"] = np.concatenate((rotation[..., 0], rotation[..., 1]), axis=-1)
        questions, states = self.questions, [q["state"] for q in self.questions]
        config = dict(horizon=planner.horizon, samples=planner.solver.num_samples, var_scale=planner.solver.var_scale,
                      temperature=planner.solver.temperature, seed=planner.seed, every=self.every)
        return dict(log, setpoint_time=np.array([a["time"] for a in env.actions]),
                    setpoints=np.stack([a["setpoints"] for a in env.actions]),
                    payload_mass=np.float64(env.scenario.payload_mass), outcome=np.str_(env.outcome),
                    scenario=np.str_(json.dumps(env.scenario.public().to_dict())), policy=np.str_(POLICY),
                    planner=np.str_(json.dumps(config)), plan=np.stack(self.plans),
                    question=np.array([q["decision"] for q in questions], dtype=np.int64),
                    generator=np.stack([q["generator"] for q in questions]),
                    pool_cost=np.stack([q["costs"] for q in questions]),
                    snapshot_state=np.stack([s["plant"].state for s in states]),
                    snapshot_lee=np.stack([np.frombuffer(s["lee"], dtype=np.uint8) for s in states]),
                    snapshot_setpoints=np.stack([s["setpoints"] for s in states]),
                    snapshot_setpoint_velocity=np.stack([s["setpoint_velocity"] for s in states]))


def record_episode(scenario, recorder):
    """`run.py`'s record of one episode and the episode file's arrays."""
    from .run import run_episode
    record = run_episode(scenario, recorder)
    return record, recorder.arrays()


def episode_name(scenario):
    c = scenario.condition
    return f"episode_{scenario.seed}_{c.obstacles}_{c.latency_ms}ms.npz"


# ---------- reading an episode file ----------
def load(path):
    with np.load(path) as f:
        episode = {key: f[key] for key in f.files}
    episode["content_sha256"] = content_sha256(episode)
    episode["layout"] = Scenario.from_dict(json.loads(str(episode["scenario"])))
    episode["config"] = json.loads(str(episode["planner"]))
    return episode


def true_scenario(episode):
    """The scenario with its hidden payload mass: for the simulator only, never for a predictor."""
    return dataclasses.replace(episode["layout"], payload_mass=float(episode["payload_mass"]))


def physics_state(env, episode, q):
    """`ArenaEnv.physics_state()` at question q, for an environment of the episode's true scenario.

    The plant's bookkeeping (step counters, metrics) is the environment's own: the arena's physics
    steps never read it.
    """
    plant = dataclasses.replace(plant_snapshot.capture(env.plant), state=episode["snapshot_state"][q].copy())
    return dict(plant=plant, lee=episode["snapshot_lee"][q].tobytes(), setpoints=episode["snapshot_setpoints"][q].copy(),
                setpoint_velocity=episode["snapshot_setpoint_velocity"][q].copy())


def regenerate_pool(episode, q):
    """The candidates U scored at question q (samples, horizon, 9), bit for bit: MPPI's sampling step
    from the recorded generator state around its warm start, the previous plan shifted by one
    decision and padded with zeros (swm 0.1.1, `MPPISolver.solve`, one iteration)."""
    config, k = episode["config"], int(episode["question"][q])
    mean = torch.zeros(1, config["horizon"], 9)
    if k:
        mean[0, :-1] = torch.from_numpy(episode["plan"][k-1][1:])
    var = config["var_scale"]*torch.ones(1, config["horizon"], 9)
    generator = torch.Generator()
    generator.set_state(torch.from_numpy(episode["generator"][q].copy()))
    noise = torch.randn(1, config["samples"], config["horizon"], 9, generator=generator)
    candidates = mean.unsqueeze(1)+noise*var.unsqueeze(1)
    candidates[:, 0] = mean
    return candidates[0]


def main():
    from .run import make_planner, summarize
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=tuple(SUITE_SEEDS), default="dev")
    parser.add_argument("--seeds", default="0:5", help="slice into the suite's seed list, e.g. 0:5")
    parser.add_argument("--conditions", default="all", help="'all' or comma-separated names like static/0ms")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True, help="directory for the episode files")
    parser.add_argument("--results", type=Path, required=True, help="run.py-format JSON of U's records")
    args = parser.parse_args()
    start, stop = (int(x) if x else None for x in args.seeds.split(":"))
    seeds = SUITE_SEEDS[args.suite][start:stop]
    conditions = CONDITIONS if args.conditions == "all" else [c for c in CONDITIONS if c.name in args.conditions.split(",")]
    # The frozen planner with the oracle, exactly as run.py builds it from its defaults.
    config = argparse.Namespace(predictor="oracle", predictor_options="{}", solver="mppi", suite=args.suite,
                                seeds=args.seeds, conditions=args.conditions, horizon=15, samples=512, var_scale=0.04,
                                temperature=0.05, workers=args.workers, seed=0, out=str(args.results))
    planner, predictor = make_planner(config)
    recorder = RecordingPlanner(planner)
    args.out.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    records, files = [], []
    try:
        for condition in conditions:
            for seed in seeds:
                scenario = make_scenario(seed, condition)
                record, arrays = record_episode(scenario, recorder)
                path = args.out/episode_name(scenario)
                np.savez_compressed(path, **arrays)
                records.append(record)
                files.append(dict(file=path.name, seed=int(seed), condition=condition.name, outcome=record["outcome"],
                                  decisions=record["decisions"], questions=len(arrays["question"]),
                                  content_sha256=content_sha256(arrays)))
                print(f"{condition.name:>20} seed {seed:>4}: {record['outcome']:<18} t={record['time']:5.1f}s "
                      f"plan={record['plan_seconds_mean']:.3f}s questions={len(arrays['question'])}", flush=True)
                args.results.write_text(json.dumps(dict(config=vars(config), records=records, summary=summarize(records)),
                                                   indent=1)+"\n")
                manifest = dict(policy=POLICY, suite=args.suite, seeds=args.seeds, conditions=args.conditions,
                                every=QUESTION_EVERY, source=source_version(), episodes=files)
                (args.out/"manifest.json").write_text(json.dumps(manifest, indent=1)+"\n")
    finally:
        predictor.close()
    for row in summarize(records):
        print(row)


if __name__ == "__main__":
    main()
