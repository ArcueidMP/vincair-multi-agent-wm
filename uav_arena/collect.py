"""World-model training data: seeded arena episodes flown by a fixed exploration policy.

    python -m uav_arena.collect --seeds 200000:202000 --out data/uav_arena/datasets/explore_v2
    python -m uav_arena.collect --verify data/uav_arena/datasets/explore_v2

Each episode is the open-field version of a training seed's scenario: the seed's start and hidden
payload mass, without obstacles (they are non-physical, and every participant scores them with the
same forecasts, so a world model only has to predict the team). A behavior policy flies it for up
to 20 s: `ExplorerV2` (`explore-v2`, the standard datasets since `arena-v0.3`) or the original
`Explorer` (`explore-v1`, kept with its reference manifests). A file keeps the true 10 ms state log
and every setpoint sent, so any observation latency can be applied offline: at decision time t with
latency d the observation is the record at t - d. Episodes are deterministic in the seed and the
policy. `manifest.json` lists each file's SHA-256 and a hash of its contents; `--verify` compares a
directory's contents with the reference manifests committed in `uav_arena/manifests/`. Unlike a
file's SHA-256, the content hash does not depend on the zlib build that compressed the file.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess

import numpy as np

from .env import ACTION_LIMIT, DECISION_DT, FORMATION_DEPTH, ArenaEnv, quaternion_matrices
from .scenarios import Condition, make_scenario

POLICY = "explore-v2"                   # the standard datasets' behavior policy
TRAINING_SEEDS = (200_000, 300_000)     # disjoint from dev, test and CrazyMARL's 100000-102047
WAYPOINT_BOX = ((-2.2, 2.2), (-1.7, 1.7), (0.6, 1.8))   # payload waypoints (m)
HANG_MARGIN = 0.35                      # explore-v1: aircraft setpoints at least this far above the payload
SEPARATION = 0.3                        # explore-v1: minimum distance between aircraft setpoints
# explore-v2 takes its action statistics from the frozen planner and cost (a test checks them):
V2_NOISE = 0.04                         # m per decision and axis: the shared MPPI's sampling scale (var_scale)
V2_PERSISTENCE = float(np.exp(-1/15))   # per decision: an offset decays over the planner's 15-decision horizon
V2_SPEEDS = (0.2, ACTION_LIMIT/DECISION_DT)   # formation speeds up to the planner's action limit (m/s)
V2_HANG_MARGIN = 0.30                   # the cost's hang_margin
V2_SEPARATION = 0.18                    # the cost's team_margin
REFERENCES = Path(__file__).with_name("manifests")     # reference manifests of the standard datasets


class Explorer:
    """Formation flight between random waypoints at random speeds and formation sizes, with a
    hover after each, plus each aircraft's own Ornstein-Uhlenbeck deviation (time constant 1 s,
    spread drawn per episode), so the data also covers per-aircraft sequences like those MPPI
    samples. Increments obey the planner's action limit; targets keep each aircraft above the
    payload and apart from the others.
    """

    def __init__(self, obs, rng):
        hover = obs["recent_setpoints"][-1]
        self.rng, self.centre, self.offsets = rng, hover.mean(0), hover-hover.mean(0)
        self.spread = rng.uniform(0.0, 0.05)        # stationary std of a deviation (m)
        self.deviation = np.zeros((3, 3))
        self._waypoint()

    def _waypoint(self):
        self.target = np.array([self.rng.uniform(*bounds) for bounds in WAYPOINT_BOX])+[0., 0., FORMATION_DEPTH]
        self.speed, self.scale, self.dwell = self.rng.uniform(0.2, 1.0), self.rng.uniform(0.9, 1.15), self.rng.uniform(0., 2.)

    def __call__(self, obs):
        step = self.target-self.centre
        distance = np.linalg.norm(step)
        if distance > 1e-9:
            self.centre = self.centre+step*min(1.0, self.speed*DECISION_DT/distance)
        else:
            self.dwell -= DECISION_DT
            if self.dwell <= 0:
                self._waypoint()
        self.deviation += -self.deviation*DECISION_DT+self.spread*np.sqrt(2*DECISION_DT)*self.rng.standard_normal((3, 3))
        target = self.centre+self.scale*self.offsets+self.deviation
        target[:, 2] = np.maximum(target[:, 2], obs["payload_pos"][2]+HANG_MARGIN)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            gap = target[i]-target[j]
            norm = np.linalg.norm(gap)
            if norm < SEPARATION:
                push = gap/max(norm, 1e-9)*(SEPARATION-norm)/2
                target[i] += push
                target[j] -= push
        self.deviation = target-self.centre-self.scale*self.offsets     # the shaped deviation carries on
        last = obs["recent_setpoints"][-1]
        return last+np.clip(target-last, -ACTION_LIMIT, ACTION_LIMIT)


class ExplorerV2:
    """explore-v2: explore-v1's formation flight between waypoints, flown with the planner's action
    statistics. Every decision, each aircraft's increment is the step to its place in the formation
    plus fresh Gaussian noise at MPPI's sampling scale, as a planner candidate's increments differ
    from its plan's. The offsets this builds up decay over the planner's horizon, so within one
    horizon they wander like a candidate's. Formation speeds reach the planner's action limit, and
    targets keep only the cost's margins: 0.30 m above the payload and 0.18 m between aircraft.
    """

    def __init__(self, obs, rng):
        hover = obs["recent_setpoints"][-1]
        self.rng, self.centre, self.offsets = rng, hover.mean(0), hover-hover.mean(0)
        self.deviation = np.zeros((3, 3))
        self._waypoint()

    def _waypoint(self):
        self.target = np.array([self.rng.uniform(*bounds) for bounds in WAYPOINT_BOX])+[0., 0., FORMATION_DEPTH]
        self.speed, self.scale, self.dwell = self.rng.uniform(*V2_SPEEDS), self.rng.uniform(0.9, 1.15), self.rng.uniform(0., 2.)

    def __call__(self, obs):
        step = self.target-self.centre
        distance = np.linalg.norm(step)
        if distance > 1e-9:
            self.centre = self.centre+step*min(1.0, self.speed*DECISION_DT/distance)
        else:
            self.dwell -= DECISION_DT
            if self.dwell <= 0:
                self._waypoint()
        self.deviation = V2_PERSISTENCE*self.deviation+V2_NOISE*self.rng.standard_normal((3, 3))
        target = self.centre+self.scale*self.offsets+self.deviation
        target[:, 2] = np.maximum(target[:, 2], obs["payload_pos"][2]+V2_HANG_MARGIN)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            gap = target[i]-target[j]
            norm = np.linalg.norm(gap)
            if norm < V2_SEPARATION:
                push = gap/max(norm, 1e-9)*(V2_SEPARATION-norm)/2
                target[i] += push
                target[j] -= push
        self.deviation = target-self.centre-self.scale*self.offsets     # the shaped offset carries on
        last = obs["recent_setpoints"][-1]
        return last+np.clip(target-last, -ACTION_LIMIT, ACTION_LIMIT)


POLICIES = {"explore-v1": (Explorer, (17,)), "explore-v2": (ExplorerV2, (17, 2))}    # class, random stream


def collect_episode(seed, policy=POLICY):
    scenario = replace(make_scenario(seed, Condition("static", 0)), pillars=(), movers=())
    env = ArenaEnv(scenario)
    obs = env.reset()
    explorer, stream = POLICIES[policy]
    behavior = explorer(obs, np.random.default_rng([int(seed), *stream]))
    while not env.done:
        obs, *_ = env.step(behavior(obs))
    log = {key: np.stack([r[key] for r in env.log]) for key in ("time", "uav_pos", "uav_quat", "uav_vel", "uav_omega",
                                                                 "payload_pos", "payload_vel")}
    rotation = quaternion_matrices(log.pop("uav_quat"))
    log["uav_rot6"] = np.concatenate((rotation[..., 0], rotation[..., 1]), axis=-1)
    return dict(log, setpoint_time=np.array([a["time"] for a in env.actions]),
                setpoints=np.stack([a["setpoints"] for a in env.actions]),
                payload_mass=np.float64(scenario.payload_mass), outcome=np.str_(env.outcome),
                scenario=np.str_(json.dumps(scenario.public().to_dict())), policy=np.str_(policy))


def content_sha256(arrays):
    """SHA-256 over an episode's array names, dtypes, shapes and values, in name order."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        digest.update(f"{key}:{value.dtype.str}:{value.shape};".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _write(task):
    seed, out, policy = task if len(task) == 3 else (*task, POLICY)
    path = Path(out)/f"episode_{seed}.npz"
    episode = collect_episode(seed, policy)
    np.savez_compressed(path, **episode)
    return dict(file=path.name, seed=int(seed), outcome=str(episode["outcome"]), records=len(episode["time"]),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), content_sha256=content_sha256(episode))


def _content(path):
    with np.load(path) as f:
        return int(path.stem.split("_")[1]), str(f["policy"]), content_sha256({key: f[key] for key in f.files})


def verify(directory, references=None, processes=None):
    """Seeds of the episodes in `directory` whose contents match the reference manifests of their
    policy (by default every manifest in `uav_arena/manifests/`), differ from them, or have no reference."""
    expected = {}
    for path in references or sorted(REFERENCES.glob("*.json")):
        manifest = json.loads(Path(path).read_text())
        expected.update({(manifest["policy"], e["seed"]): e["content_sha256"] for e in manifest["episodes"]})
    with mp.get_context("fork").Pool(processes) as pool:
        found = pool.map(_content, sorted(Path(directory).glob("episode_*.npz")), chunksize=8)
    result = dict(match=[], differ=[], unknown=[])
    for seed, policy, digest in sorted(found):
        key = (policy, seed)
        result["unknown" if key not in expected else "match" if digest == expected[key] else "differ"].append(seed)
    return result


def source_version():
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    return dict(describe=git("describe", "--tags", "--always", "--dirty"), commit=git("rev-parse", "HEAD"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", help="start:stop within the training range 200000:300000")
    parser.add_argument("--policy", choices=tuple(POLICIES), default=POLICY)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--verify", type=Path, metavar="DIR",
                        help="instead of collecting, compare a dataset's contents with the reference manifests")
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify, processes=args.processes)
        print(json.dumps(dict({key: len(seeds) for key, seeds in result.items()}, differ_seeds=result["differ"][:20],
                              unknown_seeds=result["unknown"][:20])))
        if result["differ"] or result["unknown"] or not result["match"]:
            raise SystemExit("verification failed: the dataset is not the reference data")
        return
    if args.seeds is None or args.out is None:
        parser.error("--seeds and --out are required unless --verify is given")
    start, stop = (int(x) for x in args.seeds.split(":"))
    if not TRAINING_SEEDS[0] <= start < stop <= TRAINING_SEEDS[1]:
        parser.error(f"training seeds must lie in {TRAINING_SEEDS[0]}:{TRAINING_SEEDS[1]}")
    args.out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(args.processes) as pool:
        episodes = pool.map(_write, [(seed, args.out, args.policy) for seed in range(start, stop)], chunksize=4)
    outcomes = {}
    for episode in episodes:
        outcomes[episode["outcome"]] = outcomes.get(episode["outcome"], 0)+1
    manifest = dict(policy=args.policy, seeds=[start, stop], source=source_version(), record_dt=0.01,
                    decision_dt=DECISION_DT, outcomes=outcomes, episodes=episodes)
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=1)+"\n")
    print(json.dumps(dict(episodes=len(episodes), outcomes=outcomes, out=str(args.out))))


if __name__ == "__main__":
    main()
