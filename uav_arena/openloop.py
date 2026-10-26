"""Layer 1, open-loop prediction: a predictor's multi-step error on recorded episodes.

    python -m uav_arena.openloop --predictor physics --data data/uav_arena/datasets/explore_v1_val \
        --out data/uav_arena/results/physics_openloop_val.json

Decision k of a recorded episode (`collect.py` format) gives the predictor what the arena would:
the observation recorded `latency` before the decision with the four setpoints then in force, and
the layout without the payload mass. The predictor then predicts the next H decisions under the
setpoints that were actually flown, and is scored against the recorded true states at the end of
each. Any predictor in the planning interface works, including `module:factory` participants;
the mean payload error over the horizon and the four latencies is the participant contract's
model-selection metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import DECISION_DT, RECENT_SETPOINTS, RECORD_DT
from .geometry import mover_radii, pillar_array
from .planning import info_from_observation
from .scenarios import LATENCIES_MS, Scenario

PER_DECISION = round(DECISION_DT/RECORD_DT)


def load_episode(path):
    with np.load(path) as f:
        episode = {key: f[key] for key in f.files}
    episode["layout"] = Scenario.from_dict(json.loads(str(episode["scenario"])))
    return episode


def observation(episode, k, latency_ms):
    """`ArenaEnv.observation()` before decision k, rebuilt from the record."""
    index = max(PER_DECISION*k-round(latency_ms/1000/RECORD_DT), 0)
    recent = slice(max(0, k+1-RECENT_SETPOINTS), k+1)
    layout = episode["layout"]
    mover_pos, mover_vel = layout.mover_states(float(episode["time"][index]))
    obs = {key: episode[key][index].copy() for key in ("uav_pos", "uav_vel", "uav_rot6", "uav_omega", "payload_pos",
                                                       "payload_vel")}
    return dict(obs, time=float(episode["time"][index]), decision_time=float(episode["time"][PER_DECISION*k]),
                latency=latency_ms/1000, goal=np.asarray(layout.goal, dtype=float), pillars=pillar_array(layout),
                mover_pos=mover_pos, mover_vel=mover_vel, mover_radius=mover_radii(layout),
                recent_setpoints=episode["setpoints"][recent].copy(),
                recent_setpoint_times=episode["setpoint_time"][recent].copy())


def evaluate(predictor, episodes, latency_ms, *, horizon=15, stride=1, dtype=torch.float32):
    """Position errors (m) per horizon step, averaged over scored decisions: payload and aircraft."""
    payload, uav = [], []
    for episode in episodes:
        if hasattr(predictor, "reset"):
            predictor.reset()
        complete = (len(episode["time"])-1)//PER_DECISION
        for k in range(complete):
            obs = observation(episode, k, latency_ms)
            if hasattr(predictor, "observe"):
                predictor.observe(obs, episode["layout"])
            if k+horizon > complete or k % stride:
                continue
            info = info_from_observation(obs, obs["recent_setpoints"][-1], dtype=dtype)
            setpoints = torch.as_tensor(episode["setpoints"][k+1:k+1+horizon], dtype=dtype).unsqueeze(0)
            with torch.no_grad():
                prediction = predictor(info, setpoints)
            truth = PER_DECISION*np.arange(k+1, k+1+horizon)
            payload.append(np.linalg.norm(prediction["payload"][0].double().numpy()-episode["payload_pos"][truth], axis=-1))
            uav.append(np.linalg.norm(prediction["uav"][0].double().numpy()-episode["uav_pos"][truth], axis=-1).mean(-1))
    return dict(payload=np.mean(payload, axis=0), uav=np.mean(uav, axis=0), decisions=len(payload))


def main():
    from .run import make_predictor
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictor", required=True, help="physics, adaptive or module:factory")
    parser.add_argument("--predictor-options", default="{}")
    parser.add_argument("--data", type=Path, required=True, help="a collect.py directory")
    parser.add_argument("--seeds", default=None, help="start:stop to use a subset of the directory's episodes")
    parser.add_argument("--latencies", default=",".join(map(str, LATENCIES_MS)))
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--stride", type=int, default=1, help="score every n-th decision")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.predictor == "oracle":
        parser.error("the oracle needs the simulator's true state, which recorded data does not hold")
    paths = sorted(args.data.glob("episode_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if args.seeds:
        start, stop = (int(x) for x in args.seeds.split(":"))
        paths = [p for p in paths if start <= int(p.stem.split("_")[1]) < stop]
    episodes = [load_episode(p) for p in paths]
    predictor = make_predictor(args.predictor, args.workers, json.loads(args.predictor_options))
    rows = {}
    try:
        for latency in (int(x) for x in args.latencies.split(",")):
            result = evaluate(predictor, episodes, latency, horizon=args.horizon, stride=args.stride)
            rows[f"{latency}ms"] = dict(decisions=result["decisions"], payload=result["payload"].tolist(),
                                        uav=result["uav"].tolist(), payload_mean=float(result["payload"].mean()),
                                        uav_mean=float(result["uav"].mean()))
            print(f"{latency:>4} ms: payload {rows[f'{latency}ms']['payload_mean']:.4f} m, "
                  f"aircraft {rows[f'{latency}ms']['uav_mean']:.4f} m over {result['decisions']} decisions", flush=True)
    finally:
        if hasattr(predictor, "close"):
            predictor.close()
    score = float(np.mean([row["payload_mean"] for row in rows.values()]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(config={k: str(v) for k, v in vars(args).items()},
                                        episodes=[p.name for p in paths], selection_score=score, latencies=rows),
                                   indent=1)+"\n")
    print(f"selection score (mean payload error): {score:.4f} m")


if __name__ == "__main__":
    main()
