"""How well a dataset covers the setpoints U commands and the states it flies through.

    python -m uav_arena.coverage --data data/uav_arena/datasets/explore_v2_val --states data/uav_arena/states/dev

Each quantity is compared by the share of U's values beyond the dataset's 99th percentile (below
its 1st percentile for the closest pair of setpoints). Quantities per aircraft and decision:
- increment: how far its commanded setpoint moves over one decision;
- differs: how far that increment differs from the team's mean increment;
- jerk: how much the increment changes from the previous decision;
- clipped: whether some axis of the increment is at the planner's limit (a share, not a tail);
- separation (per decision): the closest distance between two aircraft's setpoints;
- payload speed, swing (the payload's horizontal offset from the aircraft centroid), formation
  radius (the aircraft's mean distance from their centroid) and the closest distance between two
  aircraft, at decision times;
- plan increment and plan differs: increment and differs within U's 15-decision plans, against
  the dataset's flown increments, since the planner queries the model on such sequences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import ACTION_LIMIT, DECISION_DT, RECORD_DT

PER_DECISION = round(DECISION_DT/RECORD_DT)
PAIRS = ((0, 1), (0, 2), (1, 2))
LOWER_TAIL = ("separation", "aircraft_separation")
UNITS = dict(payload_speed="cm/s")


def setpoint_quantities(setpoints):
    """Quantities of a commanded setpoint sequence (D + 1, 3, 3), the hover setpoint first."""
    increment = np.diff(setpoints, axis=0)
    norm = lambda x: np.linalg.norm(x, axis=-1).reshape(-1)
    return dict(increment=norm(increment), differs=norm(increment-increment.mean(1, keepdims=True)),
                jerk=norm(np.diff(increment, axis=0)),
                clipped=(np.abs(increment) >= ACTION_LIMIT-1e-6).any(-1).reshape(-1).astype(float),
                separation=np.min([np.linalg.norm(setpoints[1:, i]-setpoints[1:, j], axis=-1) for i, j in PAIRS], 0))


def state_quantities(episode):
    """Payload speed, swing, formation radius and closest aircraft pair at each decision of an episode."""
    index = PER_DECISION*np.arange(len(episode["setpoints"])-1)
    index = index[index < len(episode["time"])]
    uav, payload = episode["uav_pos"][index], episode["payload_pos"][index]
    centroid = uav.mean(1)
    return dict(payload_speed=np.linalg.norm(episode["payload_vel"][index], axis=-1),
                swing=np.linalg.norm((payload-centroid)[:, :2], axis=-1),
                radius=np.linalg.norm(uav-centroid[:, None], axis=-1).mean(1),
                aircraft_separation=np.min([np.linalg.norm(uav[:, i]-uav[:, j], axis=-1) for i, j in PAIRS], 0))


def plan_quantities(plan):
    """Increment and differs within 15-decision plans (D, H, 9)."""
    increment = plan.reshape(*plan.shape[:2], 3, 3).astype(float)
    norm = lambda x: np.linalg.norm(x, axis=-1).reshape(-1)
    return dict(plan_increment=norm(increment), plan_differs=norm(increment-increment.mean(2, keepdims=True)))


def quantities(paths, plans=False):
    pooled = {}
    for path in paths:
        with np.load(path) as f:
            episode = {key: f[key] for key in ("time", "uav_pos", "payload_pos", "payload_vel", "setpoints")
                       + (("plan",) if plans else ())}
        parts = dict(setpoint_quantities(episode["setpoints"]), **state_quantities(episode))
        if plans:
            parts.update(plan_quantities(episode["plan"]))
        for key, value in parts.items():
            pooled.setdefault(key, []).append(value)
    return {key: np.concatenate(value) for key, value in pooled.items()}


def compare(data, flown):
    """Per quantity: the dataset's and U's median and 99th (1st) percentile, and U's share beyond it."""
    rows = {}
    for key, values in flown.items():
        reference = data[key.removeprefix("plan_")]
        if key == "clipped":
            rows[key] = dict(data_share=float(reference.mean()), flown_share=float(values.mean()))
            continue
        q = 0.01 if key in LOWER_TAIL else 0.99
        tail = float(np.quantile(reference, q))
        beyond = values < tail if key in LOWER_TAIL else values > tail
        rows[key] = dict(data_median=float(np.median(reference)), data_tail=tail, flown_median=float(np.median(values)),
                         flown_tail=float(np.quantile(values, q)), beyond=float(beyond.mean()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="a collect.py directory")
    parser.add_argument("--states", type=Path, required=True, help="a record.py directory of U's episodes")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    data = quantities(sorted(args.data.glob("episode_*.npz")))
    flown = quantities(sorted(args.states.glob("episode_*.npz")), plans=True)
    rows = compare(data, flown)
    for key, row in rows.items():
        if key == "clipped":
            print(f"{key:>19}: data {row['data_share']:.1%}  U {row['flown_share']:.1%}")
        else:
            unit = UNITS.get(key, "cm")
            print(f"{key:>19}: data median {100*row['data_median']:5.1f} {unit}, tail {100*row['data_tail']:5.1f} {unit} | "
                  f"U median {100*row['flown_median']:5.1f} {unit}, tail {100*row['flown_tail']:5.1f} {unit} | "
                  f"U beyond the data's tail {row['beyond']:.1%}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(data=str(args.data), states=str(args.states), rows=rows), indent=1)+"\n")


if __name__ == "__main__":
    main()
