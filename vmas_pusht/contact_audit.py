"""Recount actual VMAS helper contacts in immutable recorded trajectories.

This only measures stored states in a temporary environment. It does not run
physics, create new trajectories, or alter the original success/response data.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import vmas.simulator.core

from vmas_pusht.boundary import surface_gaps
from vmas_pusht.rebrace import helper_contact_forces
from vmas_pusht.scenario import make


@torch.no_grad()
def recount(path):
    path = Path(path)
    data = torch.load(path, map_location="cpu")
    if "trajectories" not in data:
        raise ValueError("Recorded trajectories are required; no physics rerun is inferred")
    installed_source = Path(vmas.simulator.core.__file__)
    digest = hashlib.sha256(installed_source.read_bytes()).hexdigest()
    original = data["metadata"]["source_sha256"]["installed_vmas/simulator/core.py"]
    if digest != original:
        raise ValueError("VMAS engine source differs from the recorded experiment")
    rows, trajectory = data["rows"], data["trajectories"]
    n = len(rows)
    env = make(num_envs=n, device="cpu", seed=101)
    durations = torch.tensor([r["duration"] for r in rows])
    counts = torch.zeros(n, 4)
    for t in range(1, int(durations.max()) + 1):
        for j, entity in enumerate((*env.world.agents, env.scenario.bar, env.scenario.stem)):
            s = trajectory[:, t, j * 6:(j + 1) * 6]
            entity.state.pos, entity.state.vel = s[:, :2].clone(), s[:, 2:4].clone()
            entity.state.rot, entity.state.ang_vel = s[:, 4:5].clone(), s[:, 5:6].clone()
        geometric = surface_gaps(env)[:, 1] <= .005
        real = helper_contact_forces(env).norm(dim=-1) >= .001
        counts += torch.stack((geometric.any(1), geometric.all(1), real.any(1), real.all(1)), 1) * (durations >= t)[:, None]
    fractions = counts / durations[:, None]
    groups = []
    keys = sorted(set((r["mode"], r["substeps"], r["angle_deg"]) for r in rows))
    names = ("old_geometric_any", "old_geometric_dual", "engine_force_any", "engine_force_dual")
    for mode, ss, angle in keys:
        idx = [i for i, r in enumerate(rows) if (r["mode"], r["substeps"], r["angle_deg"]) == (mode, ss, angle)]
        means = fractions[idx].mean(0)
        groups.append(dict(mode=mode, substeps=ss, angle_deg=angle, rows=len(idx),
                           **dict(zip(names, means.tolist()))))
    return {"source_file": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": n, "preload": data["metadata"].get("preload", 0.),
            "measurement_only_no_new_rollout": True,
            "minimum_normal_force": .001, "old_gap_tolerance": .005,
            "metric": "Per-row fraction of valid post-step samples, then equal mean over rows",
            "groups": groups}, fractions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", default=["data/novelty/abstraction_physics.pt",
                                                      "data/novelty/abstraction_physics_preload.pt"])
    parser.add_argument("--out", type=Path, default=Path("data/novelty/abstraction_physics_contact_audit.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    groups, raw = [], []
    for path in args.data:
        group, fractions = recount(path)
        groups.append(group); raw.append(fractions)
        print(json.dumps({"file": path, "rows": group["rows"]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_file = args.out.with_suffix(".pt")
    torch.save({"fractions": raw, "columns": ["old_geometric_any", "old_geometric_dual", "engine_force_any", "engine_force_dual"]}, raw_file)
    source = [Path(__file__), Path(__file__).with_name("rebrace.py"), Path(vmas.simulator.core.__file__)]
    report = {"groups": groups, "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
              "raw_file": str(raw_file), "raw_sha256": hashlib.sha256(raw_file.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
