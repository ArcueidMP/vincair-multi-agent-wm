"""Isolate VMAS angular-friction discretization under a constant finite torque.

This single-body simulator diagnostic is not a new manipulation environment or
a change to PushT. A torque below the configured friction cap leaves omega of
order sub_dt after each semi-implicit step; accumulating it creates timestep-
dependent creep. Above-cap torque is a nontrivial-motion control.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from vmas.simulator.core import Agent, Box, World


@torch.no_grad()
def probe(substeps, torque_fraction=.5, steps=60):
    if substeps < 1 or torque_fraction <= 0 or steps < 1:
        raise ValueError("positive substeps, torque fraction and steps required")
    world = World(batch_dim=1, device="cpu", dt=.1, substeps=substeps, drag=.25)
    agent = Agent(name="finite_torque_probe", movable=False, rotatable=True,
                  shape=Box(length=1., width=1.), angular_friction=1., collide=False)
    world.add_agent(agent)
    inertia = float(agent.moment_of_inertia)
    torque = torque_fraction * inertia  # cap = angular_friction * inertia.
    agent.state.torque[:] = torque
    trajectory = []
    for _ in range(steps):
        world.step()
        trajectory.append([float(agent.state.rot[0, 0]), float(agent.state.ang_vel[0, 0])])
    sub_dt = world.dt / substeps
    return {"substeps": substeps, "dt": world.dt, "sub_dt": sub_dt,
            "steps": steps, "torque_fraction": torque_fraction,
            "finite_external_torque": torque, "friction_torque_cap": inertia,
            "angle_rad": trajectory[-1][0], "angular_velocity": trajectory[-1][1],
            # VMAS computes friction before applying per-env-step drag. The
            # first substep of every later env step is therefore reduced by
            # drag*a*h; subsequent substeps return to a*h. This formula is for
            # at least two substeps and the declared below-cap constant torque.
            "angle_if_below_cap_creep_formula": torque_fraction * sub_dt ** 2 *
                (steps * substeps - .25 * (steps - 1))
                if torque_fraction < 1 and substeps >= 2 else None,
            "trajectory_theta_omega": trajectory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substeps", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("data/novelty/friction_resolution_probe.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [probe(ss, fraction, args.steps) for fraction in (.5, 1.5) for ss in args.substeps]
    import vmas.simulator.core
    report = {"versions": {key: str(importlib.metadata.version(key)) for key in ("torch", "vmas")},
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), Path(vmas.simulator.core.__file__))},
              "rows": rows,
              "interpretation": "Isolated constant-torque diagnostic: below-cap creep scales with sub_dt. This explains a possible numerical mechanism; it does not analytically prove the entire contact-rich PushT response has zero continuum limit."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "trajectory_theta_omega"} for row in rows], indent=2))


if __name__ == "__main__":
    main()
