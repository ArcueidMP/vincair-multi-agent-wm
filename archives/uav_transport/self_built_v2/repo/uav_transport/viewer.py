"""Minimal MuJoCo GUI replay of a recorded transport mission."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import time

import mujoco
import mujoco.viewer

from .plant import TransportConfig, TransportEnv


DEFAULT_TRACE = (Path(__file__).resolve().parents[1] / "data/uav_transport/"
                 "mission_v2_wm_small_seed0_mass026_donor0_route0.pkl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, default=DEFAULT_TRACE,
                        help="Recorded local mission pickle")
    args = parser.parse_args()
    if not args.trace.is_file():
        parser.error(f"Mission trace does not exist: {args.trace}")
    with args.trace.open("rb") as stream:
        mission = pickle.load(stream)
    env = TransportEnv(TransportConfig(**mission["plant_config"]))
    states = mission["trace"]["integration_states"]
    spec = mission["initial_snapshot"].state_spec
    if states.shape[1] != mujoco.mj_stateSize(env.model, spec):
        raise ValueError("Recorded state size differs from the viewer model")
    paused, restart = False, False

    def on_key(key):
        nonlocal paused, restart
        if key == ord("P"):
            paused = not paused
        elif key == ord("R"):
            restart = True

    print(f"Replaying {args.trace.name}", flush=True)
    print("Mouse: rotate / zoom. P: pause / resume. R: restart. Close window to exit.",
          flush=True)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = (0.25, 0., 1.45)
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 110
            viewer.cam.elevation = -20
        print("Viewer ready; replay loops at real-time speed.", flush=True)
        index = 0
        while viewer.is_running():
            started = time.monotonic()
            if restart:
                index, restart = 0, False
            with viewer.lock():
                mujoco.mj_setState(env.model, env.data, states[index], spec)
                mujoco.mj_forward(env.model, env.data)
            viewer.sync()
            if not paused:
                index = (index + 1) % len(states)
            time.sleep(max(0., env.config.control_dt - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
