"""Record arena episodes and replay them in a MuJoCo window or as an MP4.

A recording keeps the MuJoCo positions and the movers at every 10 ms record, plus the
scenario, so a replay rebuilds the exact scene. Replays are kinematic: they set the
recorded positions and never step the physics.

    PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.replay record --participant oracle \
        --seed 0 --condition static+moving/100ms --out data/uav_arena/replays/oracle_s0_moving100.npz
    PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.replay view data/uav_arena/replays/oracle_s0_moving100.npz
    PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.replay video data/uav_arena/replays/oracle_s0_moving100.npz \
        --out data/uav_arena/videos/oracle_s0_moving100.mp4

Participants: `oracle` and `physics` (the shared MPPI), `crazymarl` (an exported actor via
`--actor`), and `transfer` (an open-loop minimum-jerk straight line, a quick demo).
Viewer: P pauses, R restarts, the mouse orbits and zooms; the replay loops.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
import numpy as np

from .env import DECISION_DT, RECORD_DT, ArenaEnv
from .scenarios import CONDITIONS, Scenario, make_scenario

TRAIL_STEP = 0.1                     # s between payload trail markers
TRAIL_RGBA = (0.95, 0.55, 0.1, 0.55)
CABLE_WIDTH = 0.005                  # drawn cable width (m); the physics never sees it


class RecordingArenaEnv(ArenaEnv):
    """An arena episode that also keeps the MuJoCo positions of every 10 ms record."""

    def reset(self):
        self.frames = []
        return super().reset()

    def _record(self):
        super()._record()
        self.frames.append((float(self.data.time), self.data.qpos.copy(), self.data.mocap_pos.copy()))


def straight_transfer(obs, duration=4.0):
    """Open-loop reference: shift the hover formation to the goal along a minimum-jerk straight line."""
    start, shift = obs["recent_setpoints"][-1], obs["goal"]-obs["payload_pos"]

    def policy(obs):
        s = min(1.0, (obs["decision_time"]+DECISION_DT)/duration)
        return start+shift*(10*s**3-15*s**4+6*s**5)
    return policy


def record(scenario, participant, *, actor=None, task=None, workers=None):
    env = RecordingArenaEnv(scenario)
    obs = env.reset()
    close = None
    if participant in ("oracle", "physics"):
        from .planning import ArenaCost, ArenaPlanner, PhysicsPredictor, TrueSimPredictor
        predictor = (TrueSimPredictor if participant == "oracle" else PhysicsPredictor)(workers)
        close = predictor.close
        planner = ArenaPlanner(ArenaCost(predictor))
        planner.reset()
        advance = lambda obs: env.step(planner.act(obs, env))
    elif participant == "crazymarl":
        from .crazymarl_policy import CrazyMARLActor, CrazyMARLPilot
        pilot = CrazyMARLPilot(CrazyMARLActor(actor), task)
        pilot.reset(env)
        advance = lambda obs: env.step_motors(pilot)
    elif participant == "transfer":
        policy = straight_transfer(obs)
        advance = lambda obs: env.step(policy(obs))
    else:
        raise ValueError(f"Unknown participant {participant}")
    try:
        while not env.done:
            obs, _, _, info = advance(obs)
    finally:
        if close:
            close()
    times, qpos, mocap = zip(*env.frames)
    meta = dict(scenario=scenario.to_dict(), participant=participant, outcome=info["outcome"], time=info["time"],
                actor=None if actor is None else str(actor))
    return dict(time=np.array(times), qpos=np.stack(qpos), mocap_pos=np.stack(mocap), meta=meta)


def save(path, recording):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, time=recording["time"], qpos=recording["qpos"], mocap_pos=recording["mocap_pos"],
                        meta=np.array(json.dumps(recording["meta"])))


def load(path):
    with np.load(path) as data:
        return dict(time=data["time"], qpos=data["qpos"], mocap_pos=data["mocap_pos"], meta=json.loads(str(data["meta"])))


class Replay:
    """The recorded scene with any frame shown on demand."""

    def __init__(self, recording):
        self.recording = recording
        self.scenario = Scenario.from_dict(recording["meta"]["scenario"])
        self.env = ArenaEnv(self.scenario)
        self.model, self.data = self.env.model, self.env.data
        self.model.tendon_width[:] = np.maximum(self.model.tendon_width, CABLE_WIDTH)
        self.model.vis.headlight.ambient[:] = 0.35        # brighter than the simulation scene; display only
        self.model.vis.headlight.diffuse[:] = 0.6
        self.payload = self.recording["qpos"][:, self.env._payload_pos]
        stride = max(1, round(TRAIL_STEP/RECORD_DT))
        self.trail_indices = np.arange(0, len(self.recording["time"]), stride)

    def __len__(self):
        return len(self.recording["time"])

    def show(self, index):
        self.data.qpos[:] = self.recording["qpos"][index]
        self.data.mocap_pos[:] = self.recording["mocap_pos"][index]
        mujoco.mj_forward(self.model, self.data)

    def mark_trail(self, scene, index):
        """Small spheres where the payload was, every TRAIL_STEP up to frame `index`."""
        for point in self.payload[self.trail_indices[self.trail_indices <= index]]:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.012, 0., 0.]),
                                point, np.eye(3).reshape(-1), np.array(TRAIL_RGBA, dtype=np.float32))
            scene.ngeom += 1

    def caption(self, index):
        meta, scenario = self.recording["meta"], self.scenario
        lines = [f"{meta['participant']}  |  seed {scenario.seed}, {scenario.condition.name}, "
                 f"payload {1000*scenario.payload_mass:.1f} g",
                 f"t = {self.recording['time'][index]:4.1f} s"]
        if index == len(self)-1:
            lines[-1] += f"  |  {meta['outcome']}"
        return lines


def set_overview(camera, scenario):
    start, goal = np.asarray(scenario.start), np.asarray(scenario.goal)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (start+goal)/2+np.array([0.0, 0.0, 0.1])
    camera.distance = max(3.0, 1.4*np.linalg.norm(goal-start))
    camera.azimuth, camera.elevation = 120.0, -28.0


def view(recording, speed=1.0):
    import mujoco.viewer
    replay = Replay(recording)
    times = recording["time"]
    state = dict(paused=False, restart=False)

    def on_key(key):
        if key == ord("P"):
            state["paused"] = not state["paused"]
        elif key == ord("R"):
            state["restart"] = True

    print("\n".join(replay.caption(len(replay)-1)), flush=True)
    print("Mouse: orbit / zoom.  P: pause.  R: restart.  Close the window to exit.", flush=True)
    with mujoco.viewer.launch_passive(replay.model, replay.data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            set_overview(viewer.cam, replay.scenario)
        clock, index = time.monotonic(), 0
        while viewer.is_running():
            now = time.monotonic()
            if state["restart"]:
                clock, index, state["restart"] = now, 0, False
            if state["paused"]:
                clock = now-(times[index]-times[0])/speed
            else:
                target = times[0]+(now-clock)*speed
                index = int(np.searchsorted(times, target, side="right"))-1
                if index >= len(replay)-1 and target > times[-1]+1.5:      # hold the last frame, then loop
                    clock, index = now, 0
                index = min(max(index, 0), len(replay)-1)
            with viewer.lock():
                replay.show(index)
                viewer.user_scn.ngeom = 0
                replay.mark_trail(viewer.user_scn, index)
            viewer.sync()
            time.sleep(1/60)


class _Encoder:
    """H.264 through the system ffmpeg, else OpenCV's MPEG-4 writer."""

    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        self.process = self.writer = None
        if ffmpeg:
            self.process = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(self.path)],
                stdin=subprocess.PIPE)
        else:
            import cv2
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def write(self, frame):
        if self.process:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.writer.write(np.ascontiguousarray(frame[:, :, ::-1]))

    def close(self):
        if self.process:
            self.process.stdin.close()
            if self.process.wait():
                raise RuntimeError("ffmpeg failed to write the video")
        else:
            self.writer.release()


def write_video(recording, out, *, fps=50, width=1280, height=720, speed=1.0, hold=1.0):
    import cv2
    replay = Replay(recording)
    model = replay.model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    times = recording["time"]
    stamps = np.arange(times[0], times[-1]+1e-9, speed/fps)
    indices = np.clip(np.searchsorted(times, stamps-1e-9), 0, len(replay)-1)
    indices = np.concatenate([indices, np.full(round(hold*fps), len(replay)-1)])
    camera = mujoco.MjvCamera()
    set_overview(camera, replay.scenario)
    renderer = mujoco.Renderer(model, height, width)
    encoder = _Encoder(out, width, height, fps)
    try:
        for index in indices:
            replay.show(index)
            renderer.update_scene(replay.data, camera=camera)
            replay.mark_trail(renderer.scene, index)
            frame = renderer.render().copy()
            for row, line in enumerate(replay.caption(index)):
                cv2.putText(frame, line, (24, 44+38*row), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                            cv2.LINE_AA)
            encoder.write(frame)
    finally:
        encoder.close()
        renderer.close()
    return len(indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record", help="run one episode and save its frames")
    rec.add_argument("--participant", choices=("oracle", "physics", "crazymarl", "transfer"), required=True)
    rec.add_argument("--seed", type=int, required=True, help="scenario seed (dev: 0-19, test: 1000-1049)")
    rec.add_argument("--condition", choices=[c.name for c in CONDITIONS], default="static/0ms")
    rec.add_argument("--actor", type=Path, help="CrazyMARL actor weights (.npz)")
    rec.add_argument("--task", type=Path, default=Path("data/uav_arena/crazymarl/task.npz"))
    rec.add_argument("--workers", type=int, default=None)
    rec.add_argument("--out", type=Path, required=True)
    show = commands.add_parser("view", help="replay a recording in a MuJoCo window")
    show.add_argument("recording", type=Path)
    show.add_argument("--speed", type=float, default=1.0)
    film = commands.add_parser("video", help="render a recording to MP4")
    film.add_argument("recording", type=Path)
    film.add_argument("--out", type=Path, required=True)
    film.add_argument("--fps", type=int, default=50)
    film.add_argument("--width", type=int, default=1280)
    film.add_argument("--height", type=int, default=720)
    film.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "record":
        if args.participant == "crazymarl" and args.actor is None:
            parser.error("--actor is required for crazymarl")
        condition = next(c for c in CONDITIONS if c.name == args.condition)
        recording = record(make_scenario(args.seed, condition), args.participant, actor=args.actor, task=args.task,
                           workers=args.workers)
        save(args.out, recording)
        print(f"{args.participant} seed {args.seed} {args.condition}: {recording['meta']['outcome']} at "
              f"{recording['meta']['time']:.1f} s, {len(recording['time'])} frames -> {args.out}")
    elif args.command == "view":
        view(load(args.recording), speed=args.speed)
    else:
        frames = write_video(load(args.recording), args.out, fps=args.fps, width=args.width, height=args.height,
                             speed=args.speed)
        print(f"{frames} frames at {args.fps} fps -> {args.out}")


if __name__ == "__main__":
    main()
