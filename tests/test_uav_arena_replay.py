import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.scenarios import Condition, Scenario, make_scenario


def test_scenarios_survive_a_dictionary_round_trip():
    scenario = make_scenario(3, Condition("static+moving", 100))
    assert Scenario.from_dict(scenario.to_dict()) == scenario


def _recording(env):
    times, qpos, mocap = zip(*env.frames)
    return dict(time=np.array(times), qpos=np.stack(qpos), mocap_pos=np.stack(mocap),
                meta=dict(scenario=env.scenario.to_dict(), participant="transfer", outcome=env.outcome, time=env.time))


@needs_cc
def test_replay_shows_exactly_the_recorded_states(tmp_path):
    from uav_arena.replay import RecordingArenaEnv, Replay, load, save, straight_transfer
    env = RecordingArenaEnv(make_scenario(1, Condition("static+moving", 50)))
    obs = env.reset()
    policy = straight_transfer(obs)
    for _ in range(8):
        obs, *_ = env.step(policy(obs))
    save(tmp_path/"episode.npz", _recording(env))
    recording = load(tmp_path/"episode.npz")
    assert len(recording["time"]) == len(env.log) == 81
    replay = Replay(recording)
    for k in (0, 40, 80):
        replay.show(k)
        np.testing.assert_array_equal(replay.data.qpos[env._payload_pos], env.log[k]["payload_pos"])
        np.testing.assert_array_equal(replay.data.mocap_pos, recording["mocap_pos"][k])
        np.testing.assert_array_equal(replay.data.qpos[env._uav_pos].reshape(3, 3), env.log[k]["uav_pos"])
    assert replay.data.mocap_pos.shape == (2, 3) and np.ptp(recording["mocap_pos"][:, 0, 1]) > 0


@needs_cc
def test_video_export_writes_every_frame(tmp_path):
    import mujoco
    from uav_arena.replay import RecordingArenaEnv, straight_transfer, write_video
    try:
        mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 32, 32).close()
    except Exception as error:      # no display or GL context on this machine
        pytest.skip(f"offscreen rendering unavailable: {error}")
    cv2 = pytest.importorskip("cv2")
    env = RecordingArenaEnv(make_scenario(1, Condition("static", 0)))
    obs = env.reset()
    policy = straight_transfer(obs)
    for _ in range(5):
        obs, *_ = env.step(policy(obs))
    out = tmp_path/"clip.mp4"
    frames = write_video(_recording(env), out, fps=20, width=320, height=240, hold=0.5)
    capture = cv2.VideoCapture(str(out))
    count = 0
    while capture.read()[0]:
        count += 1
    assert frames == 11+10 and count == frames
