from dataclasses import replace
import hashlib
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.collect import _write, collect_episode, verify
from uav_arena.env import ACTION_LIMIT, ArenaEnv
from uav_arena.scenarios import Scenario


def test_collected_episodes_are_deterministic_and_replay_exactly():
    episode = collect_episode(200_003)
    for key, value in collect_episode(200_003).items():
        np.testing.assert_array_equal(value, episode[key])
    assert np.abs(np.diff(episode["setpoints"], axis=0)).max() <= ACTION_LIMIT+1e-12
    layout = Scenario.from_dict(json.loads(str(episode["scenario"])))
    assert layout.payload_mass is None and layout.pillars == () and layout.movers == ()
    env = ArenaEnv(replace(layout, payload_mass=float(episode["payload_mass"])))
    env.reset()
    np.testing.assert_array_equal(env.actions[0]["setpoints"], episode["setpoints"][0])
    for setpoints in episode["setpoints"][1:]:
        env.step(setpoints)
    assert env.outcome == str(episode["outcome"])
    for key in ("payload_pos", "payload_vel", "uav_pos", "uav_vel"):
        np.testing.assert_array_equal(np.stack([r[key] for r in env.log]), episode[key])


def test_verification_compares_contents_not_file_bytes(tmp_path):
    entry = _write((200_003, tmp_path))
    reference = tmp_path/"reference.json"
    reference.write_text(json.dumps(dict(episodes=[entry])))
    path = tmp_path/"episode_200003.npz"
    with np.load(path) as f:
        arrays = {key: f[key] for key in f.files}
    np.savez(path, **arrays)                          # uncompressed: other file bytes, same contents
    assert hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
    assert verify(tmp_path, [reference], processes=1) == dict(match=[200_003], differ=[], unknown=[])
    arrays["payload_pos"][100, 2] += 1e-9
    np.savez_compressed(path, **arrays)
    assert verify(tmp_path, [reference], processes=1)["differ"] == [200_003]
    _write((200_004, tmp_path))
    assert verify(tmp_path, [reference], processes=1)["unknown"] == [200_004]


def test_reference_manifests_cover_the_standard_datasets_and_match_the_collector():
    from uav_arena.collect import REFERENCES, content_sha256
    references = {e["seed"]: e["content_sha256"] for path in sorted(REFERENCES.glob("*.json"))
                  for e in json.loads(path.read_text())["episodes"]}
    assert set(references) == set(range(200_000, 202_000)) | set(range(299_000, 299_100))
    assert content_sha256(collect_episode(200_003)) == references[200_003]
