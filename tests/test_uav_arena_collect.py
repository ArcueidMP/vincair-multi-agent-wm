from dataclasses import replace
import hashlib
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.collect import POLICIES, POLICY, _write, collect_episode, verify
from uav_arena.env import ACTION_LIMIT, ArenaEnv
from uav_arena.scenarios import Scenario


@pytest.mark.parametrize("policy", POLICIES)
def test_collected_episodes_are_deterministic_and_replay_exactly(policy):
    episode = collect_episode(200_003, policy)
    assert str(episode["policy"]) == policy
    for key, value in collect_episode(200_003, policy).items():
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
    reference.write_text(json.dumps(dict(policy=POLICY, episodes=[entry])))
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
    references = {}
    for path in sorted(REFERENCES.glob("*.json")):
        manifest = json.loads(path.read_text())
        references.setdefault(manifest["policy"], {}).update({e["seed"]: e["content_sha256"] for e in manifest["episodes"]})
    assert set(references) == set(POLICIES)
    for policy, seeds in references.items():
        assert set(seeds) == set(range(200_000, 202_000)) | set(range(299_000, 299_100))
        assert content_sha256(collect_episode(200_003, policy)) == seeds[200_003]


def test_explore_v2_takes_its_action_statistics_from_the_frozen_planner_and_cost():
    pytest.importorskip("torch")
    pytest.importorskip("stable_worldmodel")
    import inspect
    from uav_arena.collect import V2_HANG_MARGIN, V2_NOISE, V2_PERSISTENCE, V2_SEPARATION, V2_SPEEDS
    from uav_arena.costs import CostWeights
    from uav_arena.env import DECISION_DT
    from uav_arena.planning import ArenaPlanner
    defaults = inspect.signature(ArenaPlanner).parameters
    assert V2_NOISE == defaults["var_scale"].default
    assert V2_PERSISTENCE == pytest.approx(np.exp(-1/defaults["horizon"].default))
    assert (V2_HANG_MARGIN, V2_SEPARATION) == (CostWeights().hang_margin, CostWeights().team_margin)
    assert V2_SPEEDS[1]*DECISION_DT == pytest.approx(ACTION_LIMIT)
    increments = np.diff(collect_episode(200_003)["setpoints"], axis=0)
    assert (np.abs(increments) >= ACTION_LIMIT-1e-9).any(-1).mean() > 0.05      # the planner's limit is reached
    own = (increments-increments.mean(1, keepdims=True)).reshape(len(increments), -1)
    lag1 = np.corrcoef(own[:-1].reshape(-1), own[1:].reshape(-1))[0, 1]
    assert abs(lag1) < 0.2      # each aircraft's own increments are nearly uncorrelated, as U's are (-0.04 on dev)
