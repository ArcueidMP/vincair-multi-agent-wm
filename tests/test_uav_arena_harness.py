import argparse
import dataclasses
import json
import multiprocessing as mp
import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_worldmodel")
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena import harness
from uav_arena.collect import content_sha256
from uav_arena.env import ArenaEnv
from uav_arena.harness import CHANGE_AIRCRAFT, CHANGE_KIND, CHANGES, changes, compute_truth, score_episode
from uav_arena.openloop import observation
from uav_arena.planning import ArenaCost, ArenaPlanner, KinematicPredictor, PhysicsPredictor, TrueSimPredictor
from uav_arena.record import (RecordingPlanner, episode_name, load, physics_state, record_episode, regenerate_pool,
                              true_scenario)
from uav_arena.run import run_episode
from uav_arena.scenarios import Condition, make_scenario

SAMPLES, HORIZON = 16, 5        # a small planner: the harness reads its settings from the recording
CONDITIONS = (Condition("static", 0), Condition("static+moving", 100))


def short(condition):
    return dataclasses.replace(make_scenario(3, condition), time_limit=1.6)


def small_planner(predictor):
    return ArenaPlanner(ArenaCost(predictor), samples=SAMPLES, horizon=HORIZON)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """Two short recorded U episodes with their ground truth, and what the planner saw while flying them."""
    states = tmp_path_factory.mktemp("states")
    oracle = TrueSimPredictor(workers=2)
    live, files = {}, []
    try:
        for condition in CONDITIONS:
            scenario = short(condition)
            recorder = RecordingPlanner(small_planner(oracle))
            seen, act = [], recorder.act

            def spy(obs, env):
                k = len(recorder.plans)
                setpoints = act(obs, env)
                seen.append(dict(obs=obs, candidates=recorder.tap.cost.candidates[0].clone(),
                                 state=env.physics_state() if k % recorder.every == 0 else None))
                return setpoints
            recorder.act = spy
            record, arrays = record_episode(scenario, recorder)
            np.savez_compressed(states/episode_name(scenario), **arrays)
            files.append(dict(file=episode_name(scenario), content_sha256=content_sha256(arrays)))
            live[condition] = dict(record=record, seen=seen, plain=run_episode(scenario, small_planner(oracle)))
    finally:
        oracle.close()
    (states/"manifest.json").write_text(json.dumps(dict(episodes=files)))
    compute_truth(states, 2)
    return states, live


def episodes(states):
    for path in harness.episode_paths(states):
        with np.load(states/harness.TRUTH/path.name) as f:
            yield load(path), {key: f[key] for key in f.files}


def scored(states, predictor):
    rows = []
    with mp.get_context("fork").Pool(2) as pool:
        for episode, truth in episodes(states):
            rows += score_episode(predictor, pool, episode, truth, float(episode["config"]["temperature"]))
    return {key: np.stack([r[key] for r in rows]) for key in rows[0]}


def test_recorded_episodes_reproduce_run_py_records(recorded):
    _, live = recorded
    for condition in CONDITIONS:
        record, plain = live[condition]["record"], live[condition]["plain"]
        assert {k: v for k, v in record.items() if not k.startswith("plan_seconds")} == \
            {k: v for k, v in plain.items() if not k.startswith("plan_seconds")}


def test_recordings_rebuild_what_the_planner_saw_bit_for_bit(recorded):
    states, live = recorded
    for condition in CONDITIONS:
        episode, seen = load(states/episode_name(short(condition))), live[condition]["seen"]
        env = ArenaEnv(true_scenario(episode))
        sequence = episode["setpoints"][-1]+np.cumsum(np.full((4, 3, 3), 0.04), axis=0)
        for k, step in enumerate(seen):
            rebuilt = observation(episode, k, condition.latency_ms)
            assert rebuilt.keys() == step["obs"].keys()
            for key in rebuilt:
                np.testing.assert_array_equal(rebuilt[key], step["obs"][key], err_msg=key)
            if step["state"] is None:
                continue
            q = k//5
            assert torch.equal(regenerate_pool(episode, q), step["candidates"])
            env.set_physics_state(step["state"])
            expected = env.simulate(sequence)
            env.set_physics_state(physics_state(env, episode, q))
            for key, value in env.simulate(sequence).items():
                np.testing.assert_array_equal(value, expected[key])


def test_changes_alter_one_aircraft_and_are_shared_across_latencies():
    plan = torch.zeros(HORIZON, 9)
    changed = changes(plan, 3, "static", 10)
    assert changed.shape == (CHANGES, HORIZON, 9) == (42, HORIZON, 9)
    moved = changed.reshape(CHANGES, HORIZON, 3, 3).abs().sum((1, 3)) > 0
    np.testing.assert_array_equal(moved.numpy(), np.eye(3, dtype=bool)[CHANGE_AIRCRAFT])
    offsets = changed[torch.from_numpy(CHANGE_KIND == 1)].reshape(-1, HORIZON, 3, 3).sum(1)   # offset at the end
    assert torch.allclose(offsets.norm(dim=-1).amax(-1), torch.tensor(0.10))
    assert torch.equal(changed, changes(plan, 3, "static", 10))
    assert not torch.equal(changed, changes(plan, 3, "static", 15))


def test_the_oracle_scores_zero_on_every_layer(recorded):
    states, _ = recorded
    oracle = TrueSimPredictor(workers=2)
    try:
        rows = scored(states, oracle)
    finally:
        oracle.close()
    for key in ("l1_payload", "l1_uav", "l2_error", "l3_regret", "l3_costly"):
        assert np.abs(rows[key]).max() == 0.0, key
    np.testing.assert_array_equal(rows["l3"], np.tile([0., 0., 1., 1.], (len(rows["l3"]), 1)))
    assert rows["l2_effect"][:, CHANGE_KIND == 0, 0].mean() > 0.005      # the changes do move the payload


def test_physics_with_the_true_mass_scores_near_zero(recorded):
    states, _ = recorded
    for episode, truth in episodes(states):
        predictor = PhysicsPredictor(workers=2, nominal_mass=float(episode["payload_mass"]))
        try:
            with mp.get_context("fork").Pool(2) as pool:
                rows = score_episode(predictor, pool, episode, truth, float(episode["config"]["temperature"]))
        finally:
            predictor.close()
        assert max(r["l1_payload"].max() for r in rows) < 1e-5
        assert max(r["l2_error"].max() for r in rows) < 1e-5
        assert max(abs(r["l3_regret"]) for r in rows) < 1e-4


class Spy(KinematicPredictor):
    def __init__(self):
        self.masses, self.states, self.shapes, self.info_keys = [], [], [], set()

    def observe(self, obs, layout):
        self.masses.append(layout.payload_mass)

    def set_state(self, scenario, state):
        self.states.append(state)

    def __call__(self, info, setpoints):
        self.shapes.append(tuple(setpoints.shape[:2]))
        self.info_keys |= set(info)
        return super().__call__(info, setpoints)


class Claims(Spy):
    privileged = True       # observing predictors never get the true state, whatever they claim


def test_predictors_get_observations_only(recorded):
    states, _ = recorded
    for spy in (Spy(), Claims()):
        scored(states, spy)
        assert spy.masses and set(spy.masses) == {None}
        assert spy.states == []
        assert spy.info_keys == {"goal", "pillars", "mover_pos", "mover_vel", "mover_radius", "forecast_offset", "anchor",
                                 "uav_pos", "uav_vel", "payload_pos", "payload_vel"}


def test_call_shapes(recorded):
    states, _ = recorded
    spy = Spy()
    rows = scored(states, spy)
    questions, batch = len(rows["l3_regret"]), (1, 1+CHANGES+SAMPLES)       # (1, 555) with the frozen planner
    assert spy.shapes == [batch]*(questions+len(CONDITIONS))     # plus a repeat per episode that checks determinism

    class StochasticSpy(Spy):
        stochastic = True
    spy = StochasticSpy()
    scored(states, spy)
    assert spy.shapes == ([(1, 1)]*(1+CHANGES)+[batch]*harness.REPEATS)*questions


class Noisy(KinematicPredictor):
    """Kinematic predictions plus noise that does not depend on the action."""
    stochastic = True

    def __call__(self, info, setpoints):
        prediction = super().__call__(info, setpoints)
        return {key: value+0.01*torch.randn(value.shape) for key, value in prediction.items()}


class Undeclared(Noisy):
    stochastic = False


def test_action_independent_noise_leaves_layer_2_unchanged(recorded):
    states, _ = recorded
    noisy, plain = scored(states, Noisy()), scored(states, KinematicPredictor())
    np.testing.assert_allclose(noisy["l2_error"], plain["l2_error"], atol=1e-6)
    assert np.abs(noisy["l1_payload"]-plain["l1_payload"]).max() > 1e-3


def test_undeclared_randomness_is_rejected(recorded):
    states, _ = recorded
    with pytest.raises(ValueError, match="stochastic = True"):
        scored(states, Undeclared())


def test_score_writes_the_analysis_quantities(recorded, tmp_path):
    states, _ = recorded
    out = tmp_path/"kinematic.json"
    harness.score(argparse.Namespace(predictor="uav_arena.planning:KinematicPredictor", predictor_options="{}",
                                     states=states, episodes=None, workers=2, out=out))
    result = json.loads(out.read_text())
    assert set(result["summary"]) == {"all", "static/0ms", "static+moving/100ms"}
    row = result["summary"]["all"]
    assert row["L1"] > 0 and row["L2"] > 0 and len(row["L1_steps"]) == HORIZON
    assert result["provenance"]["stochastic"] is False and result["provenance"]["privileged"] is False
    with np.load(out.with_suffix(".npz")) as f:
        assert f["l2_error"].shape == (row["questions"], CHANGES, 3)
        costly, reliable, regret = f["l3_costly"], f["l3_reliable"], f["l3_regret"]
    np.testing.assert_array_equal(costly, regret > harness.COSTLY)     # one call per question: an indicator
    assert row["L3"] == pytest.approx(costly[reliable].mean()) and row["L3_all"] == pytest.approx(costly.mean())
    assert row["unreliable"] == pytest.approx(1-reliable.mean())
