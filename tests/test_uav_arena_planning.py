import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_worldmodel")
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.costs import CONSTRAINTS, clearances, constraint_values
from uav_arena.env import ArenaEnv
from uav_arena.geometry import team_clearances
from uav_arena.planning import (ArenaCost, ArenaPlanner, ConstrainedArenaCost, KinematicPredictor,
                                TrueSimPredictor, info_from_observation)
from uav_arena.run import run_episode
from uav_arena.scenarios import Condition, Pillar, Scenario, make_scenario


def open_field(pillars=()):
    return Scenario(seed=0, condition=Condition("static", 0), start=(-1.0, 0., 1.2), goal=(1.0, 0.2, 1.2),
                    pillars=tuple(pillars))


def test_batched_clearances_match_the_scoring_geometry():
    rng = np.random.default_rng(1)
    pillars = np.array([[0., 0., 0.12], [0.6, -0.4, 0.15]])
    movers, radius = rng.uniform(-1, 1, size=(2, 3))+[0, 0, 1.2], np.array([0.15, 0.2])
    for _ in range(20):
        payload = rng.uniform(-1, 1, size=3)+[0, 0, 1.0]
        uav = payload+rng.uniform(-0.3, 0.3, size=(3, 3))+[0, 0, 0.5]
        expected_pillars = team_clearances(uav, payload, pillars, np.zeros((0, 3)), [])
        expected_movers = team_clearances(uav, payload, np.zeros((0, 3)), movers, radius)
        pillar, mover = clearances(torch.tensor(uav)[None], torch.tensor(payload)[None], torch.tensor(pillars),
                                   torch.tensor(movers)[None], torch.tensor(radius))
        assert float(pillar[0]) == pytest.approx(min(expected_pillars.values()), abs=1e-12)
        assert float(mover[0]) == pytest.approx(min(expected_movers.values()), abs=1e-12)


def test_constraints_flag_a_planned_pass_through_a_pillar():
    env = ArenaEnv(open_field(pillars=[Pillar(0., 0., 0.15)]))
    obs = env.reset()
    info = info_from_observation(obs, obs["uav_pos"])
    path = torch.linspace(-1.0, 1.0, 15, dtype=torch.float32)
    payload = torch.stack([path, torch.zeros(15), torch.full((15,), 1.2)], -1)[None]
    uav = payload[..., None, :]+torch.tensor(obs["uav_pos"]-obs["payload_pos"], dtype=torch.float32)
    through = constraint_values(dict(uav=uav, payload=payload), info)
    around = constraint_values(dict(uav=uav+torch.tensor([0., 1.5, 0.]), payload=payload+torch.tensor([0., 1.5, 0.])), info)
    pillars = CONSTRAINTS.index("pillars")
    assert through[0, pillars] > 0 > around[0, pillars]


def test_simulate_reproduces_stepping_exactly():
    env = ArenaEnv(make_scenario(1, Condition("static+moving", 50)))
    obs = env.reset()
    for _ in range(5):
        obs, *_ = env.step(obs["uav_pos"]+[0.08, 0., 0.])
    state = env.physics_state()
    sequence = np.stack([obs["uav_pos"]+[0.1*(k+1), 0.02*k, 0.] for k in range(6)])
    rollout = env.simulate(sequence)
    env.set_physics_state(state)
    env.log[-1]["uav_pos"] = env.data.qpos[env._uav_pos].reshape(3, 3).copy()
    for k, setpoints in enumerate(sequence):
        env.step(setpoints)
        np.testing.assert_array_equal(env.true_state()["payload_pos"], rollout["payload"][k])
        np.testing.assert_array_equal(env.true_state()["uav_pos"], rollout["uav"][k])


def test_oracle_rollouts_equal_local_simulation():
    env = ArenaEnv(make_scenario(2, Condition("static", 100)))
    obs = env.reset()
    predictor = TrueSimPredictor(workers=2)
    try:
        predictor.set_state(env.scenario, env.physics_state())
        info = info_from_observation(obs, obs["uav_pos"], dtype=torch.float64)
        candidates = torch.as_tensor(np.random.default_rng(0).normal(0, 0.05, size=(1, 3, 5, 9)))
        prediction, _ = ArenaCost(predictor).predict(info, candidates)
        for s in range(3):
            setpoints = obs["uav_pos"].reshape(-1)+np.cumsum(np.clip(candidates[0, s].numpy(), -0.12, 0.12), axis=0)
            env_state = env.physics_state()
            local = env.simulate(setpoints.reshape(5, 3, 3))
            env.set_physics_state(env_state)
            np.testing.assert_array_equal(prediction["payload"][0, s].numpy(), local["payload"])
    finally:
        predictor.close()


@pytest.mark.parametrize("solver,cost", [("mppi", ArenaCost), ("lagrangian", ConstrainedArenaCost)])
def test_kinematic_reference_plans_to_an_open_goal(solver, cost):
    planner = ArenaPlanner(cost(KinematicPredictor()), solver=solver)
    record = run_episode(open_field(), planner)
    assert record["outcome"] == "success"


def test_oracle_reaches_an_open_goal_with_a_small_budget():
    predictor = TrueSimPredictor(workers=4)
    try:
        planner = ArenaPlanner(ArenaCost(predictor), samples=64, horizon=10)
        record = run_episode(open_field(), planner)
    finally:
        predictor.close()
    assert record["outcome"] == "success"
