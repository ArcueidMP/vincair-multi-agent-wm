"""Receding-horizon planning in the arena with stable-worldmodel 0.1.1 solvers.

Every participant plans with the same solver, action parameterization and cost;
only the predictor differs. A candidate is a sequence of per-aircraft setpoint
increments (H, 9) added to the last commanded setpoints. Predictors map
setpoints to the team trajectory at each decision end:

- `TrueSimPredictor` (the oracle U) forks the true simulator at the current state.
  Bridging the observation delay with an exact model is equivalent to this.
  Its obstacle forecasts are the shared constant-velocity ones, like everyone's.
- `KinematicPredictor` assumes aircraft reach their setpoints within one decision
  and the payload hangs under the formation: a differentiable no-dynamics reference.

MPPI accepts any predictor. The augmented-Lagrangian solver needs gradients
through the prediction, so it applies to differentiable predictors only, never
to the simulator.
"""
from __future__ import annotations

import contextlib
import io
import math
import multiprocessing as mp
import os

import numpy as np
import torch
from gymnasium.spaces import Box

from .costs import CostWeights, constraint_values, soft_cost
from .env import DECISION_DT, ArenaEnv


ACTION_LIMIT = 0.12     # setpoint increment per decision and axis (m), 1.2 m/s reference speed
FORMATION_DEPTH = math.sqrt(0.6**2-0.25**2)


def info_from_observation(obs, anchor, dtype=torch.float32):
    """Batch-of-one tensors in the layout stable-worldmodel solvers expand over samples."""
    def tensor(x, shape=None):
        x = np.asarray(x, dtype=float)
        return torch.as_tensor(x.reshape(shape) if shape else x, dtype=dtype).unsqueeze(0)
    return dict(goal=tensor(obs["goal"]), pillars=tensor(obs["pillars"], (-1, 3)),
                mover_pos=tensor(obs["mover_pos"], (-1, 3)), mover_vel=tensor(obs["mover_vel"], (-1, 3)),
                mover_radius=tensor(obs["mover_radius"], (-1,)),
                forecast_offset=tensor([obs["decision_time"]-obs["time"]]), anchor=tensor(anchor, (-1,)),
                uav_pos=tensor(obs["uav_pos"]), uav_vel=tensor(obs["uav_vel"]),
                payload_pos=tensor(obs["payload_pos"]), payload_vel=tensor(obs["payload_vel"]))


def setpoints_from(info, candidates):
    """Clip increments and accumulate them from the anchor: (..., H, 9) -> (..., H, 3, 3)."""
    increments = candidates.clamp(-ACTION_LIMIT, ACTION_LIMIT)
    setpoints = info["anchor"].unsqueeze(-2)+increments.cumsum(-2)
    return increments, setpoints.unflatten(-1, (3, 3))


class ArenaCost(torch.nn.Module):
    """stable-worldmodel `Costable`: the shared soft cost of a predictor's trajectories."""

    def __init__(self, predictor, weights=CostWeights()):
        super().__init__()
        self.predictor, self.weights = predictor, weights

    def predict(self, info, candidates):
        increments, setpoints = setpoints_from(info, candidates)
        return self.predictor(info, setpoints), increments

    def get_cost(self, info_dict, action_candidates):
        prediction, increments = self.predict(info_dict, action_candidates)
        return soft_cost(prediction, info_dict, increments, self.weights)

    criterion = get_cost


class ConstrainedArenaCost(ArenaCost):
    """Adds `get_constraints` (B, S, 4) for the augmented-Lagrangian solver.

    The cost keeps its soft penalties because swm 0.1.1 returns the lowest-cost
    sample without checking constraints.
    """

    def get_constraints(self, info_dict, action_candidates):
        prediction, _ = self.predict(info_dict, action_candidates)
        return constraint_values(prediction, info_dict, self.weights)


class KinematicPredictor:
    """Differentiable reference without dynamics."""

    def __call__(self, info, setpoints):
        uav = setpoints
        payload = uav.mean(-2)-uav.new_tensor([0., 0., FORMATION_DEPTH])
        previous = torch.cat([info["payload_pos"].unsqueeze(-2).expand_as(payload[..., :1, :]), payload[..., :-1, :]], -2)
        return dict(uav=uav, payload=payload, payload_vel=(payload-previous)/DECISION_DT)


_WORKER_ENVS = {}


def _rollouts(scenario, state, sequences):
    env = _WORKER_ENVS.get(scenario)
    if env is None:
        _WORKER_ENVS.clear()
        env = _WORKER_ENVS[scenario] = ArenaEnv(scenario)
    results = []
    for sequence in sequences:
        env.set_physics_state(state)
        results.append(env.simulate(sequence))
    return {key: np.stack([r[key] for r in results]) for key in results[0]}


class TrueSimPredictor:
    """The oracle U: exact rollouts of the true simulator in a pool of worker processes."""

    def __init__(self, workers=None):
        self.workers = int(workers or max(1, (os.cpu_count() or 2)-2))
        # Workers only run NumPy, MuJoCo and the native controller, so forking is safe and,
        # unlike spawn, does not re-import the caller's __main__ (stdin, notebooks).
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        self.pool = mp.get_context(method).Pool(self.workers)
        self.scenario = self.state = None

    def set_state(self, scenario, state):
        self.scenario, self.state = scenario, state

    def __call__(self, info, setpoints):
        values = setpoints.detach().cpu().numpy().astype(float)
        flat = values.reshape(-1, *values.shape[-3:])
        chunks = [c for c in np.array_split(flat, self.workers) if len(c)]
        parts = self.pool.starmap(_rollouts, [(self.scenario, self.state, c) for c in chunks])
        lead = values.shape[:-3]
        return {key: torch.as_tensor(np.concatenate([p[key] for p in parts]), dtype=setpoints.dtype)
                .reshape(*lead, *parts[0][key].shape[1:]) for key in parts[0]}

    def close(self):
        self.pool.terminate()
        self.pool.join()


class ArenaPlanner:
    """Plan with a stable-worldmodel solver every decision; execute the first increment."""

    def __init__(self, cost, *, solver="mppi", horizon=15, samples=512, var_scale=0.04, temperature=0.05,
                 topk=None, iterations=1, device="cpu", seed=0, lagrangian=None):
        from stable_worldmodel.policy import PlanConfig
        from stable_worldmodel.solver import LagrangianSolver, MPPISolver
        if solver == "mppi":
            self.solver = MPPISolver(cost, num_samples=samples, var_scale=var_scale, n_steps=iterations,
                                     topk=topk, temperature=temperature, device=device, seed=seed)
        elif solver == "lagrangian":
            options = dict(n_steps=20, n_outer_steps=3, num_samples=8, var_scale=var_scale, rho_init=1.0,
                           optimizer_kwargs=dict(lr=0.01), persist_multipliers=False)
            options.update(lagrangian or {})
            self.solver = LagrangianSolver(cost, device=device, seed=seed, **options)
        else:
            raise ValueError(f"Unknown solver {solver}")
        self.solver.configure(action_space=Box(-ACTION_LIMIT, ACTION_LIMIT, shape=(1, 9)), n_envs=1,
                              config=PlanConfig(horizon=horizon, receding_horizon=1, action_block=1))
        self.cost, self.horizon, self.name = cost, horizon, solver
        self.reset()

    def reset(self):
        self.previous = None

    def act(self, obs, env=None):
        anchor = obs["recent_setpoints"][-1] if len(obs["recent_setpoints"]) else obs["uav_pos"]
        if hasattr(self.cost.predictor, "set_state"):
            self.cost.predictor.set_state(env.scenario, env.physics_state())   # privileged: oracle only
        info = info_from_observation(obs, anchor)
        with contextlib.redirect_stdout(io.StringIO()):     # swm 0.1.1 prints timing every solve
            outputs = self.solver.solve(info, init_action=self.previous)
        plan = outputs["actions"][0].detach().clamp(-ACTION_LIMIT, ACTION_LIMIT)
        self.previous = plan[1:].unsqueeze(0)
        return np.asarray(anchor, dtype=float)+plan[0].double().numpy().reshape(3, 3)
