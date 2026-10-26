"""Receding-horizon planning in the arena with stable-worldmodel 0.1.1 solvers.

Every participant plans with the same solver, action parameterization and cost;
only the predictor differs. A candidate is a sequence of per-aircraft setpoint
increments (H, 9) added to the last commanded setpoints. Predictors map
setpoints to the team trajectory at each decision end:

- `TrueSimPredictor` (the oracle U) forks the true simulator at the current true
  state. It is privileged: it also knows the payload mass and the onboard motor
  and controller states, none of which are observed. Its obstacle forecasts are
  the shared constant-velocity ones, like everyone's.
- `PhysicsPredictor` is the non-privileged physics reference: the same simulator
  with the nominal payload mass, started from the delayed observation. A shadow
  copy of the known onboard flight stack carries the unobserved motor and
  controller states between decisions; the delay is bridged by replaying the
  setpoint segments already sent (a Smith predictor).
- `AdaptivePhysicsPredictor` adds online payload-mass estimation to it: a bank of
  shadows over candidate masses, scored against each new observation.
- `KinematicPredictor` assumes aircraft reach their setpoints within one decision
  and the payload hangs under the formation: a differentiable test fixture for
  the augmented-Lagrangian path, not a benchmark participant.

MPPI accepts any predictor. The augmented-Lagrangian solver needs gradients
through the prediction, so it applies to differentiable predictors only, never
to the simulator.
"""
from __future__ import annotations

from collections import deque
import contextlib
import dataclasses
import io
import multiprocessing as mp
import os

import mujoco
import numpy as np
import torch
from gymnasium.spaces import Box

from .costs import CostWeights, constraint_values, soft_cost
from .env import (ACTION_LIMIT, DECISION_DT, FORMATION_DEPTH, NOMINAL_PAYLOAD_MASS, ArenaEnv,
                  quaternion_matrices)
from .scenarios import PAYLOAD_MASS_RANGE


MASS_CANDIDATES = np.round(np.arange(PAYLOAD_MASS_RANGE[0], PAYLOAD_MASS_RANGE[1]+1e-9, 0.002), 6)  # 2 g grid
ESTIMATION_WINDOW = 5   # observations scored per mass estimate (0.5 s)
ATTITUDE_SCALE = 0.05   # m per unit of rotation-column error: about g T^2/2 over one decision


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
        for key in [k for k in _WORKER_ENVS if k.public() != scenario.public()]:
            del _WORKER_ENVS[key]       # another episode; mass candidates of this one stay cached
        env = _WORKER_ENVS[scenario] = ArenaEnv(scenario)
    results = []
    for sequence in sequences:
        env.set_physics_state(state)
        results.append(env.simulate(sequence))
    return {key: np.stack([r[key] for r in results]) for key in results[0]}


class TrueSimPredictor:
    """The oracle U: exact rollouts of the true simulator in a pool of worker processes."""

    privileged = True       # the planner hands it the true state

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


class PhysicsPredictor(TrueSimPredictor):
    """Non-privileged physics reference: nominal-mass rollouts from the delayed observation."""

    privileged = False

    def __init__(self, workers=None, nominal_mass=NOMINAL_PAYLOAD_MASS):
        super().__init__(workers)
        self.nominal_mass = float(nominal_mass)
        self.reset()

    def reset(self):
        self.model = None

    def observe(self, obs, layout):
        """Advance the shadow model to the observation, re-sync what is observed, bridge to the decision.

        `layout` supplies start, goal and obstacles; its payload mass is replaced by the nominal one.
        """
        if self.model is None:
            self.scenario = dataclasses.replace(layout, payload_mass=self.nominal_mass)
            self.model = ArenaEnv(self.scenario)
            self.shadow, self.shadow_time = self.model.physics_state(), 0.0
        self.model.set_physics_state(self.shadow)
        self._replay(obs, self.shadow_time, obs["time"])
        self._sync(obs)
        self.shadow, self.shadow_time = self.model.physics_state(), obs["time"]
        self._replay(obs, obs["time"], obs["decision_time"])
        self.state = self.model.physics_state()

    def _replay(self, obs, start, stop):
        """Step the model from `start` to `stop` through the setpoint segments already sent."""
        model, dt, steps = self.model, self.model.config.dt, self.model.physics_per_decision
        times, sent = obs["recent_setpoint_times"], obs["recent_setpoints"]
        t = start
        while t < stop-1e-9:
            i = max(j for j, s in enumerate(times) if s <= t+1e-9)
            if i == 0:      # the hover setpoint before t = 0 (or a segment whose start is unknown): hold
                model.setpoint_velocity[:] = 0.0
                ramp = np.repeat(sent[0][None], steps, axis=0)
            else:
                model.setpoints[:] = sent[i-1]
                ramp = model._segment(sent[i])
            first, last = round((t-times[i])/dt), min(steps, round((stop-times[i])/dt))
            if last <= first:
                break
            for k in range(first, last):
                model.setpoints[:] = ramp[k]
                model._physics_step()
            t = times[i]+last*dt

    def _sync(self, obs):
        model = self.model
        q, v = model.data.qpos, model.data.qvel
        columns = obs["uav_rot6"]
        quaternions = np.empty((3, 4))
        for quaternion, c0, c1 in zip(quaternions, columns[:, :3], columns[:, 3:]):
            mujoco.mju_mat2Quat(quaternion, np.stack([c0, c1, np.cross(c0, c1)], axis=-1).reshape(-1))
        q[model._uav_pos] = obs["uav_pos"].reshape(-1)
        q[model._uav_quat] = quaternions.reshape(-1)
        q[model._payload_pos] = obs["payload_pos"]
        v[model._uav_lin] = obs["uav_vel"].reshape(-1)
        v[model._uav_ang] = obs["uav_omega"].reshape(-1)
        v[model._payload_lin] = obs["payload_vel"]
        mujoco.mj_forward(model.model, model.data)


class AdaptivePhysicsPredictor(PhysicsPredictor):
    """The physics reference with online payload-mass estimation (multiple-model estimation).

    A bank of shadows, one per candidate mass on a 2 g grid over the scenario prior, runs the
    physics reference's cycle at every new observation: replay the sent segments up to it, score
    the prediction against it, re-sync. The estimate is the candidate with the least squared error
    over the last 0.5 s of observations. The first observation is scored against each candidate's
    settled hover, so the initial estimate comes from the hover tilt that the cable tension sets.
    Planning rolls out the estimated candidate: its shadow states and its mass.
    """

    def __init__(self, workers=None, candidates=MASS_CANDIDATES, window=ESTIMATION_WINDOW):
        self.candidates, self.window = np.asarray(candidates, dtype=float), int(window)
        super().__init__(workers)

    def reset(self):
        super().reset()
        self.bank, self.errors, self.estimates = None, deque(maxlen=self.window), []

    def observe(self, obs, layout):
        if self.bank is None:
            self.bank = [ArenaEnv(dataclasses.replace(layout, payload_mass=float(m))) for m in self.candidates]
            self.shadows, self.shadow_time = [model.physics_state() for model in self.bank], 0.0
        errors = np.empty(len(self.bank))
        for k, model in enumerate(self.bank):
            self.model = model
            model.set_physics_state(self.shadows[k])
            self._replay(obs, self.shadow_time, obs["time"])
            errors[k] = self._error(obs)
            self._sync(obs)
            self.shadows[k] = model.physics_state()
        if not self.errors or obs["time"] > self.shadow_time+1e-9:   # a repeated observation tells nothing new
            self.errors.append(errors)
        self.shadow_time = obs["time"]
        best = int(np.argmin(np.sum(self.errors, axis=0)))
        self.model, self.scenario = self.bank[best], self.bank[best].scenario
        self.model.set_physics_state(self.shadows[best])
        self._replay(obs, obs["time"], obs["decision_time"])
        self.state = self.model.physics_state()
        self.estimates.append(float(self.candidates[best]))

    def _error(self, obs):
        """Squared distance of the shadow from the observation, in metres."""
        model = self.model
        q, v = model.data.qpos, model.data.qvel
        rotation = quaternion_matrices(q[model._uav_quat].reshape(3, 4))
        rot6 = np.concatenate((rotation[:, :, 0], rotation[:, :, 1]), axis=1)
        residuals = (q[model._uav_pos]-obs["uav_pos"].reshape(-1), q[model._payload_pos]-obs["payload_pos"],
                     DECISION_DT*(v[model._uav_lin]-obs["uav_vel"].reshape(-1)),
                     DECISION_DT*(v[model._payload_lin]-obs["payload_vel"]),
                     ATTITUDE_SCALE*(rot6-obs["uav_rot6"]).reshape(-1))
        return float(sum(np.square(r).sum() for r in residuals))


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
        self.cost, self.horizon, self.name, self.seed = cost, horizon, solver, seed
        self.reset(0)

    def reset(self, episode):
        """Start an episode. The solver is reseeded from (seed, episode), so an episode's samples do not
        depend on the episodes run before it; run.py passes the layout seed, which every condition and
        participant on that layout shares (common random numbers)."""
        self.previous = None
        self.solver.torch_gen.manual_seed(int(np.random.SeedSequence([self.seed, int(episode)]).generate_state(1)[0]))
        if hasattr(self.cost.predictor, "reset"):
            self.cost.predictor.reset()

    def act(self, obs, env=None):
        anchor = obs["recent_setpoints"][-1]
        predictor = self.cost.predictor
        if hasattr(predictor, "observe"):
            predictor.observe(obs, env.scenario.public())               # layout without the payload mass
        elif getattr(predictor, "privileged", False):
            predictor.set_state(env.scenario, env.physics_state())     # the oracle U only
        info = info_from_observation(obs, anchor)
        with contextlib.redirect_stdout(io.StringIO()):     # swm 0.1.1 prints timing every solve
            outputs = self.solver.solve(info, init_action=self.previous)
        plan = outputs["actions"][0].detach().clamp(-ACTION_LIMIT, ACTION_LIMIT)
        self.previous = plan[1:].unsqueeze(0)
        return np.asarray(anchor, dtype=float)+plan[0].double().numpy().reshape(3, 3)
