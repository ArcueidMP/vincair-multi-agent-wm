"""Arena tasks in the official CrazyMARL environment (CrazyMARL venv, inside its checkout).

`ArenaMultiQuadEnv` subclasses the official `MultiQuadEnv` and reuses its motor model,
disturbance randomization, safety checks and reward terms. It adds what the arena task
needs (see `crazymarl_task`): the arena plant parameters, the settled-hover start, the
goal, obstacles and their collisions, delayed observations of the payload, teammates and
obstacles, and a progress reward. `ArenaQuad` is the JaxMARL wrapper with the extended
per-agent observation mapping; `make` stands in for `jaxmarl.make` so the official IPPO
trainer can run unchanged.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax import envs
from brax.io import mjcf

from crazymarl.builders.quad_env_builder import QuadEnvGenerator, get_body_and_joint_ids
from crazymarl.envs.multi_quad_env import MultiQuadEnv
from crazymarl.observations.multi_quad_observation import build_obs
from crazymarl.rewards.multi_quad_reward import calc_reward
from crazymarl.utils.multi_quad_utils import R_from_quat, upright_angles
from jaxmarl.environments import spaces
from jaxmarl.environments.mabrax.mabrax_env import MABraxEnv

from . import crazymarl_task as task
from .scenarios import WORKSPACE

ENV_ID = "arenaquad_ix4"


def arena_system(arena, policy_freq):
    """The upstream generated scene with the arena's airframe mass, inertia and central cable attachment."""
    generator = QuadEnvGenerator(n_quads=task.N_QUADS, cable_length=float(arena["cable_length"]),
                                 payload_mass=float(arena["payload_mass"]), payload=True, dt=1.0/policy_freq)
    root = ET.fromstring(generator.generate_xml())
    for i in range(task.N_QUADS):
        inertial = root.find(f".//body[@name='q{i}_cf2']/inertial")
        inertial.set("mass", repr(float(arena["drone_mass"])))
        inertial.set("diaginertia", " ".join(repr(float(x)) for x in arena["inertia"]))
    root.find(".//site[@name='payload_s']").set("pos", "0 0 0")
    return mjcf.load_model(mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")))


class ArenaMultiQuadEnv(MultiQuadEnv):
    def __init__(self, task_path, progress_coef=0.25, base_thrust_range=(0.14, 0.16), **kwargs):
        arena = task.load_task(task_path)
        kwargs.update(num_quads=task.N_QUADS, payload=True, cable_length=float(arena["cable_length"]),
                      payload_mass=float(arena["payload_mass"]), trajectory=None)
        super().__init__(**kwargs)
        self.cfg.motor_tau = float(arena["motor_tau"])        # upstream randomizes it around this value
        self.sys = arena_system(arena, self.cfg.policy_freq)
        self.ids = get_body_and_joint_ids(self.sys, num_quads=task.N_QUADS)
        self.progress_coef = float(progress_coef)
        self.base_thrust_range = tuple(float(x) for x in base_thrust_range)
        self.rotor_max = float(arena["rotor_max"])
        for key in ("start", "goal", "pillars", "movers", "formation", "hover_quat"):
            setattr(self, key, jp.asarray(arena[key], dtype=jp.float32))
        self.hover_thrust = jp.asarray(arena["hover_thrust"], dtype=jp.float32).reshape(-1)
        self.latency_s = jp.asarray(task.LATENCIES_S, dtype=jp.float32)
        self.latency_steps = jp.asarray([task.latency_steps(x) for x in task.LATENCIES_S])
        self.noise = jp.asarray(task.noise_scale(), dtype=jp.float32)
        self.low = jp.asarray([w[0] for w in WORKSPACE]); self.high = jp.asarray([w[1] for w in WORKSPACE])

    # ---------- state access ----------
    def _team(self, ps):
        quad_ids = jp.array(self.ids["quad_body_ids"])
        quad_pos = ps.xpos[quad_ids]
        velocities = jp.stack([jax.lax.dynamic_slice(ps.qvel, (d,), (3,)) for d in self.ids["quad_dofadr"]])
        rates = jp.stack([jax.lax.dynamic_slice(ps.sensordata, (a,), (3,)) for a in self.ids["gyro_sensor_adrs"]])
        payload_pos = ps.xpos[self.ids["payload_body_id"]]
        payload_vel = jax.lax.dynamic_slice(ps.qvel, (self.ids["payload_dofadr"],), (3,))
        return quad_pos, R_from_quat(ps.xquat[quad_ids]), velocities, rates, payload_pos, payload_vel

    def _delayed(self, ps, layout):
        quad_pos, _, _, _, payload_pos, payload_vel = self._team(ps)
        mover_pos, mover_vel = task.mover_states(jp, self.movers[layout], ps.time)
        return task.delayed_vector(jp, payload_pos, payload_vel, quad_pos, mover_pos, mover_vel)

    def _observe(self, ps, last_action, history, layout, latency, key):
        quad_pos, rotations, velocities, rates, _, _ = self._team(ps)
        obs = task.global_observation(
            jp, goal=self.goal[layout], delayed=history[self.latency_steps[latency]], own_pos=quad_pos,
            rotations=rotations, velocities=velocities, rates=rates, last_action=last_action,
            pillars=self.pillars[layout], mover_radius=self.movers[layout][:, 8], latency=self.latency_s[latency])
        return obs+self.cfg.obs_noise*self.noise*jax.random.normal(key, obs.shape)

    # ---------- episode ----------
    def reset(self, rng):
        cfg = self.cfg
        keys = jax.random.split(rng, 9)
        # Upstream motor randomization, centred on the arena's 0.16 N rotors.
        offsets = 0.008*jax.random.normal(keys[0], (self.sys.nu,))
        base = jax.random.uniform(keys[1], (task.N_QUADS,), minval=self.base_thrust_range[0],
                                  maxval=self.base_thrust_range[1])
        max_thrust = jp.clip(jp.repeat(base, 4)+offsets, self.base_thrust_range[0]-0.01, self.rotor_max)
        tau = cfg.motor_tau*jp.clip(1+jax.random.normal(keys[2], ())*0.3, 0.1, 1.5)
        layout = jax.random.randint(keys[3], (), 0, self.start.shape[0])
        latency = jax.random.randint(keys[4], (), 0, len(task.LATENCIES_S))
        # Settled hover at the arena start, with small perturbations.
        start = self.start[layout]
        qpos = self.sys.qpos0
        s = self.ids["payload_qpos_start"]
        qpos = qpos.at[s:s+3].set(start+0.01*jax.random.normal(keys[5], (3,)))
        noise = 0.01*jax.random.normal(keys[6], (task.N_QUADS, 3))
        for i, s in enumerate(self.ids["quad_qpos_starts"]):
            qpos = qpos.at[s:s+3].set(start+self.formation[i]+noise[i])
            qpos = qpos.at[s+3:s+7].set(self.hover_quat[i])
        qvel = 0.05*jax.random.normal(keys[7], (self.sys.nv,))
        ps = self.pipeline_init(qpos, qvel)
        hover_action = 2.0*self.hover_thrust/max_thrust-1.0
        first = self._delayed(ps, layout)
        history = jp.tile(first, (task.HISTORY, 1))
        distance = jp.linalg.norm(self.goal[layout]-ps.xpos[self.ids["payload_body_id"]])
        obs = self._observe(ps, hover_action, history, layout, latency, keys[8])
        metrics = dict(time=ps.time, reward=0.0, max_thrust=max_thrust, filtered_rpm_proxy=jp.sqrt(self.hover_thrust),
                       motor_alpha=self.dt/tau, dynamic=1.1, target_position=self.goal[layout], layout=layout,
                       latency=latency, history=history, first_history=history, distance=distance,
                       first_distance=distance, clearance=jp.array(1e3, dtype=jp.float32))
        return envs.State(ps, obs, jp.array(0.0), jp.array(0.0), metrics)

    def step(self, state, action):
        cfg, m = self.cfg, state.metrics
        layout, latency, goal = m["layout"], m["latency"], m["target_position"]
        # Brax auto-reset restores only the pipeline state and observation: restart per-episode signals.
        first = state.pipeline_state.time < 0.5*self.dt
        rpm = jp.where(first, jp.sqrt(self.hover_thrust), m["filtered_rpm_proxy"])
        history = jp.where(first, m["first_history"], m["history"])
        previous = jp.where(first, m["first_distance"], m["distance"])
        dynamic = jp.where(first, 1.1, m["dynamic"])

        # Upstream motor model and disturbances.
        max_thrust = m["max_thrust"]
        commanded = jp.clip(0.5*(action+1.0), 0.0, 1.0)*max_thrust
        filtered = rpm+m["motor_alpha"]*(jp.sqrt(commanded)-rpm)
        xfrc, disturbance = self._build_disturbance_xfrc(state.pipeline_state)
        ps = self.pipeline_step(state.pipeline_state.replace(xfrc_applied=xfrc), jp.square(filtered))
        key = jax.random.PRNGKey(0)
        key = jax.random.fold_in(key, jp.int32(ps.time*1e6))
        key = jax.random.fold_in(key, jp.int32(jp.sum(ps.xpos)*1e3))
        key = jax.random.fold_in(key, jp.int32(jp.sum(ps.cvel)*1e3))
        obs_key, jump_key, delta_key = jax.random.split(key, 3)

        # Upstream safety checks, the arena's crash rule, obstacles and workspace.
        quad_pos, rotations, _, _, payload_pos, _ = self._team(ps)
        angles = upright_angles(rotations)
        gaps = jp.linalg.norm(quad_pos[:, None]-quad_pos[None], axis=-1)
        quad_collision = jp.min(jp.where(jp.eye(task.N_QUADS, dtype=bool), jp.inf, gaps)) < 0.15
        crash = jp.any(quad_pos[:, 2] < 0.05) | (payload_pos[2] < task.PAYLOAD_RADIUS)
        mover_pos, mover_vel = task.mover_states(jp, self.movers[layout], ps.time)
        clearance = task.team_clearance(jp, quad_pos, payload_pos, self.pillars[layout], mover_pos,
                                        self.movers[layout][:, 8])
        points = jp.concatenate([quad_pos, payload_pos[None]])
        outside = jp.any(points < self.low) | jp.any(points > self.high)
        collision = quad_collision | crash | (clearance < 0)
        out_of_bounds = jp.any(jp.abs(angles) > jp.radians(170)) \
            | jp.any(quad_pos[:, 2] < payload_pos[2]-0.15) | outside

        history = jp.concatenate([self._delayed(ps, layout)[None], history[:-1]])
        obs = self._observe(ps, action, history, layout, latency, obs_key)

        # Upstream reward terms on the fresh state, plus progress toward the goal.
        dynamic = jp.where(disturbance > 0.0, 1.1, dynamic*jp.exp(-self.time_per_action/self.dynamic_decay_tau))
        upstream_obs = build_obs(ps, action, goal, 0.0, obs_key, self.ids, payload=True)
        reward = calc_reward(upstream_obs, ps.time, collision, out_of_bounds, action, angles, goal, ps,
                             max_thrust, cfg, dynamic)
        distance = jp.linalg.norm(goal-payload_pos)
        reward = reward+self.progress_coef*jp.clip((previous-distance)/self.time_per_action, -2.0, 2.0)
        done = jp.logical_or(collision, out_of_bounds).astype(jp.float32)

        # Upstream random RPM jumps.
        jump = jax.random.bernoulli(jump_key, p=jp.clip(self.time_per_action/self.rpm_jump_interval_s, 0., 1.))
        filtered = jp.clip(filtered+jump*self.rpm_jump_std*jax.random.normal(delta_key, filtered.shape), 0.0, jp.inf)
        metrics = dict(m, time=ps.time, reward=reward, filtered_rpm_proxy=filtered, dynamic=dynamic, history=history,
                       distance=distance, clearance=clearance)
        return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done, metrics=metrics)


class ArenaQuad(MABraxEnv):
    """JaxMARL wrapper: upstream per-agent mapping plus obstacles and latency (crazymarl_task)."""

    def __init__(self, **kwargs):
        super().__init__(ENV_ID, **kwargs)
        self.agent_obs_mapping = {f"agent_{i}": jp.asarray(task.agent_indices(i)) for i in range(task.N_QUADS)}
        self.observation_spaces = {agent: spaces.Box(-jp.inf, jp.inf, shape=(indices.size,))
                                   for agent, indices in self.agent_obs_mapping.items()}


def register():
    envs.register_environment("arenaquad", ArenaMultiQuadEnv)


def make(upstream_make):
    """A `jaxmarl.make` that also builds the arena environment."""
    def make_env(env_id, episode_length=1000, **kwargs):
        if env_id == ENV_ID:
            register()
            return ArenaQuad(episode_length=episode_length, **kwargs)
        return upstream_make(env_id, episode_length=episode_length, **kwargs)
    return make_env
