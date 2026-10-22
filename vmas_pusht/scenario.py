"""Cooperative Push-T as a VMAS scenario (VMAS 1.5.2), state-based, vectorized in torch.

The T is a bar (0.4 x 0.1) and a stem (0.1 x 0.3) welded by a dist-0 joint with fixed relative rotation. Floor
friction on both parts uses VMAS's linear / angular friction: a friction force of at most coeff x mass opposes
motion, so a push below the total friction moves nothing and only the excess accelerates the T. With
friction_ratio r in (1, 2) the friction is r x one agent's force: one agent cannot move the T, two can, and the T
moves faster the more the pushes exceed the friction (graded, unlike the pymunk env's quasi-static threshold).
rot_friction_ratio does the same for torque (x one agent's torque at the bar tip).
friction_ratio=0 gives the plain additive regime (a heavy T with drag, like VMAS Transport).

Success: T within pos_tol of its goal position and rot_tol of its goal angle (T pose only; no agent goal spot).

    from vmas_pusht.scenario import make
    env = make(num_envs=64, device="cuda")
    obs = env.reset(); obs, rew, done, info = env.step([a0, a1])     # actions (B, 2) in [-1, 1]
    s = state(env)                                                   # (B, STATE_DIM) global state
"""
import math

import torch
from vmas import make_env
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.joints import Joint
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

BAR = (0.4, 0.1)    # length (x), width (y) in the T's frame
STEM = (0.1, 0.3)
STEM_OFFSET = -(BAR[1] + STEM[1]) / 2  # stem centre below the bar centre, in the T frame (y)
AGENT_R = 0.05
TIP_R = 0.25        # lever of a push at the bar's tip, about the T's centre of mass (approx.)
STATE_DIM = 15      # [a0 pos 2, a0 vel 2, a1 pos 2, a1 vel 2, T pos 2, T rot 1, T vel 2, T angvel 1, 0] see state()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PushT(BaseScenario):
    def make_world(self, batch_dim, device, **kwargs):
        self.n_agents = kwargs.pop("n_agents", 2)
        self.t_mass = kwargs.pop("t_mass", 2.0)            # bar + stem
        self.friction_ratio = kwargs.pop("friction_ratio", 1.5)
        self.rot_friction_ratio = kwargs.pop("rot_friction_ratio", 1.2)  # x one agent's torque at the T's tip
        self.agent_force = kwargs.pop("agent_force", 1.0)  # u_multiplier: force at |u| = 1
        self.pos_tol = kwargs.pop("pos_tol", 0.05)
        self.rot_tol = kwargs.pop("rot_tol", math.radians(20))
        self.substeps = kwargs.pop("substeps", 5)
        self.semidim = kwargs.pop("semidim", 1.0)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        world = World(batch_dim, device, substeps=self.substeps, x_semidim=self.semidim + 0.2,
                      y_semidim=self.semidim + 0.2, drag=0.25)
        for i in range(self.n_agents):
            world.add_agent(Agent(name=f"agent_{i}", shape=Sphere(AGENT_R), u_multiplier=self.agent_force,
                                  color=[Color.BLUE, Color.RED, Color.GREEN][i % 3]))
        area = BAR[0] * BAR[1] + STEM[0] * STEM[1]
        masses = [self.t_mass * BAR[0] * BAR[1] / area, self.t_mass * STEM[0] * STEM[1] / area]
        # friction force cap = coeff x mass, summed over both parts = friction_ratio x agent_force
        lin = self.friction_ratio * self.agent_force / self.t_mass if self.friction_ratio else None
        # angular friction torque cap = coeff x moment of inertia, summed = rot_friction_ratio x agent_force x TIP_R
        inertia = sum(m * (l * l + w * w) / 12 for m, (l, w) in zip(masses, (BAR, STEM)))
        ang = self.rot_friction_ratio * self.agent_force * TIP_R / inertia if self.rot_friction_ratio else None
        self.bar, self.stem = (
            Landmark(name=n, shape=Box(length=l, width=w), movable=True, rotatable=True, mass=m, collide=True,
                     linear_friction=lin, angular_friction=ang, color=Color.GRAY)
            for n, (l, w), m in (("T bar", BAR, masses[0]), ("T stem", STEM, masses[1])))
        world.add_landmark(self.bar)
        world.add_landmark(self.stem)
        world.add_joint(Joint(self.bar, self.stem, anchor_a=(0, -1), anchor_b=(0, 1), dist=0.0,
                              rotate_a=False, rotate_b=False, fixed_rotation_a=0.0, fixed_rotation_b=0.0))
        self.goal_bar, self.goal_stem = (
            Landmark(name=n, shape=Box(length=l, width=w), collide=False, movable=False, color=Color.LIGHT_GREEN)
            for n, (l, w) in (("goal bar", BAR), ("goal stem", STEM)))
        world.add_landmark(self.goal_bar)
        world.add_landmark(self.goal_stem)
        self.goal_pose = torch.zeros(batch_dim, 3, device=device)
        self.prev_err = torch.zeros(batch_dim, device=device)
        return world

    # ---------- placement ----------
    def place_t(self, pose, env_index=None, goal=False):
        """pose (B or 1, 3) = [x, y, rot] of the bar centre; the stem follows."""
        bar, stem = (self.goal_bar, self.goal_stem) if goal else (self.bar, self.stem)
        x, y, r = pose[:, 0:1], pose[:, 1:2], pose[:, 2:3]
        sp = torch.cat([x - STEM_OFFSET * torch.sin(r), y + STEM_OFFSET * torch.cos(r)], 1)
        for ent, p in ((bar, torch.cat([x, y], 1)), (stem, sp)):
            ent.set_pos(p if env_index is None else p[0], batch_index=env_index)
            ent.set_rot(r if env_index is None else r[0], batch_index=env_index)
            if not goal:
                ent.set_vel(torch.zeros_like(p) if env_index is None else torch.zeros_like(p[0]), batch_index=env_index)
                ent.set_ang_vel(torch.zeros_like(r) if env_index is None else torch.zeros_like(r[0]), batch_index=env_index)

    def reset_world_at(self, env_index=None):
        B = self.world.batch_dim if env_index is None else 1
        dev = self.world.device
        lim = self.semidim - 0.4
        pose = torch.cat([(torch.rand(B, 2, device=dev) * 2 - 1) * lim, (torch.rand(B, 1, device=dev) * 2 - 1) * math.pi], 1)
        # goal: 0.15-0.5 away, turned by up to 60 degrees, inside the arena
        ang = torch.rand(B, 1, device=dev) * 2 * math.pi
        dist = 0.15 + 0.35 * torch.rand(B, 1, device=dev)
        gxy = (pose[:, :2] + dist * torch.cat([torch.cos(ang), torch.sin(ang)], 1)).clamp(-lim, lim)
        grot = wrap(pose[:, 2:3] + (torch.rand(B, 1, device=dev) * 2 - 1) * math.radians(60))
        goal = torch.cat([gxy, grot], 1)
        self.place_t(pose, env_index)
        self.place_t(goal, env_index, goal=True)
        # agents: uniform in the arena, at least 0.4 from the T centre and 0.15 from each other
        placed = []
        for agent in self.world.agents:
            p = (torch.rand(B, 2, device=dev) * 2 - 1) * self.semidim
            for _ in range(50):
                bad = (p - pose[:, :2]).norm(dim=1) < 0.4
                for q in placed:
                    bad |= (p - q).norm(dim=1) < 0.15
                if not bad.any():
                    break
                p[bad] = (torch.rand(int(bad.sum()), 2, device=dev) * 2 - 1) * self.semidim
            placed.append(p)
            agent.set_pos(p if env_index is None else p[0], batch_index=env_index)
            agent.set_vel(torch.zeros_like(p) if env_index is None else torch.zeros_like(p[0]), batch_index=env_index)
        if env_index is None:
            self.goal_pose[:] = goal
            self.prev_err[:] = self.pose_err()
        else:
            self.goal_pose[env_index] = goal[0]
            self.prev_err[env_index] = self.pose_err()[env_index]

    # ---------- task ----------
    def t_pose(self):
        return torch.cat([self.bar.state.pos, self.bar.state.rot], 1)

    def pose_err(self):
        """position error + angle error weighted like Push-T (20 deg ~ 4 x pos_tol)."""
        d = self.t_pose() - self.goal_pose
        return d[:, :2].norm(dim=1) + (4 * self.pos_tol / self.rot_tol) * wrap(d[:, 2]).abs()

    def success(self):
        d = self.t_pose() - self.goal_pose
        return (d[:, :2].norm(dim=1) < self.pos_tol) & (wrap(d[:, 2]).abs() < self.rot_tol)

    def reward(self, agent):
        if agent is self.world.agents[0]:
            err = self.pose_err()
            self.rew = (self.prev_err - err) * 10 + 10 * self.success().float()
            self.prev_err = err
        return self.rew

    def observation(self, agent):
        others = [a.state.pos - agent.state.pos for a in self.world.agents if a is not agent]
        r, g = self.bar.state.rot, self.goal_pose
        return torch.cat([agent.state.pos, agent.state.vel, self.bar.state.pos - agent.state.pos, torch.sin(r), torch.cos(r),
                          self.bar.state.vel, g[:, :2] - self.bar.state.pos, torch.sin(g[:, 2:]), torch.cos(g[:, 2:]),
                          *others], 1)

    def done(self):
        return self.success()

    def info(self, agent):
        return {"success": self.success(), "pose_err": self.pose_err()}


def make(num_envs=32, device="cpu", seed=0, max_steps=None, **kwargs):
    return make_env(PushT(), num_envs=num_envs, device=device, seed=seed, continuous_actions=True,
                    max_steps=max_steps, **kwargs)


def scenario(env):
    return env.scenario


def state(env):
    """(B, STATE_DIM): both agents' pos + vel, T bar pos, rot, vel, angular vel, padding."""
    sc = env.scenario
    a = [torch.cat([ag.state.pos, ag.state.vel], 1) for ag in env.world.agents]
    b = sc.bar.state
    return torch.cat([*a, b.pos, wrap(b.rot), b.vel, b.ang_vel, torch.zeros_like(b.rot)], 1)


def set_state(env, s, goal=None):
    """Put every env of the batch at state s (B or 1, STATE_DIM), optionally with goal poses (B or 1, 3)."""
    sc = env.scenario
    B = env.world.batch_dim
    s = s.expand(B, -1) if s.shape[0] == 1 else s
    for i, ag in enumerate(env.world.agents):
        ag.set_pos(s[:, 4 * i:4 * i + 2].contiguous(), batch_index=None)
        ag.set_vel(s[:, 4 * i + 2:4 * i + 4].contiguous(), batch_index=None)
    sc.place_t(s[:, 8:11].contiguous())
    for ent in (sc.bar, sc.stem):
        ent.set_vel(s[:, 11:13].contiguous(), batch_index=None)
        ent.set_ang_vel(s[:, 13:14].contiguous(), batch_index=None)
    if goal is not None:
        goal = goal.expand(B, -1) if goal.shape[0] == 1 else goal
        sc.place_t(goal.contiguous(), goal=True)
        sc.goal_pose[:] = goal
    sc.prev_err[:] = sc.pose_err()
