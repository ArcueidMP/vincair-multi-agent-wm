"""Vectorized scripted team for VMAS Push-T (the data source; the same idea as collect.CoopPolicy).

Contact spots lie every SPACING along the T's outer boundary (T frame). A pair of spots is scored by the part of its
two pushes (unit force along the inward normals) that exceeds the friction, compared with the T's remaining error,
minus a walking cost; both agents walk to their spots (orbiting the T at a safe radius to a point out along the
spot's normal, then straight in), push together for push_steps, and the team replans. For data diversity, each replan is with
probability p_solo a solo segment (one agent pushes one spot alone, the other wanders) and with p_random a random
segment; actions get Gaussian noise.
"""
import math

import torch

from vmas_pusht.scenario import AGENT_R, BAR, STEM, STEM_OFFSET, TIP_R, wrap

SPACING = 0.04
STANDOFF = AGENT_R + 0.015
ORBIT_R = 0.38      # clearance radius around the T's centre of mass while walking to a spot


def contact_spots(device):
    """(K, 2) points and (K, 2) outward normals on the T's outer boundary, T frame (bar centre at the origin)."""
    bx, by = BAR[0] / 2, BAR[1] / 2
    sx, sy0, sy1 = STEM[0] / 2, -by, STEM_OFFSET - STEM[1] / 2
    pts, nrm = [], []

    def edge(a, b, n, skip=None):
        L = math.dist(a, b)
        for f in torch.arange(SPACING / 2, L, SPACING) / L:
            p = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
            if skip and skip(p):
                continue
            pts.append(p), nrm.append(n)

    edge((-bx, by), (bx, by), (0.0, 1.0))                                            # bar top
    edge((-bx, -by), (-bx, by), (-1.0, 0.0)); edge((bx, -by), (bx, by), (1.0, 0.0))   # bar ends
    edge((-bx, -by), (bx, -by), (0.0, -1.0), skip=lambda p: abs(p[0]) < sx + AGENT_R)  # bar underside beside the stem
    edge((-sx, sy1), (-sx, sy0), (-1.0, 0.0), skip=lambda p: p[1] > sy0 - AGENT_R)    # stem sides
    edge((sx, sy1), (sx, sy0), (1.0, 0.0), skip=lambda p: p[1] > sy0 - AGENT_R)
    edge((-sx, sy1), (sx, sy1), (0.0, -1.0))                                         # stem end
    return torch.tensor(pts, device=device, dtype=torch.float32), torch.tensor(nrm, device=device, dtype=torch.float32)


def rot(v, r):
    """rotate (..., 2) by angle r (broadcast)."""
    c, s = torch.cos(r), torch.sin(r)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


class ScriptedTeam:
    def __init__(self, env, friction_ratio=1.5, rot_friction_ratio=1.2, push_steps=10, approach_steps=60,
                 p_solo=0.1, p_random=0.05, noise=0.1, walk_cost=0.3, seed=0):
        self.env, self.dev, self.B = env, env.world.device, env.world.batch_dim
        self.pts, self.nrm = contact_spots(self.dev)
        self.K = len(self.pts)
        mass = [BAR[0] * BAR[1], STEM[0] * STEM[1]]
        self.cog = torch.tensor([0.0, STEM_OFFSET * mass[1] / sum(mass)], device=self.dev)
        self.fr, self.rfr = friction_ratio, rot_friction_ratio
        self.push_steps, self.approach_steps = push_steps, approach_steps
        self.p_solo, self.p_random, self.noise, self.walk_cost = p_solo, p_random, noise, walk_cost
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        d = -self.nrm                                            # push direction (inward), T frame
        tau = torch.cross(torch.cat([self.pts - self.cog, torch.zeros_like(d[:, :1])], 1),
                          torch.cat([d, torch.zeros_like(d[:, :1])], 1), dim=1)[:, 2]
        self.F = d[:, None] + d[None]                            # (K, K, 2) pair force
        self.T = tau[:, None] + tau[None]                        # (K, K) pair torque
        sep = (self.pts[:, None] - self.pts[None]).norm(dim=-1)
        self.valid = sep > 2 * AGENT_R + 0.01
        self.mode = torch.zeros(self.B, dtype=torch.long, device=self.dev)  # 0 team approach, 1 team push, 2 solo, 3 random
        self.t = torch.zeros(self.B, dtype=torch.long, device=self.dev)
        self.pair = torch.zeros(self.B, 2, dtype=torch.long, device=self.dev)
        self.rand_act = torch.zeros(self.B, 2, 2, device=self.dev)
        self.solo_agent = torch.zeros(self.B, dtype=torch.long, device=self.dev)
        self.at_out = torch.zeros(self.B, 2, dtype=torch.bool, device=self.dev)
        self.replan(torch.ones(self.B, dtype=torch.bool, device=self.dev))

    def _t_pose(self):
        sc = self.env.scenario
        return sc.bar.state.pos, sc.bar.state.rot[:, 0], sc.goal_pose

    def _choose(self, idx, tabu=None):
        """idx (n,) env indices -> (n, 2) spot indices for agents 0 / 1; tabu (n, K, K) bool masks pairs out."""
        pos, r, goal = self._t_pose()
        pos, r, goal = pos[idx], r[idx], goal[idx]
        e_loc = rot(goal[:, :2] - pos, -r)                        # T error in the T frame
        dth = wrap(goal[:, 2] - r)
        lin = self.F.norm(dim=-1)
        ex_lin = (lin - self.fr).clamp(min=0)
        ex_rot = (self.T.abs() - self.rfr * TIP_R).clamp(min=0) / TIP_R
        u = self.F / lin.clamp(min=1e-6)[..., None]
        en = e_loc.norm(dim=1)
        w_t = (en / 0.2).clamp(max=1)
        w_r = (dth.abs() / math.radians(25)).clamp(max=1)
        align = (u[None] * (e_loc / en.clamp(min=1e-6)[:, None])[:, None, None]).sum(-1)       # (n, K, K)
        score = ex_lin[None] * w_t[:, None, None] * align + ex_rot[None] * torch.sign(self.T)[None] \
            * torch.sign(dth)[:, None, None] * w_r[:, None, None] - 0.3 * ex_rot[None] * (1 - w_r[:, None, None])
        spots = pos[:, None] + rot(self.pts + self.nrm * STANDOFF, r[:, None])               # (n, K, 2) world
        ag = torch.stack([a.state.pos[idx] for a in self.env.world.agents[:2]], 1)           # (n, 2, 2)
        d0, d1 = (spots - ag[:, 0:1]).norm(dim=-1), (spots - ag[:, 1:2]).norm(dim=-1)
        walk = torch.minimum(d0[:, :, None] + d1[:, None], d1[:, :, None] + d0[:, None])
        score = score - self.walk_cost * walk / 2.0
        score = score.masked_fill(~self.valid[None], -1e9)
        if tabu is not None:
            score = score.masked_fill(tabu, -1e9)
        best = score.flatten(1).argmax(1)
        i, j = best // self.K, best % self.K
        swap = d0.gather(1, i[:, None])[:, 0] + d1.gather(1, j[:, None])[:, 0] > d1.gather(1, i[:, None])[:, 0] + d0.gather(1, j[:, None])[:, 0]
        return torch.where(swap[:, None], torch.stack([j, i], 1), torch.stack([i, j], 1))

    def replan(self, mask):
        idx = mask.nonzero()[:, 0]
        if not len(idx):
            return
        u = torch.rand(len(idx), generator=self.g, device=self.dev)
        mode = torch.where(u < self.p_random, 3, torch.where(u < self.p_random + self.p_solo, 2, 0))
        self.mode[idx], self.t[idx] = mode, 0
        self.at_out[idx] = False
        self.pair[idx] = self._choose(idx)
        solo = idx[mode == 2]
        if len(solo):
            self.solo_agent[solo] = torch.randint(0, 2, (len(solo),), generator=self.g, device=self.dev)
            self.pair[solo, 0] = torch.randint(0, self.K, (len(solo),), generator=self.g, device=self.dev)
            self.pair[solo, 1] = self.pair[solo, 0]
        rnd = idx[mode == 3]
        if len(rnd):
            self.rand_act[rnd] = torch.rand(len(rnd), 2, 2, generator=self.g, device=self.dev) * 2 - 1

    def __call__(self):
        """-> list of per-agent actions (B, 2)."""
        pos, r, _ = self._t_pose()
        ag = torch.stack([a.state.pos for a in self.env.world.agents[:2]], 1)
        vel = torch.stack([a.state.vel for a in self.env.world.agents[:2]], 1)
        spot = pos[:, None] + rot(self.pts[self.pair] + self.nrm[self.pair] * STANDOFF, r[:, None])   # (B, 2, 2)
        nw = rot(self.nrm[self.pair], r[:, None])                                                     # world normals
        # approach: orbit the T at a safe radius to a point 0.15 out along the spot's normal, then come straight in
        out = spot + nw * 0.15
        self.at_out |= (ag - out).norm(dim=-1) < 0.05
        c = (pos + rot(self.cog[None].expand(self.B, 2), r))[:, None]  # the T's centre of mass, world
        a_rel, o_rel = ag - c, out - c
        seg = out - ag
        tproj = (((c - ag) * seg).sum(-1) / seg.pow(2).sum(-1).clamp(min=1e-9)).clamp(0, 1)
        # blocked: the straight line to the approach point passes the T (its closest point to the T is not the end)
        blocked = ((ag + tproj[..., None] * seg - c).norm(dim=-1) < ORBIT_R - 0.05) & (tproj < 0.9)
        ang_a, ang_o = torch.atan2(a_rel[..., 1], a_rel[..., 0]), torch.atan2(o_rel[..., 1], o_rel[..., 0])
        step = wrap(ang_o - ang_a).clamp(-0.8, 0.8)
        orbit = c + ORBIT_R * torch.stack([torch.cos(ang_a + step), torch.sin(ang_a + step)], -1)
        target = torch.where(self.at_out[..., None], spot, torch.where(blocked[..., None], orbit, out))
        act = (4.0 * (target - ag) - 0.8 * vel).clamp(-1, 1)
        ready = (ag - spot).norm(dim=-1) < 0.03
        push = (-nw + 1.5 * (spot - ag) * (1 - nw.abs())).clamp(-1, 1)
        m = self.mode
        team_push = (m == 1)[:, None, None]
        act = torch.where(team_push, push, act)
        # solo: one agent goes to its spot and pushes alone, the other wanders slowly
        solo = (m == 2)
        sa = self.solo_agent
        solo_push = solo & ready[torch.arange(self.B), sa]
        sel = torch.zeros(self.B, 2, dtype=torch.bool, device=self.dev)
        sel[torch.arange(self.B), sa] = True
        act = torch.where((solo_push[:, None] & sel)[..., None], push, act)
        act = torch.where((solo[:, None] & ~sel)[..., None], 0.3 * torch.randn(self.B, 2, 2, generator=self.g, device=self.dev).clamp(-1, 1), act)
        act = torch.where((m == 3)[:, None, None], self.rand_act, act)
        act = (act + self.noise * torch.randn(act.shape, generator=self.g, device=self.dev)).clamp(-1, 1)
        # mode transitions
        self.t += 1
        start_push = (m == 0) & ready.all(1)
        self.mode[start_push], self.t[start_push] = 1, 0
        done = ((m == 0) & (self.t > self.approach_steps)) | ((m == 1) & (self.t >= self.push_steps)) \
            | ((m == 2) & (self.t >= 30)) | ((m == 3) & (self.t >= 15))
        self.replan(done)
        return [act[:, 0], act[:, 1]]
