"""Oracle subgoals for VMAS Push-T, vectorized: the scripted team replays every episode from the same start, and its states
become a schedule (as subgoal.py does for MultiPush-T): at each replan the planner is re-anchored to the closest scripted
state at most `window` steps past the last anchor, and the subgoal is the scripted state `lookahead` steps after that anchor
(or the scripted run's solved state).

    states, solved_at = scripted_rollout(s0, goal, steps=600, device="cuda")   # (E, steps+1, D), (E,) step of first success or -1
    sched = Schedule(states, solved_at, lookahead=15, window=15)
    sg = sched.update(state(env))                                               # (E, D) subgoal states
"""
import math

import torch

from vmas_pusht.policy import ScriptedTeam
from vmas_pusht.scenario import make, set_state, state, wrap

ANGLE_W = 4 * 0.05 / math.radians(20)   # 20 deg costs like 4 x pos_tol, as scenario.pose_err


def state_dist(a, b):
    """(..., D) x (..., D) -> (...): both agents' positions + T position + weighted T angle."""
    pos = (a[..., 0:2] - b[..., 0:2]).norm(dim=-1) + (a[..., 4:6] - b[..., 4:6]).norm(dim=-1) + (a[..., 8:10] - b[..., 8:10]).norm(dim=-1)
    return pos + ANGLE_W * wrap(a[..., 10] - b[..., 10]).abs()


@torch.no_grad()
def scripted_rollout(s0, goal, steps=600, device="cuda", seed=0, **team_kwargs):
    """Noise-free scripted team from states s0 (E, D) with goals (E, 3) -> states (E, steps+1, D), solved_at (E,)."""
    E = s0.shape[0]
    env = make(num_envs=E, device=device, seed=seed)
    env.reset()
    set_state(env, s0.to(device), goal.to(device))
    team = ScriptedTeam(env, p_solo=0.0, p_random=0.0, noise=0.0, seed=seed, **team_kwargs)
    states, solved_at = [state(env)], torch.full((E,), -1, device=device, dtype=torch.long)
    for t in range(steps):
        env.step(team())
        s = state(env)
        ok = env.scenario.success() & (solved_at < 0)
        solved_at[ok] = t + 1
        states.append(s)
    states = torch.stack(states, 1)
    # freeze every episode at its first solved state, so the schedule ends there
    for e in torch.nonzero(solved_at >= 0)[:, 0].tolist():
        states[e, solved_at[e]:] = states[e, solved_at[e]]
    return states, solved_at


class Schedule:
    def __init__(self, states, solved_at, lookahead=15, window=15):
        self.states, self.L, self.W = states, lookahead, window
        E, T1, _ = states.shape
        self.end = torch.where(solved_at >= 0, solved_at, torch.full_like(solved_at, T1 - 1))   # last useful index per episode
        self.k = torch.zeros(E, dtype=torch.long, device=states.device)

    def update(self, cur):
        """cur (E, D) -> subgoal (E, D); re-anchors within [k, k + W]."""
        E = cur.shape[0]
        idx = (self.k[:, None] + torch.arange(self.W + 1, device=cur.device)[None]).clamp(max=self.end[:, None])
        cand = self.states[torch.arange(E, device=cur.device)[:, None], idx]                  # (E, W+1, D)
        j = state_dist(cur[:, None], cand).argmin(1)
        self.k = idx[torch.arange(E, device=cur.device), j]
        tgt = (self.k + self.L).clamp(max=self.end)
        return self.states[torch.arange(E, device=cur.device), tgt]


def goal_state(s, goal):
    """The eval goal as a state: the current state with the T at its goal pose and at rest."""
    g = s.clone()
    g[:, 8:11] = goal
    g[:, 11:14] = 0
    return g


class RoleSchedule:
    """Roles-only coordinator, vectorised (port of subgoal.RoleSchedule from MultiPush-T, round 5: 35 / 95 there).

    At each replan the scripted team's pair chooser runs on the current state (no scripted rollout) and the subgoal puts
    agent 0 / 1 at their contact spots with the T where it is and everything at rest. The pair is kept until both agents
    stand within `ready` of their spots (a pair not reached after `patience` replans is set aside for `patience` more).
    Then the subgoal is the push: the T moved toward its goal by at most `step` / `step_rad`, with the agents at the same
    spots on the moved T (push=False hands back the eval goal instead). Once the T is within the success tolerance, the
    subgoal is the eval goal. A pushed pair is re-chosen at the next replan (the chooser is deterministic, so a pair that
    still works is kept).
    """

    def __init__(self, env, goal, step=0.15, step_rad=math.radians(25), ready=0.05, patience=2, push=True):
        self.env, self.goal, self.step, self.step_rad, self.ready, self.patience, self.push = env, goal, step, step_rad, ready, patience, push
        self.team = ScriptedTeam(env, p_solo=0.0, p_random=0.0, noise=0.0)
        E, dev, K = env.world.batch_dim, env.world.device, self.team.K
        self.pair = torch.zeros(E, 2, dtype=torch.long, device=dev)
        self.has = torch.zeros(E, dtype=torch.bool, device=dev)
        self.waited = torch.zeros(E, dtype=torch.long, device=dev)
        self.tabu_until = torch.zeros(E, K, K, dtype=torch.long, device=dev)
        self.n_push = torch.zeros(E, dtype=torch.long, device=dev)
        self.t = 0

    def _spots(self, pos, r, pair):
        from vmas_pusht.policy import STANDOFF, rot
        return pos[:, None] + rot(self.team.pts[pair] + self.team.nrm[pair] * STANDOFF, r[:, None])   # (E, 2, 2)

    def update(self, cur):
        """cur (E, D) -> subgoal (E, D)."""
        from vmas_pusht.policy import rot
        self.t += 1
        sc, goal = self.env.scenario, self.goal
        pos, r = sc.bar.state.pos, sc.bar.state.rot[:, 0]
        d = goal[:, :2] - pos
        dth = wrap(goal[:, 2] - r)
        placed = (d.norm(dim=1) < sc.pos_tol) & (dth.abs() < sc.rot_tol)
        need = ~self.has & ~placed
        if need.any():
            idx = need.nonzero()[:, 0]
            self.pair[idx] = self.team._choose(idx, tabu=self.tabu_until[idx] > self.t)
            self.has[idx], self.waited[idx] = True, 0
        E = cur.shape[0]
        ar = torch.arange(E, device=cur.device)
        spots = self._spots(pos, r, self.pair)
        ag = torch.stack([a.state.pos for a in self.env.world.agents[:2]], 1)
        ready = ((ag - spots).norm(dim=-1) < self.ready).all(1) & self.has & ~placed
        sg = cur.clone()
        sg[:, 2:4], sg[:, 6:8], sg[:, 11:14] = 0, 0, 0
        sg[:, 0:2], sg[:, 4:6] = spots[:, 0], spots[:, 1]
        if ready.any():
            if self.push:
                e = d.norm(dim=1, keepdim=True)
                npos = pos + d * (self.step / e.clamp(min=1e-6)).clamp(max=1.0)
                nrot = wrap(r + dth.clamp(-self.step_rad, self.step_rad))
                nsp = self._spots(npos, nrot, self.pair)
                sg[ready, 8:10], sg[ready, 10] = npos[ready], nrot[ready]
                sg[ready, 0:2], sg[ready, 4:6] = nsp[ready, 0], nsp[ready, 1]
            else:
                sg[ready] = goal_state(cur, goal)[ready]
            self.n_push[ready] += 1
            self.has[ready] = False
        self.waited[self.has & ~ready] += 1
        late = self.waited > self.patience
        if late.any():
            idx = late.nonzero()[:, 0]
            i, j = self.pair[idx, 0], self.pair[idx, 1]
            self.tabu_until[idx, i, j] = self.t + self.patience
            self.tabu_until[idx, j, i] = self.t + self.patience
            self.has[idx] = False
        if placed.any():
            sg[placed] = goal_state(cur, goal)[placed]
            self.has[placed] = False
        return sg
