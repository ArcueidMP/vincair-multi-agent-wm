"""Seeded arena scenarios: start, goal, payload and obstacles under eight conditions.

A condition crosses two obstacle settings (static pillars; the same pillars plus
moving spheres) with four observation latencies. For one seed the layout is
shared by every condition, so obstacle and latency effects are paired. Moving
obstacles follow pure functions of time, so any episode replays exactly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import math

import numpy as np


OBSTACLE_SETTINGS = ("static", "static+moving")
LATENCIES_MS = (0, 50, 100, 200)
WORKSPACE = ((-3.0, 3.0), (-2.5, 2.5), (0.0, 3.0))   # x, y, z bounds in metres
PILLAR_HEIGHT = 3.0
SUITE_SEEDS = dict(dev=tuple(range(20)), test=tuple(range(1000, 1050)))


@dataclass(frozen=True)
class Condition:
    obstacles: str = "static"
    latency_ms: int = 0

    def __post_init__(self):
        if self.obstacles not in OBSTACLE_SETTINGS or self.latency_ms not in LATENCIES_MS:
            raise ValueError(f"Unknown condition {self}")

    @property
    def name(self):
        return f"{self.obstacles}/{self.latency_ms}ms"


CONDITIONS = tuple(Condition(o, d) for o in OBSTACLE_SETTINGS for d in LATENCIES_MS)


@dataclass(frozen=True)
class Pillar:
    """Static vertical cylinder from the floor to PILLAR_HEIGHT."""
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class Mover:
    """Sphere moving back and forth on the segment a-b at constant speed."""
    a: tuple
    b: tuple
    speed: float
    phase: float
    radius: float

    def state(self, t):
        a, b = np.asarray(self.a, dtype=float), np.asarray(self.b, dtype=float)
        length = float(np.linalg.norm(b-a))
        period = 2*length/self.speed
        s = ((t/period+self.phase) % 1.0)*2.0          # 0..2 along a->b->a
        forward = s < 1.0
        fraction = s if forward else 2.0-s
        direction = (b-a)/length
        return a+fraction*(b-a), (1.0 if forward else -1.0)*self.speed*direction


@dataclass(frozen=True)
class Scenario:
    seed: int
    condition: Condition
    start: tuple                     # payload position at the hover start
    goal: tuple                      # payload goal position
    pillars: tuple
    movers: tuple = ()
    payload_mass: float = 0.03
    time_limit: float = 20.0
    goal_tolerance: float = 0.15
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def mover_states(self, t):
        if not self.movers:
            return np.zeros((0, 3)), np.zeros((0, 3))
        states = [m.state(t) for m in self.movers]
        return np.stack([s[0] for s in states]), np.stack([s[1] for s in states])

    def to_dict(self):
        values = asdict(self)
        values["condition"] = dict(obstacles=self.condition.obstacles, latency_ms=self.condition.latency_ms)
        return values


def _free_path(pillars, start, goal, clearance, resolution=0.05):
    """Grid search in the plane: can a disc of radius `clearance` pass the pillars?"""
    (x0, x1), (y0, y1), _ = WORKSPACE
    xs, ys = np.arange(x0, x1+1e-9, resolution), np.arange(y0, y1+1e-9, resolution)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    free = (X-x0 >= clearance) & (x1-X >= clearance) & (Y-y0 >= clearance) & (y1-Y >= clearance)
    for p in pillars:
        free &= np.hypot(X-p.x, Y-p.y) >= p.radius+clearance
    index = lambda point: (int(round((point[0]-x0)/resolution)), int(round((point[1]-y0)/resolution)))
    begin, end = index(start), index(goal)
    if not (free[begin] and free[end]):
        return False
    seen, queue = {begin}, deque([begin])
    while queue:
        i, j = queue.popleft()
        if (i, j) == end:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i+di, j+dj)
            if 0 <= n[0] < free.shape[0] and 0 <= n[1] < free.shape[1] and free[n] and n not in seen:
                seen.add(n); queue.append(n)
    return False


def team_clearance(formation_radius=0.25, uav_radius=0.06, margin=0.10):
    """Planar radius the whole team needs to pass between pillars."""
    return formation_radius+uav_radius+margin


def make_scenario(seed, condition, *, n_pillars=4, n_movers=2, mover_radius=0.15,
                  pillar_radius=(0.10, 0.18), mover_speed=(0.25, 0.45), payload_mass=0.03):
    """Layout depends only on the seed; movers are added only in the moving setting."""
    layout = np.random.default_rng([int(seed), 7])
    start = (-1.6, float(layout.uniform(-0.4, 0.4)), float(layout.uniform(1.0, 1.4)))
    goal = (1.6, float(layout.uniform(-0.4, 0.4)), float(layout.uniform(1.0, 1.4)))
    clearance = team_clearance()
    for attempt in range(1000):
        pillars = []
        while len(pillars) < n_pillars:
            p = Pillar(float(layout.uniform(-0.8, 0.8)), float(layout.uniform(-1.0, 1.0)),
                       float(layout.uniform(*pillar_radius)))
            if all(math.hypot(p.x-q.x, p.y-q.y) > p.radius+q.radius+0.15 for q in pillars):
                pillars.append(p)
        if _free_path(pillars, start, goal, clearance):
            break
    else:
        raise RuntimeError(f"No feasible pillar layout for seed {seed}")
    movers = ()
    if condition.obstacles == "static+moving":
        motion = np.random.default_rng([int(seed), 11])
        movers = tuple(Mover(a=(float(motion.uniform(-0.7, 0.7)), -1.3, float(motion.uniform(start[2]-0.1, start[2]+0.6))),
                             b=(float(motion.uniform(-0.7, 0.7)), 1.3, float(motion.uniform(start[2]-0.1, start[2]+0.6))),
                             speed=float(motion.uniform(*mover_speed)), phase=float(motion.uniform()),
                             radius=float(mover_radius)) for _ in range(n_movers))
    return Scenario(seed=int(seed), condition=condition, start=start, goal=goal, pillars=tuple(pillars),
                    movers=movers, payload_mass=float(payload_mass), meta=dict(layout_attempts=attempt+1))


def suite(name, conditions=CONDITIONS, **options):
    """Every condition on every seed of a named suite; seeds are frozen with the code."""
    return [make_scenario(seed, condition, **options) for condition in conditions for seed in SUITE_SEEDS[name]]
