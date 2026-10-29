"""Probabilistic ensemble dynamics (PETS) for the UAV arena.

Each member predicts a Gaussian over the next *physical state increment*.  The
first transition starts at the delayed observation and spans its age plus one
decision.  Later transitions span one decision.  A candidate uses one sampled
member for its entire rollout (one-particle TS-infinity); member assignments
and Gaussian samples are redrawn for each planning call.
"""
from __future__ import annotations

from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from ...env import DECISION_DT
from ...windows import STATE_KEYS, padded_recent
from ..state import STATE_DIM, pack_state, unpack_state

HORIZON = 15
HISTORY = 4
INPUT_DIM = HISTORY * STATE_DIM + 1 + 4 * 9 + 4 + 9


def features(state_history: torch.Tensor, origin: torch.Tensor, age: torch.Tensor,
             recent: torch.Tensor, recent_times: torch.Tensor,
             setpoint: torch.Tensor, center_current: bool = True) -> torch.Tensor:
    """One-transition input. Setpoints are world-frame segment end points."""
    if center_current:
        current = state_history[:, -1, 45:48]
        aircraft = state_history[..., :9].unflatten(-1, (3, 3)) - current[:, None, None, :]
        state_history = torch.cat((aircraft.flatten(-2), state_history[..., 9:45],
                                   state_history[..., 45:48] - current[:, None, :],
                                   state_history[..., 48:51]), dim=-1)
        origin = origin + current
    return torch.cat((state_history.flatten(-2), age[..., None],
                      (recent - origin[..., None, None, :]).flatten(-3),
                      recent_times,
                      (setpoint - origin[..., None, :]).flatten(-2)), dim=-1)


def transition(batch: dict[str, torch.Tensor], steps: torch.Tensor,
               center_current: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced PETS inputs and state increments for selected horizon steps.

    ``steps`` has shape (B,) and selects one of the 15 transitions in each
    windows.py example. The setpoint queue is advanced exactly as in rollout.
    """
    size = len(steps)
    rows = torch.arange(size, device=steps.device)
    observed = {key: batch[f"obs_{key}"] for key in STATE_KEYS}
    origin = observed["payload_pos"][:, -1]
    packed_observed = pack_state(observed, origin[:, None])
    packed_future = pack_state({key: batch[f"next_{key}"] for key in STATE_KEYS}, origin[:, None])
    all_states = torch.cat((packed_observed, packed_future), dim=1)
    history_slots = steps[:, None] + torch.arange(HISTORY, device=steps.device)
    state_history = all_states[rows[:, None], history_slots]
    following = {key: batch[f"next_{key}"][rows, steps] for key in STATE_KEYS}
    delta = pack_state(following, origin) - state_history[:, -1]
    all_recent = torch.cat((batch["recent_setpoints"], batch["setpoints"]), dim=1)
    slots = steps[:, None] + torch.arange(4, device=steps.device)
    recent = all_recent[rows[:, None], slots]
    future_times = DECISION_DT * torch.arange(batch["setpoints"].shape[1], device=steps.device,
                                               dtype=batch["recent_setpoint_times"].dtype)
    all_times = torch.cat((batch["recent_setpoint_times"], future_times[None].expand(size, -1)), dim=1)
    recent_times = all_times[rows[:, None], slots] - steps[:, None] * DECISION_DT
    age = torch.where(steps == 0, batch["age"], torch.zeros_like(batch["age"]))
    x = features(state_history, origin, age, recent, recent_times,
                 batch["setpoints"][rows, steps], center_current)
    return x, delta


class ProbabilisticMLP(nn.Module):
    """Four Swish hidden layers and bounded diagonal Gaussian log variance."""

    def __init__(self, width: int = 200, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        size = INPUT_DIM
        for _ in range(depth):
            layers.extend((nn.Linear(size, width), nn.SiLU()))
            size = width
        layers.append(nn.Linear(size, 2 * STATE_DIM))
        self.net = nn.Sequential(*layers)
        self.max_logvar = nn.Parameter(torch.full((STATE_DIM,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((STATE_DIM,), -10.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_logvar = self.net(x).chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def nll(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean, logvar = self(x)
        likelihood = ((mean - target).square() * torch.exp(-logvar) + logvar).sum(-1).mean()
        return likelihood + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())


class PETSPredictor:
    stochastic = True

    def __init__(self, checkpoint: str, device: str | None = None, seed: int = 0,
                 sample_gaussian: bool = False):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        pack = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = pack["config"]
        if config.get("history") != HISTORY:
            raise ValueError(f"checkpoint history {config.get('history')} does not match PETS history {HISTORY}")
        self.center_current = bool(config.get("center_current", False))
        self.members = nn.ModuleList([
            ProbabilisticMLP(config["width"], config["depth"]) for _ in range(config["members"])
        ]).to(self.device).eval()
        self.members.load_state_dict(pack["state"])
        self.stats = {key: value.to(self.device) for key, value in pack["stats"].items()}
        self.seed = seed
        self.sample_gaussian = sample_gaussian
        self.reset()

    def reset(self):
        self.obs = None
        self.frames = deque(maxlen=HISTORY)
        torch.manual_seed(self.seed)

    def observe(self, obs, layout):
        # The public layout is deliberately unused; in particular there is no
        # path from its diagnostic payload_mass field into this predictor.
        recent, recent_times = padded_recent(obs)
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
        state = {key: tensor(obs[key]) for key in STATE_KEYS}
        if not self.frames:
            for _ in range(HISTORY - 1):
                self.frames.append(state)
        self.frames.append(state)
        self.obs = dict(
            age=tensor([obs["decision_time"] - obs["time"]]),
            recent=tensor(recent)[None], recent_times=tensor(recent_times)[None],
        )

    @torch.no_grad()
    def __call__(self, info, setpoints):
        if self.obs is None:
            raise RuntimeError("observe must be called before PETS prediction")
        leading, horizon = setpoints.shape[:-3], setpoints.shape[-3]
        candidates = setpoints.to(self.device, torch.float32).reshape(-1, horizon, 3, 3)
        size = len(candidates)
        origin = self.frames[-1]["payload_pos"][None].expand(size, -1)
        history = {key: torch.stack([frame[key] for frame in self.frames], dim=0)[None]
                   for key in STATE_KEYS}
        state_history = pack_state(
            {key: value.expand(size, *value.shape[1:]) for key, value in history.items()}, origin[:, None])
        recent = self.obs["recent"].expand(size, -1, -1, -1)
        recent_times = self.obs["recent_times"].expand(size, -1)
        member_ids = torch.randint(len(self.members), (size,), device=self.device)
        all_states = []
        for step in range(horizon):
            age = self.obs["age"].expand(size) if step == 0 else torch.zeros(size, device=self.device)
            x = features(state_history, origin, age, recent, recent_times,
                         candidates[:, step], self.center_current)
            x = (x - self.stats["x_mean"]) / self.stats["x_std"]
            delta = torch.empty_like(state_history[:, -1])
            for member_id, member in enumerate(self.members):
                rows = torch.nonzero(member_ids == member_id, as_tuple=True)[0]
                if len(rows) == 0:
                    continue
                mean, logvar = member(x.index_select(0, rows))
                if self.sample_gaussian:
                    noise = torch.randn(mean.shape, device=self.device)
                    sample = mean + noise * torch.exp(0.5 * logvar)
                else:
                    sample = mean
                delta.index_copy_(0, rows, sample * self.stats["y_std"] + self.stats["y_mean"])
            state = state_history[:, -1] + delta
            all_states.append(state)
            state_history = torch.cat((state_history[:, 1:], state[:, None]), dim=1)
            recent = torch.cat((recent[:, 1:], candidates[:, step:step + 1]), dim=1)
            recent_times = torch.cat((recent_times[:, 1:] - DECISION_DT,
                                      torch.full((size, 1), -DECISION_DT, device=self.device)), dim=1)
        state = torch.stack(all_states, dim=1)
        full = unpack_state(state, origin[:, None])
        prediction = dict(uav=full["uav_pos"], payload=full["payload_pos"], payload_vel=full["payload_vel"])
        return {key: value.reshape(*leading, *value.shape[1:]).to(setpoints.device, setpoints.dtype)
                for key, value in prediction.items()}


def make(checkpoint: str, device: str | None = None, seed: int = 0,
         sample_gaussian: bool = False):
    """Arena ``module:factory`` entry point."""
    return PETSPredictor(checkpoint, device=device, seed=seed, sample_gaussian=sample_gaussian)
