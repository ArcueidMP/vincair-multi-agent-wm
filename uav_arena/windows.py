"""Training windows from `collect.py` episodes, aligned exactly as `openloop.py` scores predictions.

A window (episode, decision k, latency) holds what a predictor sees before decision k at that
latency: the delayed team state, the delayed states of the `history` - 1 decisions before it, and
the last four setpoints with their start times. With it come the setpoints flown over the next
`horizon` decisions and the true team state at the end of each: what `openloop.evaluate` compares
a prediction with. The payload mass is dropped on loading, so it cannot reach training
(PARTICIPANTS.md, Section 3).

    from torch.utils.data import DataLoader
    from uav_arena.windows import Windows, load
    data = Windows(load("data/uav_arena/datasets/explore_v1"), horizon=15, history=4)
    for batch in DataLoader(data, batch_size=256, shuffle=True, num_workers=8):
        ...     # batch["obs_payload_pos"]: (256, 4, 3); batch["next_payload_pos"]: (256, 15, 3)

At the episode start the history repeats the first observation and the setpoints repeat the hover
setpoint held before t = 0; at evaluation, `padded_recent` gives `observe` the same setpoint input.
Using this module is optional, but a participant's own pipeline must keep the same alignment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import DECISION_DT, RECENT_SETPOINTS
from .openloop import PER_DECISION, load_episode, observation
from .scenarios import LATENCIES_MS

STATE_KEYS = ("uav_pos", "uav_vel", "uav_rot6", "uav_omega", "payload_pos", "payload_vel")


def load(directory, seeds=None):
    """Episodes of a `collect.py` directory in seed order (`seeds`: optional start, stop), without the payload mass."""
    paths = sorted(Path(directory).glob("episode_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if seeds is not None:
        paths = [p for p in paths if seeds[0] <= int(p.stem.split("_")[1]) < seeds[1]]
    episodes = [load_episode(p) for p in paths]
    for episode in episodes:
        del episode["payload_mass"]
    return episodes


def decisions(episode):
    """Complete decisions in an episode."""
    return (len(episode["time"])-1)//PER_DECISION


def padded_recent(obs):
    """An observation's setpoints in force, padded to RECENT_SETPOINTS at the episode start with the hover
    setpoint, and their start times relative to the decision: (RECENT_SETPOINTS, 3, 3), (RECENT_SETPOINTS,)."""
    setpoints, times = obs["recent_setpoints"], obs["recent_setpoint_times"]
    missing = RECENT_SETPOINTS-len(setpoints)
    if missing > 0:
        setpoints = np.concatenate([np.repeat(setpoints[:1], missing, axis=0), setpoints])
        times = np.concatenate([times[0]-DECISION_DT*np.arange(missing, 0, -1), times])
    return setpoints, times-obs["decision_time"]


def window(episode, k, latency_ms, horizon=15, history=1, dtype=np.float32):
    """The window before decision k.

    `obs_*` (history, ...): delayed states, oldest first; `latency` and `age` (s): the nominal delay and the newest
    observation's actual age (shorter in the first decisions); `recent_setpoints`, `recent_setpoint_times`: see
    `padded_recent`; `setpoints` (horizon, 3, 3): the segment end points of decisions k ... k + horizon - 1;
    `next_*` (horizon, ...): the true state at the end of each of those decisions.
    """
    if not 0 <= k <= decisions(episode)-horizon:
        raise IndexError(f"decision {k} has no complete {horizon}-decision horizon")
    past = [observation(episode, max(k-j, 0), latency_ms) for j in range(history-1, -1, -1)]
    obs = past[-1]
    recent, recent_times = padded_recent(obs)
    truth = PER_DECISION*np.arange(k+1, k+1+horizon)
    result = {f"obs_{key}": np.stack([p[key] for p in past]) for key in STATE_KEYS}
    result.update(latency=obs["latency"], age=obs["decision_time"]-obs["time"], recent_setpoints=recent,
                  recent_setpoint_times=recent_times, setpoints=episode["setpoints"][k+1:k+1+horizon])
    result.update({f"next_{key}": episode[key][truth] for key in STATE_KEYS})
    return {key: np.asarray(value, dtype=dtype) for key, value in result.items()}


class Windows:
    """Every window of `episodes`: each complete decision at each latency. Map-style, so a PyTorch
    `DataLoader` batches it directly."""

    def __init__(self, episodes, latencies_ms=LATENCIES_MS, horizon=15, history=1, dtype=np.float32):
        self.episodes, self.latencies = episodes, tuple(latencies_ms)
        self.horizon, self.history, self.dtype = horizon, history, dtype
        counts = [max(decisions(e)-horizon+1, 0)*len(self.latencies) for e in episodes]
        self.offsets = np.concatenate([[0], np.cumsum(counts)])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        e = int(np.searchsorted(self.offsets, index, side="right"))-1
        k, latency = divmod(int(index-self.offsets[e]), len(self.latencies))
        return window(self.episodes[e], k, self.latencies[latency], self.horizon, self.history, self.dtype)
