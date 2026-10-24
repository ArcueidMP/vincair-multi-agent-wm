"""State readout for the MultiPush-T models: an MLP probe from a model's latent z (encoder CLS + projector, 192-d)
to the success test's quantities [self x, self y, T x, T y, T sin, T cos] (px, unit circle).

Fit by scripts/fit_readout.py on the model's encoder latents of the held-out coop scenes; shipped as
checkpoints/<model>/readout/cost_probe.pt (a subfolder: load_pretrained wants a single .pt next to config.json).
Used for the readout cost (plan.py cost=probe) and scripts/cost_audit.py.
"""
import numpy as np
import torch
from torch import nn

TARGETS = ["self x", "self y", "T x", "T y", "T sin", "T cos"]
ANGLE_W = 20.0 / np.radians(20)  # 20 deg weighs like 20 px, the two success tolerances


def make_probe(d_in, d_out=len(TARGETS)):
    return nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, d_out))


def load_probe(path, device="cuda"):
    """cost_probe.pt -> f: z (..., D) -> pose (..., 6) in px / unit circle, plus the file's dict (metrics, split)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    probe = make_probe(ck["in_dim"], ck["out_dim"])
    probe.load_state_dict(ck["state_dict"])
    probe = probe.to(device).eval().requires_grad_(False)
    m, s = ck["mean"].to(device), ck["std"].to(device)
    return (lambda z: probe(z.float()) * s + m), ck


def pose_cost(a, b):
    """|self xy|^2 + |T xy|^2 + (ANGLE_W * T angle)^2 between two poses (..., 6), px^2. Works on torch or numpy."""
    lib = torch if torch.is_tensor(a) else np
    ang = lib.arctan2(a[..., 4], a[..., 5]) - lib.arctan2(b[..., 4], b[..., 5])
    ang = lib.arctan2(lib.sin(ang), lib.cos(ang))
    return ((a[..., :4] - b[..., :4]) ** 2).sum(-1) + (ANGLE_W * ang) ** 2


def state_pose(global_state, agent, n_agents=2):
    """Global states (..., 4N + 3) [agents xy, T xy, T angle, agents v] -> agent's pose (..., 6), the probe's targets."""
    g = np.asarray(global_state, dtype=np.float32)
    N = n_agents
    T, ang = g[..., 2 * N:2 * N + 2], g[..., 2 * N + 2]
    return np.concatenate([g[..., 2 * agent:2 * agent + 2], T, np.sin(ang)[..., None], np.cos(ang)[..., None]], -1)
