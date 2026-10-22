"""MLP probes from token latents to state quantities, fitted on training tokens, with their held-out error reported so
that every decoded quantity in the audits can be read against the probe's own accuracy."""
import torch
from torch import nn


def fit_mlp_probe(Z, Y, steps=3000, hidden=256, seed=0):
    """Z (n, d) -> Y (n, m). Returns (probe, held-out median error): the last 10% of rows are held out."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(Z)
    cut = int(0.9 * n)
    mu, sd = Y[:cut].mean(0), Y[:cut].std(0).clamp(min=1e-6)
    net = nn.Sequential(nn.Linear(Z.shape[1], hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, Y.shape[1])).to(Z.device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    with torch.enable_grad():
        for _ in range(steps):
            idx = torch.randint(cut, (1024,), generator=g).to(Z.device)
            loss = (net(Z[idx]) - (Y[idx] - mu) / sd).pow(2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    net.eval()
    f = lambda z: net(z) * sd + mu
    with torch.no_grad():
        err = float((f(Z[cut:]) - Y[cut:]).norm(dim=-1).median())
        base = float((Y[cut:] - mu).norm(dim=-1).median())
    return f, {"heldout_median_err": err, "predict_mean_err": base}
