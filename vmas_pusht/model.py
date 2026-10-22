"""Entity-token JEPA world model for VMAS Push-T: a transformer over agent tokens and a T token, instead of a ViT.

Each frame is a set of tokens: one per agent ([pos, vel]) and one for the T ([pos, sin, cos, vel, angular vel]), each
embedded by its type's MLP plus a type embedding. Agent identity is not embedded: an agent token is told apart only by
its state and by the actions attached to it, so the model is permutation-equivariant over agents and the same weights
serve any team size. The encoder (a transformer over the frame's tokens) gives one latent per token. The predictor
takes the last `history` frames of latents, adds each agent's action block to its own token, attends over all tokens
of those frames (causal in time), and predicts every token's latent one model step (frameskip env steps) ahead.
Training: prediction MSE + SIGReg on the token latents, end to end, as LeWM.

Variants (`attn`), after the relational / independent comparison of the earlier VMAS study:
  full      every token attends to every token (relational: agents see each other and the T)
  no_cross  agent tokens cannot attend to other agent tokens (they still see the T and themselves)
  flat      one token per frame: all entities concatenated (the "joint" baseline, no entity structure)

Readout head (`head=True`, round 5): an MLP per token type from the latent to the token's own features, trained on
encoded and on predicted latents, plus a loss on the predicted *change* of those features over a block, normalised to
unit variance per feature (so a small T motion is a unit-sized target, as in graph-network simulators). The gradient
flows into the encoder and predictor (Strohm et al. 2610.03137, GA-WM 2609.03565): the head is the planner's cost readout.
`sigreg_per_token=True` regularises agent tokens and the T token as two distributions instead of one pool.
`loss(..., k)` trains on k-step open-loop rollouts instead of one teacher-forced step.
"""
import math

import torch
from torch import nn

from module import SIGReg

AGENT_FEATS, T_FEATS = 4, 7


def tokens_from_state(s, n_agents=2):
    """state (..., STATE_DIM) -> agent feats (..., N, 4), T feats (..., 7)."""
    ag = torch.stack([s[..., 4 * i:4 * i + 4] for i in range(n_agents)], -2)
    r = s[..., 10:11]
    t = torch.cat([s[..., 8:10], torch.sin(r), torch.cos(r), s[..., 11:14]], -1)
    return ag, t


def feats_from_state(s, n_agents=2):
    """state (..., STATE_DIM) -> all token features flattened (..., N*4 + 7): [agent 0, agent 1, ..., T]."""
    ag, t = tokens_from_state(s, n_agents)
    return torch.cat([ag.flatten(-2), t], -1)


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class EntityJEPA(nn.Module):
    def __init__(self, n_agents=2, dim=128, latent=64, enc_layers=2, pred_layers=4, heads=4, history=3,
                 act_dim=10, attn="full", sigreg_weight=0.09, head=False, sigreg_per_token=False):
        super().__init__()
        assert attn in ("full", "no_cross", "flat")
        self.N, self.attn, self.history, self.lam = n_agents, attn, history, sigreg_weight
        self.per_token = sigreg_per_token
        self.n_tok = 1 if attn == "flat" else n_agents + 1
        if attn == "flat":
            self.embed_flat = mlp(n_agents * AGENT_FEATS + T_FEATS, dim)
        else:
            self.embed_agent, self.embed_t = mlp(AGENT_FEATS, dim), mlp(T_FEATS, dim)
            self.type_emb = nn.Parameter(torch.randn(2, dim) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer(), enc_layers)
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent))
        self.unproj = nn.Linear(latent, dim)
        self.act = mlp(act_dim * (n_agents if attn == "flat" else 1), dim)
        self.time = nn.Parameter(torch.randn(history, dim) * 0.02)
        self.tok = nn.Parameter(torch.randn(self.n_tok, dim) * 0.02)
        self.predictor = nn.TransformerEncoder(layer(), pred_layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent))
        self.sigreg = SIGReg()
        self.state_head = None
        if head:
            self.state_head = (mlp(latent, n_agents * AGENT_FEATS + T_FEATS) if attn == "flat"
                               else nn.ModuleList([mlp(latent, AGENT_FEATS), mlp(latent, T_FEATS)]))
        self.register_buffer("enc_mask", self._mask(1), persistent=False)
        self.register_buffer("pred_mask", self._mask(history), persistent=False)

    def _mask(self, frames):
        """bool (L, L), True = blocked. Causal in time; no_cross also blocks agent -> other agent."""
        n = self.n_tok
        L = frames * n
        f = torch.arange(L) // n
        k = torch.arange(L) % n
        m = f[None, :] > f[:, None]
        if self.attn == "no_cross":
            is_agent = k < self.N
            m |= is_agent[:, None] & is_agent[None, :] & (k[:, None] != k[None, :])
        return m

    # ---------- encoder ----------
    def encode(self, state):
        """state (..., STATE_DIM) -> token latents (..., n_tok, latent)."""
        lead = state.shape[:-1]
        ag, t = tokens_from_state(state.reshape(-1, state.shape[-1]), self.N)
        if self.attn == "flat":
            x = self.embed_flat(torch.cat([ag.flatten(1), t], -1))[:, None]
        else:
            x = torch.cat([self.embed_agent(ag) + self.type_emb[0], (self.embed_t(t) + self.type_emb[1])[:, None]], 1)
        x = self.encoder(x, mask=self.enc_mask)
        return self.proj(x).reshape(*lead, self.n_tok, -1)

    def decode(self, z):
        """token latents (..., n_tok, latent) -> token features (..., N*4 + 7) in the normalised state's units (head)."""
        assert self.state_head is not None, "model trained without a state head"
        if self.attn == "flat":
            return self.state_head(z[..., 0, :])
        ag = self.state_head[0](z[..., :self.N, :]).flatten(-2)
        return torch.cat([ag, self.state_head[1](z[..., -1, :])], -1)

    # ---------- predictor ----------
    def predict(self, z, act):
        """z (B, H, n_tok, latent) history latents, act (B, H, N, act_dim) each frame's action block per agent
        -> (B, n_tok, latent) the next frame's latents."""
        B, H = z.shape[:2]
        x = self.unproj(z) + self.tok + self.time[-H:, None]
        if self.attn == "flat":
            x = x + self.act(act.flatten(2))[:, :, None]
        else:
            a = self.act(act)                                          # (B, H, N, dim)
            x = torch.cat([x[:, :, :self.N] + a, x[:, :, self.N:]], 2)
        L = H * self.n_tok
        out = self.predictor(x.reshape(B, L, -1), mask=self.pred_mask[-L:, -L:])
        return self.head(out.reshape(B, H, self.n_tok, -1)[:, -1])

    def rollout(self, z_hist, a_hist, a_future):
        """z_hist (B, h, n_tok, latent) past frames, a_hist (B, h - 1, N, act_dim) the blocks executed between them,
        a_future (B, K, N, act_dim) planned blocks -> (B, K, n_tok, latent) the predicted next K frames."""
        z, a, out = z_hist, a_hist, []
        for k in range(a_future.shape[1]):
            a = torch.cat([a, a_future[:, k:k + 1]], 1)               # a[t] = block applied after frame t
            nz = self.predict(z[:, -self.history:], a[:, -self.history:])
            out.append(nz)
            z = torch.cat([z, nz[:, None]], 1)
        return torch.stack(out, 1)

    def _sigreg(self, z):
        if not self.per_token:
            return self.sigreg(z.reshape(1, -1, z.shape[-1]))
        zf = z.reshape(-1, self.n_tok, z.shape[-1])
        if self.n_tok == 1:
            return self.sigreg(zf.reshape(1, -1, z.shape[-1]))
        return 0.5 * (self.sigreg(zf[:, :self.N].reshape(1, -1, z.shape[-1])) + self.sigreg(zf[:, self.N:].reshape(1, -1, z.shape[-1])))

    def loss(self, states, acts, k=1, delta_std=None, head_weight=1.0):
        """states (B, H+k, STATE_DIM) normalised, at model-step spacing; acts (B, H+k-1, N, act_dim) -> loss, logs.
        k = 1: one teacher-forced step; k > 1: open-loop rollout of k blocks, MSE at every block."""
        H = self.history
        z = self.encode(states)                                       # (B, H+k, n_tok, latent)
        if k == 1:
            pred = self.predict(z[:, :H], acts[:, :H])[:, None]
        else:
            pred = self.rollout(z[:, :H], acts[:, :H - 1], acts[:, H - 1:H - 1 + k])
        tgt = z[:, H:H + k]
        l_pred = (pred - tgt).pow(2).mean()
        l_sig = self._sigreg(z)
        loss = l_pred + self.lam * l_sig
        logs = {"pred": l_pred.item(), "sigreg": l_sig.item()}
        if self.state_head is not None:
            f = feats_from_state(states, self.N)                      # (B, H+k, F) normalised units
            f_enc, f_pred = self.decode(z), self.decode(pred)
            l_abs = (f_enc - f).pow(2).mean() + (f_pred - f[:, H:H + k]).pow(2).mean()
            sd = delta_std if delta_std is not None else torch.ones(f.shape[-1], device=f.device)
            d_true = (f[:, H:H + k] - f[:, H - 1:H + k - 1]) / sd
            d_pred = (f_pred - f_enc[:, H - 1:H + k - 1]) / sd
            l_delta = (d_pred - d_true).pow(2).mean()
            loss = loss + head_weight * (l_abs + l_delta)
            logs.update({"head_abs": l_abs.item(), "head_delta": l_delta.item()})
        return loss, logs
