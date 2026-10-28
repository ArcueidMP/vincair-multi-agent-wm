"""UAV entity tokenizer and transformer shared by JEPA and W-ar.

Adapted from ``vmas_pusht.model.EntityJEPA``: type-specific MLP embeddings,
frame encoder, latent projection, and action-conditioned causal predictor.
JEPA objectives, SIGReg and physical-state heads belong to the caller.

The 51-state layout matches june's ``participants/state.py`` implementation:
all positions (9), velocities (9), rotation columns (18), body rates (9),
then payload position (3) and velocity (3). It is NOT aircraft-major.
"""
from __future__ import annotations

import torch
from torch import nn

AIRCRAFT = 3
STATE_DIM = 51
RECENT = 4
TOKENS = 4


def pack_state(state, origin):
    """Pack physical tensors relative to ``origin``; ignore non-state metadata.

    ``origin`` has the state's leading dimensions followed by (3,). For a
    history (B, H, ...), pass (B, 1, 3) to use one origin for every frame.
    This does not fit or apply normalization; callers use training statistics.
    """
    return torch.cat(((state["uav_pos"]-origin[..., None, :]).flatten(-2),
                      state["uav_vel"].flatten(-2), state["uav_rot6"].flatten(-2),
                      state["uav_omega"].flatten(-2), state["payload_pos"]-origin,
                      state["payload_vel"]), dim=-1)


def split_state(state):
    """Field-major (..., 51) -> aircraft (..., 3, 15), payload (..., 6)."""
    if state.ndim < 2 or state.shape[-1] != STATE_DIM:
        raise ValueError("state must have a batch dimension and 51 field-major features")
    aircraft = torch.cat((state[..., :9].unflatten(-1, (AIRCRAFT, 3)),
                          state[..., 9:18].unflatten(-1, (AIRCRAFT, 3)),
                          state[..., 18:36].unflatten(-1, (AIRCRAFT, 6)),
                          state[..., 36:45].unflatten(-1, (AIRCRAFT, 3))), dim=-1)
    return aircraft, state[..., 45:]


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 128), nn.GELU(), nn.Linear(128, outputs))


def attention_mask(frames, tokens, attn):
    """Bool mask: True blocks a key; time is causal, payload is never isolated."""
    index = torch.arange(frames*tokens)
    frame, entity = index//tokens, index % tokens
    mask = frame[None, :] > frame[:, None]
    if attn == "no_cross":
        aircraft = entity < AIRCRAFT
        mask |= aircraft[:, None] & aircraft[None, :] & (entity[:, None] != entity[None, :])
    return mask


class EntityTokenizer(nn.Module):
    """Shared aircraft embedding plus payload embedding, or one flat token."""

    def __init__(self, dim, attn="full"):
        super().__init__()
        self.attn = attn
        if attn == "flat":
            self.flat = mlp(STATE_DIM, dim)
        else:
            self.aircraft = mlp(15, dim)
            self.payload = mlp(6, dim)
            self.type_embedding = nn.Parameter(torch.randn(2, dim)*0.02)

    def forward(self, state):
        aircraft, payload = split_state(state)
        if self.attn == "flat":
            return self.flat(state).unsqueeze(-2)
        return torch.cat((self.aircraft(aircraft)+self.type_embedding[0],
                          (self.payload(payload)+self.type_embedding[1]).unsqueeze(-2)), dim=-2)


class UAVBackbone(nn.Module):
    """Encode observed frames and predict next-step token features.

    forward(state_history, recent_setpoints, recent_times, setpoint, age)
      -> (B, 4, token_dim), or (B, 1, token_dim) for flat.

    History is normalized field-major state, oldest first. Action coordinates
    are unnormalized metres relative to the same observed payload origin.
    Times are seconds relative to the current decision; age is seconds.
    Callers handle rollout, state normalization and latency targets.
    """

    def __init__(self, dim=128, latent=64, enc_layers=2, pred_layers=4, heads=4,
                 history=4, attn="full"):
        super().__init__()
        if attn not in ("full", "no_cross", "flat"):
            raise ValueError("attn must be full, no_cross or flat")
        if min(dim, latent, enc_layers, pred_layers, heads, history) < 1 or dim % heads:
            raise ValueError("dimensions/layers/history must be positive and dim divisible by heads")
        self.config = dict(dim=dim, latent=latent, enc_layers=enc_layers, pred_layers=pred_layers,
                           heads=heads, history=history, attn=attn)
        self.history, self.attn = history, attn
        self.n_tok = 1 if attn == "flat" else TOKENS
        # Match total latent width per frame, without silently scaling hidden
        # transformer width/parameters by four in the flat ablation.
        self.token_dim = latent*TOKENS if attn == "flat" else latent
        self.latent_budget = self.n_tok*self.token_dim
        self.tokenizer = EntityTokenizer(dim, attn)

        def transformer(layers):
            layer = nn.TransformerEncoderLayer(dim, heads, 4*dim, dropout=0.0,
                                               batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

        self.encoder = transformer(enc_layers)
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.token_dim))
        self.unproj = nn.Linear(self.token_dim, dim)
        # Each aircraft gets its own 4x3 past endpoints, four times, next
        # endpoint and age. The payload gets no direct action embedding.
        self.act = mlp(20*AIRCRAFT if attn == "flat" else 20, dim)
        self.time = nn.Parameter(torch.randn(history, dim)*0.02)
        self.tok = nn.Parameter(torch.randn(self.n_tok, dim)*0.02)
        self.predictor = transformer(pred_layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.token_dim))
        self.register_buffer("enc_mask", attention_mask(1, self.n_tok, attn), persistent=False)
        self.register_buffer("pred_mask", attention_mask(history, self.n_tok, attn), persistent=False)

    def encode(self, state):
        """Normalized (..., 51) -> (..., n_tok, token_dim), independently per frame."""
        lead = state.shape[:-1]
        x = self.tokenizer(state.reshape(-1, state.shape[-1]))
        x = self.encoder(x, mask=self.enc_mask)
        return self.proj(x).reshape(*lead, self.n_tok, self.token_dim)

    def _action_tokens(self, recent, times, setpoint, age):
        size = recent.shape[0]
        features = torch.cat((recent.transpose(1, 2).reshape(size, AIRCRAFT, RECENT*3),
                              times[:, None].expand(-1, AIRCRAFT, -1), setpoint,
                              age[:, None, None].expand(-1, AIRCRAFT, 1)), dim=-1)
        if self.attn == "flat":
            return self.act(features.flatten(1))[:, None]
        aircraft = self.act(features)
        return torch.cat((aircraft, aircraft.new_zeros(size, 1, aircraft.shape[-1])), dim=1)

    def _predict_history(self, z, recent, times, setpoint, age):
        """Causal token outputs; the query's action context enters only the last frame."""
        if z.ndim != 4 or z.shape[-2:] != (self.n_tok, self.token_dim):
            raise ValueError("latent history must be (B, H, n_tok, token_dim)")
        size, frames = z.shape[:2]
        if not 1 <= frames <= self.history:
            raise ValueError("history length must be between 1 and the configured history")
        for value, shape in ((recent, (size, RECENT, AIRCRAFT, 3)), (times, (size, RECENT)),
                             (setpoint, (size, AIRCRAFT, 3)), (age, (size,))):
            if tuple(value.shape) != shape:
                raise ValueError(f"action context shape {tuple(value.shape)}, expected {shape}")
        x = self.unproj(z)+self.tok+self.time[-frames:, None]
        action = self._action_tokens(recent, times, setpoint, age)
        # Older outputs must not see the current candidate action, including
        # commands that occurred after those older observed states.
        x = torch.cat((x[:, :-1], x[:, -1:]+action[:, None]), dim=1)
        length = frames*self.n_tok
        x = self.predictor(x.reshape(size, length, -1), mask=self.pred_mask[-length:, -length:])
        return self.head(x.reshape(size, frames, self.n_tok, -1))

    def predict(self, latent_history, recent_setpoints, recent_times, setpoint, age):
        """One latent transition; JEPA can roll this forward without re-encoding states."""
        return self._predict_history(latent_history, recent_setpoints, recent_times, setpoint, age)[:, -1]

    def forward(self, state_history, recent_setpoints, recent_times, setpoint, age):
        if state_history.ndim != 3 or state_history.shape[-1] != STATE_DIM:
            raise ValueError("state_history must be (B, H, 51), in field-major order")
        if not 1 <= state_history.shape[1] <= self.history:
            raise ValueError("history length must be between 1 and the configured history")
        return self.predict(self.encode(state_history), recent_setpoints, recent_times, setpoint, age)


def make_backbone(**options):
    """W-ar's module:factory entry point; options are saved in its checkpoint."""
    return UAVBackbone(**options)
