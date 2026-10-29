# Shared UAV backbone (block 2)

Factory: **`uav_arena.participants.jepa.backbone:make_backbone`**.
This module adapts `vmas_pusht/model.py`'s type-specific token embeddings,
frame encoder, latent bottleneck and causal action-conditioned predictor to
the arena. It contains no JEPA loss, SIGReg, state head or training loop yet.
Those are the next block; W-ar supplies its own direct-state head and loss.

## W-ar import contract

```python
from uav_arena.participants.jepa.backbone import make_backbone

backbone = make_backbone(history=4, attn="full")
tokens = backbone(state_history, recent_setpoints, recent_times, setpoint, age)
# tokens: (B, 4, backbone.token_dim), aircraft 0, 1, 2, payload
```

Pass the same `history` in W-ar's CLI and `--backbone-options`. Defaults:
`history=4, dim=128, latent=64, enc_layers=2, pred_layers=4, heads=4`.
Dropout is zero. `backbone.config` is sufficient to rebuild the architecture;
save it with `state_dict`. Normalization statistics belong to the caller's
checkpoint, not the backbone. Do not pass mass, layout or simulator state.

| Argument | Shape | Units and meaning |
|---|---|---|
| `state_history` | `(B, H, 51)`, `1 <= H <= history` | Normalized physical states, oldest first; field-major order below |
| `recent_setpoints` | `(B, 4, 3, 3)` | Metres relative to the latest observed payload; padded with the hover endpoint at episode start |
| `recent_times` | `(B, 4)` | Endpoint command start times relative to the current decision, in seconds |
| `setpoint` | `(B, 3, 3)` | Next candidate segment endpoint, metres relative to the same origin |
| `age` | `(B,)` | Latest observation's actual age in seconds, not always the nominal latency |

Inputs must share the module's device and floating dtype (W-ar uses float32).
The backbone preserves them. It is stateless: episode history and `reset()`
are owned by the participant adapter. No future state is an input.

**Important layout correction:** june's README says “3 aircraft x 15 values,
then payload”, but the actual `participants/state.py::pack_state` and W-ar
objective use a **field-major** 51-vector:

| Slice | Contents |
|---|---|
| `0:9` | All three aircraft positions |
| `9:18` | All three aircraft velocities |
| `18:36` | All three aircraft rotation columns (6 per aircraft) |
| `36:45` | All three aircraft body rates |
| `45:48` | Payload position |
| `48:51` | Payload velocity |

This factory follows the executable layout. `split_state` assembles each
aircraft's 15 features from those slices. `pack_state(state, origin)` is a
compatible utility, reading only six named state fields. For a history,
use the same newest observed payload origin for **all** frames; do not centre
each historical frame separately. Fit moments only on training data.

## Delays, actions and rollout

The first requested transition ends one decision after the current decision
time: its prediction spans `age + 0.1 s` from the latest observation. The
backbone receives age and all four past commands; the objective must use the
matching target supplied by `windows.py`, without an extra index shift.
This is a learned delay bridge, not a simulator or a hard-coded integrator.

Each aircraft's four past endpoints, their times, its candidate endpoint and
age form that aircraft's 20-value action input. The payload has no direct
action embedding. Action context enters the newest history frame only;
the transformer uses a causal time mask, so older outputs cannot see it.

- **W-ar:** call `forward` after every physical-state update; it re-encodes
  normalized predicted states. Its head predicts state increments. W-ar's
  existing rollout updates recent commands/times and sets age to zero after
  the first transition. Keep gradients through the whole rollout.
- **JEPA:** use `encode(state)` for observed/target latents, and
  `predict(latent_history, recent_setpoints, recent_times, setpoint, age)`
  for latent autoregression. Append the predicted latents to the history,
  update the action context on each step, then decode with JEPA's readout.
  SIGReg belongs to JEPA training only; prediction itself is deterministic.

The two objectives share the actual tokenizer/transformer implementation,
not trained weights. W-ar must select `attn="full"`. Final history/width
and total parameter counts should be checked once the JEPA readout exists;
this block does not assert final model-capacity equality.

## Variants and capacity

- `full`: four tokens, all entity pairs visible at permitted times.
- `no_cross`: four tokens, aircraft queries cannot directly attend to another
  aircraft; self and payload remain visible. Multiple layers can transmit
  cross-aircraft information **through the payload**. This is not complete
  causal independence.
- `flat`: one token per frame. `latent` means per-entity width in the
  structured variants, so the flat token automatically has width `4*latent`.
  All variants have `latent_budget = n_tok * token_dim = 4*latent`.
  Hidden transformer `dim` stays fixed; parameter counts can differ and must
  be reported. Flat is not compatible with W-ar's four-token state head.

`full` and `no_cross` have identical learned parameter shapes; only their
attention masks differ. The encoder and predictor masks both enforce the
variant. Learned token/time embeddings follow the upstream implementation;
no claim of full permutation equivariance is made.

## Checks and handoff

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  data/uav_arena/.venv/bin/python -m pytest -q tests/test_uav_jepa_backbone.py
```

The tests cover field order, payload-relative packing, excluded metadata,
candidate batch sizes 1/512/555, action ownership, causal masks and numeric
causal isolation, gradient flow over 15 steps, deterministic repeated calls,
checkpoint reconstruction and invalid shapes.

Suggested W-ar arguments, once this module is available in june's checkout:

```bash
--backbone uav_arena.participants.jepa.backbone:make_backbone \
--history 4 \
--backbone-options '{"history":4,"attn":"full","dim":128,"latent":64,"enc_layers":2,"pred_layers":4,"heads":4}'
```

Integration was checked against the W-ar branch revision recorded in the
block-2 report. This is a code-compatibility check, **not** confirmation from
the teammate, not Phase A training, and not a trained JEPA participant.
