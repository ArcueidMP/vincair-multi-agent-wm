# W-ar shared-backbone interface

W-ar owns the physical-state head, multistep state loss, training loop and arena
adapter. Yihao owns the shared entity tokenizer and transformer, including the
`full` attention mask. The agreed factory is
`uav_arena.participants.jepa.backbone:make_backbone`:

```python
backbone = make_backbone(history=4, attn="full")
backbone.token_dim: int
tokens = backbone(state_history, recent_setpoints, recent_times, setpoint, age)
# tokens.shape == (batch, 4, backbone.token_dim)
```

Inputs are `float32` tensors on the backbone's device:

| Input | Shape | Meaning |
| --- | --- | --- |
| `state_history` | `(B, history, 51)` | normalized field-major physical states, oldest first; slices are listed below |
| `recent_setpoints` | `(B, 4, 3, 3)` | last four setpoints, relative to the newest observed payload position |
| `recent_times` | `(B, 4)` | those setpoints' start times relative to the current decision, in seconds |
| `setpoint` | `(B, 3, 3)` | current candidate segment endpoint, relative to the newest observed payload position |
| `age` | `(B,)` | age of the delayed observation in seconds; zero after the first predicted step |

The 51-state vector follows the executable field-major order in
`participants/state.py`, not an aircraft-major order:

| Slice | Contents |
| --- | --- |
| `0:9` | all three aircraft positions |
| `9:18` | all three aircraft velocities |
| `18:36` | all three aircraft rotation columns |
| `36:45` | all three aircraft body rates |
| `45:48` | payload position |
| `48:51` | payload velocity |

Token order is aircraft 0, 1, 2, payload. The head maps those four tokens to
normalized physical-state increments. No latent target or SIGReg is used. The
first increment predicts across the observation's age plus one decision. The
same module rolls out all 15 decisions, keeping gradients through every step.
The backbone should attach each aircraft's action inputs to that aircraft's
token and use the JEPA `full` attention mask, so the backbone is identical to
the `full` variant apart from W-ar's direct-state objective and head.

The Phase A-compatible command-line options supplied with block 2 are:

```bash
--backbone uav_arena.participants.jepa.backbone:make_backbone \
--history 4 \
--backbone-options '{"history":4,"attn":"full","dim":128,"latent":64,"enc_layers":2,"pred_layers":4,"heads":4}'
```

The training CLI defaults to the standard `explore_v2` dataset. It rejects a
mismatch between `--history` and the backbone option. The factory path and
options are stored in the checkpoint so
`uav_arena.participants.war.model:make` can rebuild it for evaluation. The
options must select the JEPA `full` attention mask. The delivered default full
backbone has 1,271,424 parameters, excluding W-ar's head and JEPA's future
readout. Final structural options and capacity matching remain coupled to the
JEPA full row and must be frozen together before formal training.

Local integration verification used the delivered default full backbone:

- the 30 shared-backbone tests plus five W-ar tests passed together;
- a deliberately untrained smoke run used two `explore_v2` episodes, batch 4,
  a 15-step loss and three optimizer steps on an RTX 5060 Laptop GPU;
- the 1,281,365-parameter backbone-plus-head produced finite gradients and a
  finite final smoke loss. This is not a Phase A training result.
