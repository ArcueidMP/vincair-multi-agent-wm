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

## Training recipe

The maintainer trains W-ar from 2026-11-01 (PR #2), starting from the code
above at `f837514`. The optimizer follows JEPA's C/D recipe
(`participants/jepa/train.py` on PR #3), so W-ar and JEPA `full` differ only in
the objective and the head: AdamW with weight decay 1e-3; a one-cycle learning
rate that peaks at `--lr` (default 1e-3) after `min(0.5, max(0.05, 2/steps))`
of the run; the gradient norm clipped at 1; batch 128 by default; 15-step
rollouts; statistics from 5,000 training windows. A nonfinite loss or gradient
stops the run. JEPA's SIGReg, latent targets and head weight have no W-ar
counterpart.

History and backbone options are not tuned for W-ar. They follow JEPA `full`'s
Phase B selection: history 4 (G4 on PR #3, 2026-11-01) with the options above.
If a longer JEPA schedule changes that selection, W-ar follows it. W-ar's
backbone plus state head has 1,281,365 parameters against JEPA `full`'s
1,289,225 with its readout; the difference of 7,860 is reported, not padded.

Each run writes `<out>.pt` and `<out>.json` and refuses to overwrite either.
The JSON records the arguments, the selected episode seeds, a log every 100
steps (loss, gradient norm, learning rate, seconds), the parameter count, wall
seconds, peak GPU memory, the GPU, torch and host, the dataset manifest's
SHA-256 and the SHA-256 of the W-ar, backbone and state sources.
