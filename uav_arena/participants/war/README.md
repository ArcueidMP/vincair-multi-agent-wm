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

## Selection

Phase A and Phase B ran on 2026-11-01 from `f65e007`, with the recipe, history
and backbone options above, on AIHUB's `aihub1`: one RTX 4090, shared by three
runs at a time. Scores are `openloop.py`'s selection score on the validation
set: the payload error over the 15-decision horizon, averaged over the four
latencies (PARTICIPANTS.md, Section 6).

- **Phase A.** The first 250 standard episodes (seeds 200000–200249), 23,000
  steps, seed 0, scored with `--stride 10`: **6.58 cm** against the bar of
  8.26 cm (6.5784 against 8.2593). By latency, 0 / 50 / 100 / 200 ms: 6.21 /
  6.40 / 6.63 / 7.08 cm. Its training loss ended at 0.0054, against 0.0448 for
  the same steps on 2,000 episodes, so it fits its 250 episodes closely.
- **Phase B.** Three configurations on the 2,000 standard episodes, seed 0,
  scored on every validation decision. They differ only in the number of
  steps, which follows JEPA's grid; 60,000 steps is the cap, set on 2026-11-01
  before these runs. Payload error by latency:

| Steps | Selection score | 0 ms | 50 ms | 100 ms | 200 ms | Training (wall) |
| --- | --- | --- | --- | --- | --- | --- |
| 23,000 | 5.09 cm | 4.60 cm | 4.81 cm | 5.13 cm | 5.82 cm | 1.55 h |
| 40,000 | 4.80 cm | 4.38 cm | 4.52 cm | 4.81 cm | 5.48 cm | 2.77 h |
| **60,000** | **4.49 cm** | 4.09 cm | 4.18 cm | 4.49 cm | 5.21 cm | 3.92 h |

60,000 steps scored best and is the selected configuration. The score still
fell from 40,000 to 60,000 steps.

## Formal models

The selected configuration trained with the new seeds 1, 2 and 3, on the 2,000
standard episodes and on the first 500 (seeds 200000–200499), with the same
60,000 steps and no new selection. All six ran in one job on 2026-11-01: 8.9 h
on one RTX 4090, validation scores included. Their validation scores are
diagnostics; nothing was selected among them.

| Model | Selection score | 0 ms | 50 ms | 100 ms | 200 ms | Training (wall) | Final loss | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,000 episodes, seed 1 | 4.49 cm | 4.08 cm | 4.18 cm | 4.49 cm | 5.22 cm | 4.09 h | 0.0180 | `b028f0e2…` |
| 2,000 episodes, seed 2 | 4.45 cm | 4.05 cm | 4.13 cm | 4.44 cm | 5.16 cm | 4.12 h | 0.0164 | `b5dbc196…` |
| 2,000 episodes, seed 3 | 4.44 cm | 4.06 cm | 4.12 cm | 4.43 cm | 5.16 cm | 4.04 h | 0.0195 | `c3180650…` |
| 500 episodes, seed 1 | 5.83 cm | 5.49 cm | 5.64 cm | 5.87 cm | 6.33 cm | 3.87 h | 0.0051 | `5333f72a…` |
| 500 episodes, seed 2 | 5.87 cm | 5.53 cm | 5.68 cm | 5.90 cm | 6.38 cm | 3.98 h | 0.0051 | `6c9cfd2f…` |
| 500 episodes, seed 3 | 5.83 cm | 5.48 cm | 5.64 cm | 5.85 cm | 6.35 cm | 3.86 h | 0.0048 | `fbc7d3ab…` |

- On 500 episodes the 60,000 steps are about 54 passes over the 141,208
  training windows, against about 14 over the 565,288 of 2,000 episodes, and
  the training loss ends at 0.0048–0.0051 against 0.0164–0.0195.
- **Cost.** 1,281,365 parameters. With three runs sharing one RTX 4090, a
  formal run takes 3.9–4.1 h of wall time, about 1.3 GPU-hours, and the six
  take 8.0 GPU-hours. With the data and timing check (8 min), Phase A and
  selection (4.25 h) and the formal job (8.9 h), W-ar used 13.3 GPU-hours of
  training and validation. One planning call (512 × 15) takes 0.075 s on the
  maintainer's RTX 5070 Laptop GPU.

## Results on `dev`

For debugging only: closed-loop `dev` runs never choose a configuration
(PARTICIPANTS.md, Section 6), and these were flown after the selection, once
per model, on the laptop from `f65e007`. Success count out of 5 (one count per
seed) and mean time of the successes:

| Condition | Phase A (250 episodes) | 2,000 episodes (seeds 1, 2, 3) | 500 episodes (seeds 1, 2, 3) |
| --- | --- | --- | --- |
| static, 0 ms | 5 (6.1 s) | 5, 4, 5 (6.7 s) | 4, 5, 4 (5.9 s) |
| static, 50 ms | 4 (6.3 s) | 4, 4, 3 (6.1 s) | 5, 5, 3 (6.2 s) |
| static, 100 ms | 3 (5.3 s) | 2, 3, 3 (5.3 s) | 4, 4, 3 (6.3 s) |
| static, 200 ms | 3 (5.6 s) | 3, 5, 4 (5.9 s) | 4, 3, 3 (6.3 s) |
| static+moving, 0 ms | 5 (6.2 s) | 5, 4, 5 (8.0 s) | 4, 4, 4 (6.5 s) |
| static+moving, 50 ms | 4 (6.7 s) | 3, 4, 1 (6.2 s) | 3, 4, 3 (6.5 s) |
| static+moving, 100 ms | 3 (6.4 s) | 2, 4, 4 (6.2 s) | 3, 3, 1 (5.9 s) |
| static+moving, 200 ms | 4 (6.6 s) | 3, 3, 2 (6.4 s) | 2, 3, 2 (6.9 s) |
| **Total** | **31/40** | **27, 31, 27 of 40** | **29, 31, 23 of 40** |

- On `dev` the nominal physics reference scores 28/40, the example MLP 21/40
  and U 40/40. The six formal models succeed in 168 of 240 episodes; their 72
  failures are 37 aircraft-separation failures, 32 obstacle collisions and 3
  crashes.
- On U's `dev` states, the 2,000-episode seed-1 model scores L1 6.78 cm, L2
  2.42 cm and costly decisions in 26.9% of the 544 questions. This was the
  harness check before `test`.

## Results on `test`

The success counts per condition and layers 1–3 are in the arena
[README](../../README.md#w-ar). ANALYSIS.md, Section 3, also asks for these,
with the nominal physics reference for comparison (HARNESS.md, Sections 4–5,
defines them):

| Model | L1 aircraft | L2, B changes | Relative CEE, payload / others / own | Unresolved, payload / others / own | L3 over all questions | No reliable reference | Top-8 | Time per question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,000 episodes, seed 1 | 9.47 cm | 1.66 cm | 0.56 / 0.87 / 0.40 | 0.1% / 0.5% / 0.0% | 40.9% | 10.9% | 12.48 | 73 ms |
| 2,000 episodes, seed 2 | 9.65 cm | 1.69 cm | 0.56 / 0.87 / 0.40 | 0.1% / 0.5% / 0.0% | 41.4% | 10.9% | 12.31 | 73 ms |
| 2,000 episodes, seed 3 | 9.57 cm | 1.67 cm | 0.57 / 0.88 / 0.40 | 0.1% / 0.5% / 0.0% | 40.9% | 10.9% | 12.66 | 73 ms |
| 500 episodes, seed 1 | 10.29 cm | 1.87 cm | 0.61 / 1.04 / 0.42 | 0.1% / 0.5% / 0.0% | 43.1% | 10.9% | 15.10 | 73 ms |
| 500 episodes, seed 2 | 10.65 cm | 1.91 cm | 0.62 / 1.05 / 0.43 | 0.1% / 0.5% / 0.0% | 44.3% | 10.9% | 15.82 | 74 ms |
| 500 episodes, seed 3 | 10.24 cm | 1.85 cm | 0.61 / 1.03 / 0.42 | 0.1% / 0.5% / 0.0% | 43.0% | 10.9% | 15.14 | 74 ms |
| Nominal physics reference | 10.20 cm | 2.49 cm | 0.57 / 1.31 / 0.40 | 0.1% / 0.5% / 0.0% | 33.8% | 10.9% | 12.37 | 542 ms |

Per condition, pooled over seeds 1–3: successes of 150, and the mean of the
three models' L1 / L2 (cm) / L3, with the nominal physics reference's
successes of 50 and layers:

| Condition | 2,000 episodes | L1 / L2 / L3 | 500 episodes | L1 / L2 / L3 | Physics | L1 / L2 / L3 |
| --- | --- | --- | --- | --- | --- | --- |
| static, 0 ms | 91/150 | 7.66 / 2.94 / 40.3% | 87/150 | 8.66 / 3.33 / 40.4% | 33/50 | 7.74 / 3.51 / 31.4% |
| static, 50 ms | 84/150 | 7.86 / 2.94 / 39.7% | 80/150 | 8.94 / 3.35 / 41.6% | 27/50 | 7.49 / 3.48 / 31.8% |
| static, 100 ms | 73/150 | 8.41 / 3.01 / 40.3% | 72/150 | 9.33 / 3.39 / 41.7% | 27/50 | 8.53 / 3.84 / 32.4% |
| static, 200 ms | 49/150 | 9.45 / 3.13 / 42.6% | 47/150 | 10.07 / 3.44 / 43.7% | 25/50 | 9.29 / 3.98 / 32.6% |
| static+moving, 0 ms | 84/150 | 7.83 / 3.03 / 39.6% | 85/150 | 8.94 / 3.41 / 43.5% | 30/50 | 8.07 / 3.23 / 34.8% |
| static+moving, 50 ms | 80/150 | 8.42 / 3.06 / 40.1% | 63/150 | 9.55 / 3.47 / 44.6% | 26/50 | 7.94 / 3.35 / 34.6% |
| static+moving, 100 ms | 60/150 | 8.97 / 3.26 / 44.8% | 64/150 | 9.82 / 3.59 / 47.2% | 23/50 | 9.35 / 3.76 / 36.3% |
| static+moving, 200 ms | 40/150 | 9.83 / 3.23 / 43.7% | 45/150 | 10.38 / 3.56 / 47.3% | 22/50 | 8.96 / 3.76 / 36.5% |

How the run was made:
- On 2026-11-02, on the maintainer's laptop (Ryzen 9 7845HX, RTX 5070 Laptop
  GPU), from a clean detached checkout of `f65e007`, one model at a time: the
  closed loop, then the harness with 20 workers.
- Before the first `test` episode, that checkout reproduced the three
  2,000-episode models' `dev` records exactly, timings aside, and the harness
  accepted the predictor on `dev` (identical calls give identical outputs).
- The environment was the reference rows': Python 3.10.21, torch
  2.11.0+cu128, MuJoCo 3.3.0, NumPy 2.2.6, SciPy 1.15.3, stable-worldmodel
  0.1.1 and gcc 15.2.0.
- Each model ran once and nothing was rerun. Every harness result records
  `arena-v0.3-8-gf65e007`, the reference rows' question manifest
  (`3c352ce9…`) and its checkpoint's SHA-256.
- Files (not in git): the checkpoints and their training JSON in
  `data/uav_arena/participants/war/`; validation, `dev` and `test` records in
  `data/uav_arena/results/` (`<model>_openloop.json`, `<model>_dev_s0-5.json`,
  `<model>_test.json`); harness scores in
  `data/uav_arena/results/harness/<model>_{dev,test}.{json,npz}`. Models are
  named `formal_war_h4_60k_ep{2000,500}_s{1,2,3}`, `phase_a_war_h4_23k_s0` and
  `phase_b_select_war_h4_{23,40,60}k_s0`.

## Commands

Training, here the 2,000-episode seed-1 model; the 500-episode models use
`--episodes 500`, Phase A `--episodes 250 --steps 23000 --seed 0` and the
selection runs `--seed 0` with their step count:

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.war.train \
  --data data/uav_arena/datasets/explore_v2 --episodes 2000 --steps 60000 --batch 128 --lr 1e-3 \
  --seed 1 --workers 1 --device cuda \
  --backbone uav_arena.participants.jepa.backbone:make_backbone --history 4 \
  --backbone-options '{"history":4,"attn":"full","dim":128,"latent":64,"enc_layers":2,"pred_layers":4,"heads":4}' \
  --out data/uav_arena/participants/war/formal_war_h4_60k_ep2000_s1.pt
```

Validation (Phase A adds `--stride 10`), `test` and its harness scores; `dev`
uses `--suite dev --seeds 0:5`:

```bash
M=formal_war_h4_60k_ep2000_s1
O="{\"checkpoint\": \"data/uav_arena/participants/war/$M.pt\", \"device\": \"cuda\"}"
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop --predictor uav_arena.participants.war.model:make \
  --predictor-options "$O" --data data/uav_arena/datasets/explore_v2_val \
  --out data/uav_arena/results/${M}_openloop.json
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run --predictor uav_arena.participants.war.model:make \
  --predictor-options "$O" --suite test --seeds 0:50 --out data/uav_arena/results/${M}_test.json
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.harness score --predictor uav_arena.participants.war.model:make \
  --predictor-options "$O" --states data/uav_arena/states/test --workers 20 \
  --out data/uav_arena/results/harness/${M}_test.json
```
