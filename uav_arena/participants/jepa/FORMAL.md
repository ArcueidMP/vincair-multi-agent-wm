# JEPA formal models and `test` results

The 18 formal models of the entity-token JEPA rows: `full`, `no_cross` and `flat` at the
configuration each selected in Phase B, G6 (history 4 and 60,000 steps;
[PHASE_B_SELECTION.md](PHASE_B_SELECTION.md)), trained with the new seeds 1, 2 and 3 on the 2,000
standard episodes and on the first 500 (seeds 200000–200499), with no new selection. The recipe is
the selection's: batch 128, learning rate 0.001 with AdamW and a one-cycle schedule, dim 128,
latent 64, 2 encoder and 4 predictor layers, 4 heads, readout weight 20, SIGReg weight 0.01 with
1,024 projections, a 15-step rollout loss and 5,000 samples for the normalization statistics.

## Formal models

The maintainer trained them on 2026-11-03 from a clean detached checkout of `57e9b1a` (PR #3),
whose five training files match the hashes recorded in PHASE_B_SELECTION.md. Each training was
followed by the complete validation (`explore_v2_val`, seeds 299000–299099, every decision).
- **Where.** 15 trained on AIHUB, three at a time on one RTX 4090 (jobs 8234 and 8237), and 3 on
  the maintainer's laptop (RTX 5070 Laptop GPU), one at a time.
- **Copies.** Every model copied from AIHUB matched AIHUB's SHA-256 before its `test` run. The two
  `flat` models trained on the laptop were put on AIHUB the same way, and the jobs skipped them.
- **Validation scores** are diagnostics; nothing was selected among them.

| Model | Trained on | Selection score | 0 ms | 50 ms | 100 ms | 200 ms | Training (wall) | Final loss | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full`, 2,000 episodes, seed 1 | AIHUB, job 8234 | 4.85 cm | 4.45 cm | 4.64 cm | 4.93 cm | 5.36 cm | 3.19 h | 1.7917 | `5fca7f71…` |
| `full`, 2,000 episodes, seed 2 | AIHUB, job 8234 | 5.04 cm | 4.68 cm | 4.85 cm | 5.12 cm | 5.49 cm | 3.15 h | 1.7368 | `7338ca34…` |
| `full`, 2,000 episodes, seed 3 | laptop | 4.93 cm | 4.56 cm | 4.73 cm | 5.02 cm | 5.41 cm | 2.28 h | 1.8726 | `f0b22a2d…` |
| `full`, 500 episodes, seed 1 | AIHUB, job 8234 | 6.93 cm | 6.61 cm | 6.80 cm | 6.99 cm | 7.33 cm | 3.16 h | 0.7999 | `07a495d5…` |
| `full`, 500 episodes, seed 2 | AIHUB, job 8237 | 6.68 cm | 6.37 cm | 6.55 cm | 6.73 cm | 7.07 cm | 3.12 h | 0.7266 | `7998e633…` |
| `full`, 500 episodes, seed 3 | AIHUB, job 8237 | 6.68 cm | 6.37 cm | 6.56 cm | 6.74 cm | 7.07 cm | 2.92 h | 0.7733 | `64151208…` |
| `no_cross`, 2,000 episodes, seed 1 | AIHUB, job 8234 | 4.66 cm | 4.28 cm | 4.45 cm | 4.74 cm | 5.17 cm | 3.21 h | 1.8721 | `16afc477…` |
| `no_cross`, 2,000 episodes, seed 2 | AIHUB, job 8234 | 4.74 cm | 4.35 cm | 4.54 cm | 4.81 cm | 5.25 cm | 3.15 h | 1.7491 | `77e7da02…` |
| `no_cross`, 2,000 episodes, seed 3 | AIHUB, job 8237 | 4.54 cm | 4.17 cm | 4.32 cm | 4.60 cm | 5.08 cm | 3.10 h | 1.8561 | `3dffa9fc…` |
| `no_cross`, 500 episodes, seed 1 | AIHUB, job 8234 | 6.41 cm | 6.10 cm | 6.28 cm | 6.47 cm | 6.78 cm | 3.08 h | 0.7702 | `eb2d18df…` |
| `no_cross`, 500 episodes, seed 2 | AIHUB, job 8237 | 6.44 cm | 6.14 cm | 6.32 cm | 6.49 cm | 6.79 cm | 3.07 h | 0.7760 | `97c9500b…` |
| `no_cross`, 500 episodes, seed 3 | AIHUB, job 8237 | 6.36 cm | 6.05 cm | 6.23 cm | 6.42 cm | 6.74 cm | 2.89 h | 0.7610 | `4e9cd1e4…` |
| `flat`, 2,000 episodes, seed 1 | AIHUB, job 8234 | 6.50 cm | 6.19 cm | 6.38 cm | 6.56 cm | 6.88 cm | 2.50 h | 2.3254 | `a4d98584…` |
| `flat`, 2,000 episodes, seed 2 | AIHUB, job 8234 | 6.54 cm | 6.24 cm | 6.42 cm | 6.60 cm | 6.89 cm | 2.46 h | 2.2785 | `f511422c…` |
| `flat`, 2,000 episodes, seed 3 | AIHUB, job 8237 | 6.47 cm | 6.16 cm | 6.34 cm | 6.53 cm | 6.85 cm | 2.54 h | 2.3579 | `2203afbc…` |
| `flat`, 500 episodes, seed 1 | AIHUB, job 8234 | 7.39 cm | 7.09 cm | 7.27 cm | 7.44 cm | 7.77 cm | 2.35 h | 0.7111 | `da4a2aac…` |
| `flat`, 500 episodes, seed 2 | laptop | 7.46 cm | 7.15 cm | 7.33 cm | 7.50 cm | 7.84 cm | 1.19 h | 0.6896 | `986c558a…` |
| `flat`, 500 episodes, seed 3 | laptop | 7.48 cm | 7.17 cm | 7.36 cm | 7.52 cm | 7.86 cm | 1.21 h | 0.7125 | `691c731e…` |

- **The final loss** is the whole objective at step 60,000: prediction, SIGReg and readout.
- **Passes over the data.** On 500 episodes the 60,000 steps are about 54 passes over the 141,208
  training windows, against about 14 over the 565,288 of 2,000 episodes. The final loss is
  accordingly lower on 500 episodes:

  | Variant | 500 episodes | 2,000 episodes |
  | --- | --- | --- |
  | `full`, `no_cross` | 0.73–0.80 | 1.74–1.87 |
  | `flat` | 0.69–0.71 | 2.28–2.36 |
- **Data.** Every model records the same `explore_v2` manifest hash (`9f4f89c7…`) on both
  machines, and `collect --verify` reports every episode of that dataset as a match with the
  reference manifests. The hash differs from the one in PHASE_B_SELECTION.md (`cb196ded…`).
- **Parameters.** 1,289,225 for `full` and `no_cross`, and 1,372,047 for `flat`.
- **Training cost on AIHUB** (15 models, 16.1 GPU-hours with validation):
  - With three runs sharing one RTX 4090, a `full` or `no_cross` run takes 2.89–3.21 h of
    training and a `flat` run 2.35–2.54 h.
  - Job 8234 (9 models) took 9.57 h and job 8237 (6 models) 6.49 h, validation included.
- **Training cost on the laptop** (3 models, 5.0 GPU-hours): `full` took 2.28 h alone and each
  `flat` 1.19–1.21 h, plus about 7 min of validation each.
- **Selection cost.** The 18 selection runs had taken 37.9 GPU-hours of training
  (PHASE_B_SELECTION.md).
- **Planning time.** One planning call (512 × 15) takes 52–59 ms for `full` and `no_cross` and
  19–24 ms for `flat` on the RTX 5070 Laptop GPU.

## Results on `test`

The success counts per condition and layers 1–3 are in the arena
[README](../../README.md#jepa). ANALYSIS.md, Section 3, also asks for these, with the nominal
physics reference for comparison (HARNESS.md, Sections 4–5, defines them):

| Model | L1 aircraft | L2, B changes | Relative CEE, payload / others / own | Unresolved, payload / others / own | L3 over all questions | No reliable reference | Top-8 | Time per question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full`, 2,000 episodes, seed 1 | 11.39 cm | 1.84 cm | 0.62 / 1.09 / 0.45 | 0.1% / 0.5% / 0.0% | 46.9% | 10.9% | 19.94 | 48 ms |
| `full`, 2,000 episodes, seed 2 | 11.77 cm | 1.88 cm | 0.62 / 1.12 / 0.46 | 0.1% / 0.5% / 0.0% | 47.2% | 10.9% | 20.85 | 54 ms |
| `full`, 2,000 episodes, seed 3 | 11.65 cm | 1.90 cm | 0.63 / 1.11 / 0.46 | 0.1% / 0.5% / 0.0% | 46.2% | 10.9% | 20.13 | 52 ms |
| `full`, 500 episodes, seed 1 | 12.57 cm | 2.36 cm | 0.71 / 1.36 / 0.51 | 0.1% / 0.5% / 0.0% | 49.2% | 10.9% | 22.56 | 48 ms |
| `full`, 500 episodes, seed 2 | 12.33 cm | 2.22 cm | 0.68 / 1.29 / 0.51 | 0.1% / 0.5% / 0.0% | 47.0% | 10.9% | 21.42 | 54 ms |
| `full`, 500 episodes, seed 3 | 12.25 cm | 2.28 cm | 0.70 / 1.32 / 0.51 | 0.1% / 0.5% / 0.0% | 47.0% | 10.9% | 20.71 | 54 ms |
| `no_cross`, 2,000 episodes, seed 1 | 11.18 cm | 1.82 cm | 0.61 / 1.07 / 0.43 | 0.1% / 0.5% / 0.0% | 45.0% | 10.9% | 18.69 | 48 ms |
| `no_cross`, 2,000 episodes, seed 2 | 11.35 cm | 1.85 cm | 0.62 / 1.07 / 0.44 | 0.1% / 0.5% / 0.0% | 46.9% | 10.9% | 19.47 | 54 ms |
| `no_cross`, 2,000 episodes, seed 3 | 11.04 cm | 1.83 cm | 0.61 / 1.07 / 0.43 | 0.1% / 0.5% / 0.0% | 44.4% | 10.9% | 19.19 | 54 ms |
| `no_cross`, 500 episodes, seed 1 | 11.70 cm | 2.08 cm | 0.68 / 1.14 / 0.49 | 0.1% / 0.5% / 0.0% | 46.1% | 10.9% | 21.24 | 48 ms |
| `no_cross`, 500 episodes, seed 2 | 11.58 cm | 2.11 cm | 0.68 / 1.14 / 0.49 | 0.1% / 0.5% / 0.0% | 46.7% | 10.9% | 20.47 | 54 ms |
| `no_cross`, 500 episodes, seed 3 | 11.38 cm | 2.11 cm | 0.68 / 1.14 / 0.49 | 0.1% / 0.5% / 0.0% | 46.5% | 10.9% | 20.83 | 54 ms |
| `flat`, 2,000 episodes, seed 1 | 15.18 cm | 2.28 cm | 0.71 / 1.39 / 0.55 | 0.1% / 0.5% / 0.0% | 56.2% | 10.9% | 28.21 | 16 ms |
| `flat`, 2,000 episodes, seed 2 | 15.26 cm | 2.30 cm | 0.72 / 1.41 / 0.55 | 0.1% / 0.5% / 0.0% | 56.6% | 10.9% | 26.29 | 16 ms |
| `flat`, 2,000 episodes, seed 3 | 15.19 cm | 2.34 cm | 0.73 / 1.43 / 0.55 | 0.1% / 0.5% / 0.0% | 55.8% | 10.9% | 27.12 | 16 ms |
| `flat`, 500 episodes, seed 1 | 13.95 cm | 2.49 cm | 0.74 / 1.45 / 0.53 | 0.1% / 0.5% / 0.0% | 50.6% | 10.9% | 26.25 | 16 ms |
| `flat`, 500 episodes, seed 2 | 13.95 cm | 2.47 cm | 0.74 / 1.44 / 0.53 | 0.1% / 0.5% / 0.0% | 50.2% | 10.9% | 24.96 | 14 ms |
| `flat`, 500 episodes, seed 3 | 14.54 cm | 2.53 cm | 0.76 / 1.47 / 0.53 | 0.1% / 0.5% / 0.0% | 52.1% | 10.9% | 25.83 | 17 ms |
| Nominal physics reference | 10.20 cm | 2.49 cm | 0.57 / 1.31 / 0.40 | 0.1% / 0.5% / 0.0% | 33.8% | 10.9% | 12.37 | 542 ms |

Per condition, pooled over seeds 1–3: successes of 150, and the mean of the three models' L1 / L2
(cm) / L3. The nominal physics reference's per-condition numbers are in
[the W-ar README](../war/README.md#results-on-test).

| Condition | `full`, 2,000 episodes | L1 / L2 / L3 | `full`, 500 episodes | L1 / L2 / L3 |
| --- | --- | --- | --- | --- |
| static, 0 ms | 79/150 | 8.14 / 3.46 / 46.6% | 58/150 | 10.09 / 3.98 / 47.1% |
| static, 50 ms | 85/150 | 8.45 / 3.47 / 47.0% | 64/150 | 10.38 / 4.00 / 46.6% |
| static, 100 ms | 83/150 | 8.90 / 3.52 / 47.0% | 56/150 | 10.66 / 4.03 / 47.0% |
| static, 200 ms | 65/150 | 9.60 / 3.57 / 46.6% | 47/150 | 11.21 / 4.07 / 46.9% |
| static+moving, 0 ms | 82/150 | 8.44 / 3.53 / 48.0% | 58/150 | 10.45 / 4.05 / 48.4% |
| static+moving, 50 ms | 75/150 | 9.17 / 3.57 / 47.3% | 51/150 | 11.04 / 4.08 / 47.5% |
| static+moving, 100 ms | 80/150 | 9.49 / 3.72 / 50.7% | 48/150 | 11.34 / 4.19 / 50.7% |
| static+moving, 200 ms | 56/150 | 9.91 / 3.65 / 47.4% | 37/150 | 11.61 / 4.13 / 48.7% |

| Condition | `no_cross`, 2,000 episodes | L1 / L2 / L3 | `no_cross`, 500 episodes | L1 / L2 / L3 |
| --- | --- | --- | --- | --- |
| static, 0 ms | 87/150 | 7.89 / 3.37 / 45.6% | 52/150 | 9.99 / 3.74 / 46.2% |
| static, 50 ms | 84/150 | 8.23 / 3.39 / 45.8% | 54/150 | 10.24 / 3.76 / 45.3% |
| static, 100 ms | 78/150 | 8.69 / 3.44 / 46.0% | 49/150 | 10.50 / 3.78 / 46.0% |
| static, 200 ms | 71/150 | 9.45 / 3.50 / 46.4% | 42/150 | 11.00 / 3.81 / 46.2% |
| static+moving, 0 ms | 77/150 | 8.16 / 3.43 / 45.8% | 49/150 | 10.26 / 3.80 / 47.4% |
| static+moving, 50 ms | 75/150 | 8.96 / 3.49 / 45.6% | 48/150 | 10.97 / 3.83 / 46.2% |
| static+moving, 100 ms | 76/150 | 9.27 / 3.64 / 47.5% | 35/150 | 11.12 / 3.95 / 49.1% |
| static+moving, 200 ms | 45/150 | 9.75 / 3.57 / 46.3% | 31/150 | 11.32 / 3.88 / 47.0% |

| Condition | `flat`, 2,000 episodes | L1 / L2 / L3 | `flat`, 500 episodes | L1 / L2 / L3 |
| --- | --- | --- | --- | --- |
| static, 0 ms | 27/150 | 10.41 / 4.15 / 56.4% | 21/150 | 11.94 / 4.35 / 50.1% |
| static, 50 ms | 22/150 | 10.69 / 4.17 / 55.5% | 23/150 | 12.19 / 4.37 / 49.9% |
| static, 100 ms | 27/150 | 10.98 / 4.20 / 55.6% | 17/150 | 12.45 / 4.38 / 50.5% |
| static, 200 ms | 22/150 | 11.54 / 4.24 / 55.7% | 29/150 | 13.01 / 4.41 / 50.7% |
| static+moving, 0 ms | 31/150 | 10.82 / 4.23 / 57.2% | 12/150 | 12.19 / 4.40 / 52.4% |
| static+moving, 50 ms | 24/150 | 11.42 / 4.27 / 56.1% | 17/150 | 12.77 / 4.41 / 51.3% |
| static+moving, 100 ms | 26/150 | 11.73 / 4.38 / 57.7% | 14/150 | 13.02 / 4.52 / 54.7% |
| static+moving, 200 ms | 15/150 | 12.04 / 4.33 / 57.3% | 13/150 | 13.20 / 4.48 / 51.6% |

How the run was made:
- From 2026-11-03 16:15 to 2026-11-04 07:23, on the maintainer's laptop (Ryzen 9 7845HX, RTX 5070
  Laptop GPU), from the clean detached checkout of `57e9b1a` that trained the laptop's models.
- One model at a time, as each arrived: `dev` (seeds 0–4), the closed loop on `test`, then the
  harness with 20 workers. Tests waited while the laptop trained or validated.
- Before the first `test` episode, that checkout reproduced the 2,000-episode `full` seed-3 model's
  `dev` records exactly, timings aside, and the harness accepted the predictor on `dev` (identical
  calls give identical outputs).
- The environment was the reference rows': Python 3.10.21, torch 2.11.0+cu128, MuJoCo 3.3.0, NumPy
  2.2.6, SciPy 1.15.3, stable-worldmodel 0.1.1 and gcc 15.2.0.
- Each model ran once and nothing was rerun. Every harness result records `arena-v0.3-8-g57e9b1a`,
  the reference rows' question manifest (`3c352ce9…`) and its checkpoint's SHA-256, the one in the
  table above.
- Files (not in git):
  - the checkpoints and their training JSON in `data/uav_arena/participants/jepa/`;
  - validation, `dev` and `test` records in `data/uav_arena/results/` (`<model>_openloop.json`,
    `<model>_dev_s0-5.json`, `<model>_test.json`);
  - harness scores in `data/uav_arena/results/harness/<model>_test.{json,npz}`, plus `_dev` for the
    model of the check;
  - training and validation logs in `data/uav_arena/logs/`.

  Models are named `formal_{full,no_cross,flat}_h4_60k_ep{2000,500}_s{1,2,3}`.

## Commands

Training and validation, here `full` on 2,000 episodes with seed 1; the other models change
`--attn`, `--episodes` and `--seed`:

```bash
M=formal_full_h4_60k_ep2000_s1
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.train \
  --data data/uav_arena/datasets/explore_v2 --episodes 2000 --history 4 --steps 60000 --attn full \
  --batch 128 --lr 0.001 --dim 128 --latent 64 --enc-layers 2 --pred-layers 4 --heads 4 --head-weight 20 \
  --sigreg-weight 0.01 --sigreg-projections 1024 --rollout-steps 15 --stats-samples 5000 --seed 1 \
  --workers 4 --device cuda --out data/uav_arena/participants/jepa/$M.pt
O="{\"checkpoint\":\"data/uav_arena/participants/jepa/$M.pt\",\"device\":\"cuda\"}"
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop --predictor uav_arena.participants.jepa.model:make \
  --predictor-options "$O" --data data/uav_arena/datasets/explore_v2_val --seeds 299000:299100 \
  --out data/uav_arena/results/${M}_openloop.json
```

`test` and its harness scores; `dev` uses `--suite dev --seeds 0:5`:

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run --predictor uav_arena.participants.jepa.model:make \
  --predictor-options "$O" --suite test --seeds 0:50 --workers 20 --out data/uav_arena/results/${M}_test.json
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.harness score --predictor uav_arena.participants.jepa.model:make \
  --predictor-options "$O" --states data/uav_arena/states/test --workers 20 \
  --out data/uav_arena/results/harness/${M}_test.json
```
