# PETS Phase B progress report

Status: selection, formal validation, the six formal `dev` runs and the
central `test` run are complete on the frozen `arena-v0.3` protocol (results
below). G7 (depth 3, width 291, 60,000 steps/member, learning rate 1e-3) was
selected after all eight allowed configurations. Every
selection run uses all 2,000 `explore_v2` episodes, seed 0, mean propagation
(`sample_gaussian=false`) and every decision of all 100 `explore_v2_val`
episodes. Neither `dev` nor `test` is used for selection.

The total parameter limit is 1,372,047, counting all five ensemble members and
their bounded log-variance parameters. The search was staged and used all
eight allowed configurations. G1 and G2 compared the largest compliant depth-4
and depth-5 models. Depth 4 then went on to longer schedules (G3 at 40,000
steps, G5 at 60,000) and a lower learning rate (G4). G6, G7 and G8 tried the
largest compliant depth-3 model at 40,000 and 60,000 steps and, at 60,000, the
lower learning rate. Every attempted configuration remains in the table.

## Selection results

| ID | Depth | Width | Steps/member | Learning rate | Parameters | Train time | Selection score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 | 4 | 248 | 20,000 | 1e-3 | 1,369,980 | 1,308.17 s (0.363 h) | **6.64 cm** |
| G2 | 5 | 220 | 20,000 | 1e-3 | 1,366,120 | 1,325.04 s (0.368 h) | 8.91 cm |
| G3 | 4 | 248 | 40,000 | 1e-3 | 1,369,980 | 2,600.10 s (0.722 h) | **6.43 cm** |
| G4 | 4 | 248 | 40,000 | 5e-4 | 1,369,980 | 2,572.51 s (0.715 h) | 6.88 cm |
| G5 | 4 | 248 | 60,000 | 1e-3 | 1,369,980 | 3,760.60 s (1.045 h) | **6.32 cm** |
| G6 | 3 | 291 | 40,000 | 1e-3 | 1,370,175 | 2,573.30 s (0.715 h) | 6.34 cm |
| G7 | 3 | 291 | 60,000 | 1e-3 | 1,370,175 | 3,764.89 s (1.046 h) | **6.26 cm** |
| G8 | 3 | 291 | 60,000 | 5e-4 | 1,370,175 | 3,852.56 s (1.070 h) | 6.58 cm |

The example trained on all 2,000 episodes scores 6.24 cm. G2 is 2.27 cm worse
than G1, so the depth-5 architecture did not advance. G4 is 0.45 cm worse than
G3, ruling out its lower learning rate for depth 4. G6's shallow/wide
architecture improves on G3 by 0.09 cm at the same 40k schedule. Extending it
to 60k in G7 improves a further 0.08 cm to 6.26 cm, beating G5 by 0.06 cm and
finishing 0.02 cm behind the example. G8 shows that lowering the learning rate
for the selected architecture worsens the score by 0.32 cm. G7 is therefore
selected for formal seed runs. The eight selection runs total 6.044 GPU-hours.

The complete validation results (7,429 decisions per latency) are:

Payload error:

| Latency | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 ms | 5.12 cm | 7.47 cm | 4.65 cm | 5.30 cm | 4.42 cm | 4.54 cm | 4.32 cm | 4.80 cm |
| 50 ms | 5.71 cm | 7.84 cm | 5.37 cm | 5.94 cm | 5.27 cm | 5.27 cm | 5.20 cm | 5.51 cm |
| 100 ms | 6.77 cm | 9.13 cm | 6.66 cm | 7.03 cm | 6.68 cm | 6.56 cm | 6.62 cm | 6.74 cm |
| 200 ms | 8.97 cm | 11.22 cm | 9.04 cm | 9.23 cm | 8.93 cm | 8.99 cm | 8.92 cm | 9.28 cm |
| **Selection score** | **6.64 cm** | **8.91 cm** | **6.43 cm** | **6.88 cm** | **6.32 cm** | **6.34 cm** | **6.26 cm** | **6.58 cm** |

Aircraft error:

| Latency | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 ms | 4.71 cm | 8.14 cm | 4.35 cm | 4.91 cm | 4.19 cm | 4.28 cm | 4.11 cm | 4.50 cm |
| 50 ms | 5.26 cm | 8.38 cm | 4.96 cm | 5.50 cm | 4.91 cm | 4.92 cm | 4.86 cm | 5.13 cm |
| 100 ms | 6.24 cm | 9.66 cm | 6.11 cm | 6.50 cm | 6.16 cm | 6.08 cm | 6.14 cm | 6.24 cm |
| 200 ms | 8.21 cm | 11.53 cm | 8.19 cm | 8.47 cm | 8.16 cm | 8.21 cm | 8.20 cm | 8.42 cm |

All eight configurations trained on an RTX 5060 Laptop GPU. G7 and G8's
ignored local checkpoints are each 5.3 MB. G1 has SHA-256
`1afe1cf75d214546419089f700aadd4c7512cca9c5e1ef2b9f061afcf9aa3a03`;
G2 has SHA-256
`0e4ed21b20e9b40cdf554051495d5528e345caca1d5b1030601e7f36443778e9`;
G3 has SHA-256
`e9cafbae6f6433117efcd65e996ce9b5599ca52b466efce5121d40ec540baf72`;
G4 has SHA-256
`de1f725a614c99df5326bba8de114593ee8ee4dc45be013104efaada2847dc5f`;
G5 has SHA-256
`c41a0316399325c387ca23b2ff08b1370d8381dc51af6151a570c2d2ddf0a6bc`;
G6 has SHA-256
`5d52c393843393cdefe6810b1ff450a5851771b205d605c33187711b94d5efcb`;
G7 has SHA-256
`1842ccd30e4c1a5b8f753ccd24494cd92a4174d2512a4cc79d4625a0494a15b7`;
G8 has SHA-256
`2311ca934ff3f79e7a2fe569279c53c270d64fb2bced3e24bc87421fcd7a5a7d`.
The checkpoints and raw result JSON are not committed.

## Formal seed results

The selected G7 configuration was retrained from scratch with seeds 1--3 at
both required data budgets. Each checkpoint was scored at every decision of
all 100 `explore_v2_val` episodes, with 7,429 decisions per latency. Values
below are payload error; `±` is the sample standard deviation over the three
formal seeds.

| Training episodes | Seed 1 | Seed 2 | Seed 3 | Mean ± std |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 6.23 cm | 6.27 cm | 6.21 cm | **6.24 ± 0.03 cm** |
| 500 | 6.40 cm | 6.43 cm | 6.41 cm | **6.41 ± 0.01 cm** |

Mean payload error by latency:

| Training episodes | 0 ms | 50 ms | 100 ms | 200 ms |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 4.28 ± 0.03 cm | 5.17 ± 0.03 cm | 6.60 ± 0.03 cm | 8.90 ± 0.07 cm |
| 500 | 4.62 ± 0.04 cm | 5.39 ± 0.03 cm | 6.70 ± 0.01 cm | 8.95 ± 0.03 cm |

Mean aircraft error by latency:

| Training episodes | 0 ms | 50 ms | 100 ms | 200 ms |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 4.08 ± 0.02 cm | 4.85 ± 0.02 cm | 6.13 ± 0.01 cm | 8.19 ± 0.02 cm |
| 500 | 4.37 ± 0.02 cm | 5.04 ± 0.02 cm | 6.21 ± 0.01 cm | 8.22 ± 0.02 cm |

The 2,000-episode mean matches the 6.24 cm example reference. Restricting
training to the first 500 episodes worsens the mean by 0.17 cm, while both
budgets have low across-seed variation. Training took 3.149 GPU-hours for the
three 2,000-episode runs and 3.173 GPU-hours for the three 500-episode runs.
Neither `dev` nor `test` was used.

Formal checkpoint SHA-256 values are:

- 2,000 episodes, seeds 1--3: `ea5f8adabdb38995cabba09ab37061986f38bdb5777746cb83445e9fe8a272bc`,
  `e23fe83b845a8e1e5f3eed0ae1d91d740472280e686d754c0f698bf80c89b0a3`,
  `e7e55332ecb44f2e4aea480777372d6722785cb5ffbe56fe68cfd15ec0bcfccc`.
- 500 episodes, seeds 1--3: `1bd299045bdf8a0b31ae265113ad702aebd0f9081588c25ee16796651350ea84`,
  `b022450c782d54058d27bc6558f074155b6d43c1db1e89b6bd041627083b684a`,
  `0412db9e2e690b9c48def1cff946944d53f234c739d8c6e4f66809bb8e0f4573`.
These checkpoints and the six raw validation JSON files remain ignored and
local.

## Formal dev results and planning time

After selection was frozen, each formal checkpoint ran once on all 40 `dev`
episodes. `dev` was not used to choose or change the configuration. Each cell
is successes out of five episodes, followed by mean time among successes.

| Condition | 2,000 s1 | 2,000 s2 | 2,000 s3 | 500 s1 | 500 s2 | 500 s3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static / 0 ms | 3/5 (8.1 s) | 5/5 (6.3 s) | 4/5 (5.2 s) | 1/5 (4.5 s) | 1/5 (4.8 s) | 4/5 (7.4 s) |
| static / 50 ms | 0/5 | 4/5 (6.7 s) | 3/5 (6.2 s) | 2/5 (11.0 s) | 2/5 (5.7 s) | 3/5 (5.4 s) |
| static / 100 ms | 2/5 (5.5 s) | 1/5 (8.6 s) | 1/5 (5.4 s) | 1/5 (5.4 s) | 2/5 (8.4 s) | 2/5 (6.0 s) |
| static / 200 ms | 1/5 (5.2 s) | 2/5 (7.8 s) | 1/5 (5.9 s) | 0/5 | 2/5 (5.9 s) | 1/5 (5.8 s) |
| static+moving / 0 ms | 2/5 (5.1 s) | 2/5 (5.6 s) | 4/5 (6.2 s) | 2/5 (7.7 s) | 0/5 | 1/5 (4.4 s) |
| static+moving / 50 ms | 1/5 (6.3 s) | 0/5 | 2/5 (5.1 s) | 2/5 (5.0 s) | 2/5 (10.5 s) | 2/5 (5.3 s) |
| static+moving / 100 ms | 1/5 (4.9 s) | 2/5 (7.4 s) | 2/5 (5.8 s) | 2/5 (6.9 s) | 1/5 (5.4 s) | 2/5 (6.2 s) |
| static+moving / 200 ms | 1/5 (5.2 s) | 1/5 (6.6 s) | 2/5 (7.7 s) | 0/5 | 1/5 (5.5 s) | 1/5 (5.8 s) |
| **Total** | **11/40** | **17/40** | **19/40** | **10/40** | **11/40** | **16/40** |

The mean time of one frozen-planner call (512 candidates × 15 decisions),
weighted by the number of decisions in each episode, was 20.86 ± 0.49 ms over
the three 2,000-episode seeds and 20.15 ± 0.56 ms over the three 500-episode
seeds (mean ± sample standard deviation between training seeds). Pooling all
13,504 formal `dev` planning calls gives **20.51 ms per call** on the RTX 5060
Laptop GPU.

The six formal checkpoints were not run on `test` locally. They went to the
maintainer with the exact `harness score` and closed-loop commands; the central
`test` results follow.

## Central `test` results

The maintainer ran each formal checkpoint once on `test` on 2026-11-02, one at
a time on an RTX 5070 Laptop GPU, from a clean detached checkout of `87a502b`
with `sample_gaussian=false`: the closed loop on all 50 layouts and 8
conditions, then the harness on its 8,660 questions. Every output records the
source `arena-v0.3-17-g87a502b` and the checkpoint's SHA-256. Before the first
`test` episode, on the same machine, the six checkpoints flew `dev` (12, 20 and
20 of 40 at 2,000 episodes; 8, 13 and 17 at 500), the 2,000-episode seed-1
checkpoint flew it again with identical records, timings aside, and the harness
accepted the predictor on `dev`. The `dev` table above was flown on Ubuntu
22.04, where the simulator does not reproduce this machine's episodes bit for
bit; 21 of its 48 cells agree with the maintainer's.

Each cell is successes out of 50 episodes, followed by mean time among
successes. Intervals are 95% layout bootstraps with the arena README's
resamples.

| Condition | 2,000 s1 | 2,000 s2 | 2,000 s3 | 500 s1 | 500 s2 | 500 s3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static / 0 ms | 10/50 (6.2 s) | 15/50 (7.8 s) | 22/50 (7.8 s) | 6/50 (6.5 s) | 14/50 (6.8 s) | 14/50 (6.9 s) |
| static / 50 ms | 13/50 (7.0 s) | 20/50 (7.8 s) | 19/50 (6.5 s) | 12/50 (7.4 s) | 13/50 (7.7 s) | 12/50 (6.5 s) |
| static / 100 ms | 12/50 (9.1 s) | 11/50 (8.1 s) | 14/50 (6.7 s) | 9/50 (5.6 s) | 7/50 (8.9 s) | 10/50 (7.3 s) |
| static / 200 ms | 5/50 (5.6 s) | 7/50 (8.6 s) | 8/50 (8.5 s) | 6/50 (7.4 s) | 8/50 (6.7 s) | 6/50 (6.1 s) |
| static+moving / 0 ms | 9/50 (7.0 s) | 15/50 (8.0 s) | 17/50 (8.3 s) | 6/50 (6.3 s) | 11/50 (7.8 s) | 10/50 (7.8 s) |
| static+moving / 50 ms | 13/50 (8.7 s) | 15/50 (9.1 s) | 14/50 (6.7 s) | 9/50 (6.5 s) | 14/50 (8.9 s) | 10/50 (6.1 s) |
| static+moving / 100 ms | 7/50 (7.6 s) | 10/50 (8.4 s) | 8/50 (7.8 s) | 2/50 (5.7 s) | 5/50 (6.1 s) | 12/50 (9.6 s) |
| static+moving / 200 ms | 4/50 (6.2 s) | 6/50 (8.4 s) | 5/50 (8.2 s) | 2/50 (5.6 s) | 2/50 (5.0 s) | 6/50 (7.3 s) |
| **Total** | **73/400** | **99/400** | **107/400** | **52/400** | **74/400** | **80/400** |
| Success rate (95% interval) | 18.25% (11.50–25.75%) | 24.75% (17.25–33.00%) | 26.75% (19.00–35.25%) | 13.00% (8.00–19.00%) | 18.50% (11.75–25.75%) | 20.00% (12.75–28.00%) |

Harness layers on the 8,660 questions, with 95% intervals for L1, L2 and L3:

| Model | L1 payload | L2 cross-agent | L3 costly decisions | Regret, mean (median) | Top-1 | Spearman | Pairwise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 episodes, seed 1 | 12.93 cm (11.82–14.03) | 3.78 cm (3.51–4.06) | 46.7% (39.7–53.0%) | +12.32 (+2.22) | 19.06 | 0.43 | 0.66 |
| 2,000 episodes, seed 2 | 12.22 cm (11.22–13.22) | 3.67 cm (3.41–3.93) | 46.6% (39.7–52.8%) | +12.25 (+2.17) | 19.44 | 0.44 | 0.66 |
| 2,000 episodes, seed 3 | 12.44 cm (11.40–13.48) | 3.70 cm (3.43–3.97) | 46.6% (39.6–52.8%) | +12.07 (+2.24) | 19.39 | 0.44 | 0.66 |
| 500 episodes, seed 1 | 13.11 cm (11.97–14.25) | 3.88 cm (3.60–4.16) | 47.1% (40.2–53.2%) | +12.69 (+2.33) | 19.87 | 0.42 | 0.65 |
| 500 episodes, seed 2 | 13.64 cm (12.42–14.84) | 3.99 cm (3.69–4.27) | 48.4% (41.3–54.6%) | +13.19 (+2.61) | 20.18 | 0.41 | 0.65 |
| 500 episodes, seed 3 | 12.97 cm (11.88–14.05) | 3.89 cm (3.61–4.17) | 47.9% (40.8–54.1%) | +12.76 (+2.54) | 19.56 | 0.42 | 0.65 |

Mean over the three seeds, per condition:

| Condition | 2,000: L1 / L2 / L3 | 500: L1 / L2 / L3 |
| --- | ---: | ---: |
| static / 0 ms | 10.54 cm / 3.49 cm / 43.5% | 11.33 cm / 3.69 cm / 45.4% |
| static / 50 ms | 11.38 cm / 3.55 cm / 44.4% | 12.20 cm / 3.75 cm / 45.8% |
| static / 100 ms | 12.67 cm / 3.68 cm / 47.0% | 13.37 cm / 3.87 cm / 47.3% |
| static / 200 ms | 15.07 cm / 3.94 cm / 48.7% | 15.63 cm / 4.10 cm / 49.4% |
| static+moving / 0 ms | 10.65 cm / 3.57 cm / 44.5% | 11.49 cm / 3.79 cm / 46.6% |
| static+moving / 50 ms | 11.69 cm / 3.65 cm / 46.0% | 12.45 cm / 3.86 cm / 47.4% |
| static+moving / 100 ms | 12.99 cm / 3.89 cm / 49.6% | 13.65 cm / 4.10 cm / 49.9% |
| static+moving / 200 ms | 15.30 cm / 4.01 cm / 49.7% | 15.82 cm / 4.19 cm / 50.7% |

Pooled over the three seeds, the 500-episode checkpoints are worse than the
2,000-episode checkpoints in the closed loop and on every layer; the difference
pairs the checkpoints on each resample:

| | Success rate | L1 | L2 | L3 |
| --- | ---: | ---: | ---: | ---: |
| 2,000 episodes | 23.25% (16.42–30.92%) | 12.53 cm (11.49–13.58) | 3.72 cm (3.45–3.99) | 46.65% (39.66–52.82%) |
| 500 episodes | 17.17% (11.42–23.67%) | 13.24 cm (12.10–14.37) | 3.92 cm (3.63–4.20) | 47.80% (40.77–54.01%) |
| 500 − 2,000 | −6.08 points (−10.00 to −2.25) | +0.71 cm (+0.59 to +0.82) | +0.20 cm (+0.18 to +0.22) | +1.16 points (+0.71 to +1.57) |

A planning call took 19.2–21.9 ms. A closed loop took 10.1–10.8 min and a
harness run 50.2–52.9 min, since a stochastic predictor makes 48 calls per
question; the six checkpoints took 6.2 h in all. The arena README compares the
row with the reference rows and W-ar.

## Commands

G1 training:

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.pets.train \
  --data data/uav_arena/datasets/explore_v2 --episodes 2000 \
  --steps 20000 --batch 256 --lr 0.001 --members 5 \
  --width 248 --depth 4 --seed 0 --device cuda \
  --out data/uav_arena/participants/pets/phase_b_sel_g1_d4w248_s20k_seed0.pt
```

G1 complete validation (there is deliberately no `--stride`):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  data/uav_arena/.venv/bin/python -m uav_arena.openloop \
  --predictor uav_arena.participants.pets.model:make \
  --predictor-options \
  '{"checkpoint":"data/uav_arena/participants/pets/phase_b_sel_g1_d4w248_s20k_seed0.pt","device":"cuda","sample_gaussian":false}' \
  --data data/uav_arena/datasets/explore_v2_val \
  --out data/uav_arena/results/pets_phase_b_sel_g1_d4w248_s20k_seed0_openloop_val.json
```

G2 changes only depth and width in the training command: `--width 220
--depth 5`, with output
`data/uav_arena/participants/pets/phase_b_sel_g2_d5w220_s20k_seed0.pt`.
Its complete validation uses the same command as G1 with that checkpoint and
writes
`data/uav_arena/results/pets_phase_b_sel_g2_d5w220_s20k_seed0_openloop_val.json`.

G3 changes G1's training command to `--steps 40000` and writes
`data/uav_arena/participants/pets/phase_b_sel_g3_d4w248_s40k_seed0.pt`.
Its complete validation result is
`data/uav_arena/results/pets_phase_b_sel_g3_d4w248_s40k_seed0_openloop_val.json`.

G4 changes G3's training command to `--lr 0.0005` and writes
`data/uav_arena/participants/pets/phase_b_sel_g4_d4w248_s40k_lr5e-4_seed0.pt`.
Its complete validation result is
`data/uav_arena/results/pets_phase_b_sel_g4_d4w248_s40k_lr5e-4_seed0_openloop_val.json`.

G5 changes G3's training command to `--steps 60000` and writes
`data/uav_arena/participants/pets/phase_b_sel_g5_d4w248_s60k_seed0.pt`.
Its complete validation result is
`data/uav_arena/results/pets_phase_b_sel_g5_d4w248_s60k_seed0_openloop_val.json`.

G6 changes G3's training command to `--depth 3 --width 291` and writes
`data/uav_arena/participants/pets/phase_b_sel_g6_d3w291_s40k_seed0.pt`.
Its complete validation result is
`data/uav_arena/results/pets_phase_b_sel_g6_d3w291_s40k_seed0_openloop_val.json`.
Width 292 would exceed the parameter cap.

G7 changes G6 to `--steps 60000` and writes
`data/uav_arena/participants/pets/phase_b_sel_g7_d3w291_s60k_seed0.pt`.
Its complete validation result is
`data/uav_arena/results/pets_phase_b_sel_g7_d3w291_s60k_seed0_openloop_val.json`.

G8 changes G7 to `--lr 0.0005` and writes
`data/uav_arena/participants/pets/phase_b_sel_g8_d3w291_s60k_lr5e4_seed0.pt`.
Its complete validation result is
`data/uav_arena/results/pets_phase_b_sel_g8_d3w291_s60k_lr5e4_seed0_openloop_val.json`.

For the formal runs, G7's training command changes `--episodes` to either
`2000` or `500`, changes `--seed` to 1, 2, or 3, and writes
`data/uav_arena/participants/pets/phase_b_formal_d3w291_s60k_e<episodes>_seed<seed>.pt`.
Complete validation writes the matching
`data/uav_arena/results/phase_b_formal_d3w291_s60k_e<episodes>_seed<seed>_openloop_val.json`.

The central `test` run used, for each formal checkpoint, with `<m>` standing
for `e<episodes>_s<seed>`:

```bash
OPTIONS='{"checkpoint":"data/uav_arena/participants/pets/phase_b_formal_d3w291_s60k_e<episodes>_seed<seed>.pt","device":"cuda","sample_gaussian":false}'
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run \
  --predictor uav_arena.participants.pets.model:make --predictor-options "$OPTIONS" \
  --suite test --seeds 0:50 --workers 20 \
  --out data/uav_arena/results/pets_<m>_mppi_test.json
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.harness score \
  --predictor uav_arena.participants.pets.model:make --predictor-options "$OPTIONS" \
  --states data/uav_arena/states/test --workers 20 \
  --out data/uav_arena/results/harness/pets_<m>_test.json
```

