# PETS Phase B progress report

Status: selection is complete after all eight allowed configurations on the
frozen `arena-v0.3` protocol. G7 (depth 3, width 291, 60,000 steps/member,
learning rate 1e-3) is selected for the formal seed runs. Every
selection run uses all 2,000 `explore_v2` episodes, seed 0, mean propagation
(`sample_gaussian=false`) and every decision of all 100 `explore_v2_val`
episodes. Neither `dev` nor `test` is used for selection.

The total parameter limit is 1,372,047, counting all five ensemble members and
their bounded log-variance parameters. The search is staged and will use at
most eight configurations: compare the largest compliant depth-4 and depth-5
models first, then spend longer schedules or learning-rate checks only on the
better architecture. Every attempted configuration remains in the table.

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

