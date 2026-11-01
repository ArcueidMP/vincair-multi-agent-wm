# PETS Phase B progress report

Status: selection is in progress on the frozen `arena-v0.3` protocol. Every
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

The example trained on all 2,000 episodes scores 6.24 cm. G1 is 0.40 cm worse,
so selection continues rather than declaring it final.

G1's complete validation results (7,429 decisions per latency) are:

| Latency | Payload | Aircraft |
| --- | ---: | ---: |
| 0 ms | 5.12 cm | 4.71 cm |
| 50 ms | 5.71 cm | 5.26 cm |
| 100 ms | 6.77 cm | 6.24 cm |
| 200 ms | 8.97 cm | 8.21 cm |
| **Mean payload selection score** | **6.64 cm** | — |

G1 trained on an RTX 5060 Laptop GPU. Its ignored local checkpoint is 5.3 MB
and has SHA-256
`1afe1cf75d214546419089f700aadd4c7512cca9c5e1ef2b9f061afcf9aa3a03`.
The checkpoint and raw result JSON are not committed.

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

The next registered configuration is G2: depth 5, width 220, 20,000 steps per
member, learning rate 1e-3 and seed 0. It has 1,366,120 parameters.

