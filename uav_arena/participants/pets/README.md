# PETS arena participant

Five independently bootstrapped probabilistic MLPs, each with four 200-unit
Swish hidden layers, predict a diagonal Gaussian over the next physical-state
increment. Inputs and targets use training-set normalization. The first step
bridges the observation age plus one decision; subsequent steps use a rolling
four-frame state history and the setpoints already sent. Positions and
setpoints are centered on the current payload position at every predicted step.

The shared planner accepts one trajectory per candidate. The default adapter
therefore draws one ensemble member for each candidate at each planning call
and keeps that member for all 15 predicted decisions (TS-infinity), using the
member's Gaussian mean at every step. This is **one particle per candidate**,
whereas published PETS uses 20 particles and averages their costs. Gaussian
increment sampling remains available through `sample_gaussian=true` for
diagnostics, but is not used for the benchmark row. The predictor declares
`stochastic = True`, uses torch's global random generator and reseeds it in
`reset()`, as required by the participant contract.

Phase A commands from the repository root (standard data under `data/`):

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.pets.train \
  --data data/uav_arena/datasets/explore_v2 --episodes 250 --steps 10000 \
  --width 256 --device cuda \
  --out data/uav_arena/participants/pets/phase_a_v2_w256_s10k.pt

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop \
  --predictor uav_arena.participants.pets.model:make \
  --predictor-options '{"checkpoint":"data/uav_arena/participants/pets/phase_a_v2_w256_s10k.pt","device":"cuda"}' \
  --data data/uav_arena/datasets/explore_v2_val --stride 10 \
  --out data/uav_arena/results/pets_phase_a_v2_w256_s10k_openloop_val_stride10.json

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run \
  --predictor uav_arena.participants.pets.model:make \
  --predictor-options '{"checkpoint":"data/uav_arena/participants/pets/phase_a_v2_w256_s10k.pt","device":"cuda"}' \
  --suite dev --seeds 0:5 \
  --out data/uav_arena/results/pets_phase_a_v2_w256_s10k_dev_s0-5.json
```

Phase A selection scores on the 100-episode validation set, every tenth
decision (lower is better):

| Configuration | Payload error |
| --- | ---: |
| Example Phase A bar, 250 episodes, 5,000 steps | 8.26 cm |
| PETS, width 200, 5,000 steps/member, mean propagation | 9.42 cm |
| PETS, width 256, 10,000 steps/member, mean propagation | **7.12 cm** |

The selected configuration uses 1.44M parameters and took 665.75 s of wall
time on an RTX 5060 Laptop GPU (about 0.185 GPU-hours). A 512-candidate,
15-step planning call averaged 23.4 ms during its one `dev` flight. It passed
the Phase A open-loop bar and then succeeded on 7/40 `dev` episodes.

Checkpoints, diagnostics and JSON results remain under ignored `data/`; only
model code, tests and these commands belong in the pull request.
