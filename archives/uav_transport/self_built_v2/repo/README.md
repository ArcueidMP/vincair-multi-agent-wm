# Multi-agent world models and executable interaction interfaces

**Physical interaction coverage is an independent determinant of multi-agent
planning capability.** This repository connects locally learned world models
with joint action selection in cooperative pushing and three-UAV aerial
transport. Contact transitions and load-sharing geometry make coordinated task
corrections available to a frozen predictive system.

This is the **`codex/novelty` research snapshot, 2026-10-25**. The latest completed
studies are joint primary/helper planning in VMAS Push-T and cooperative aerial
load recovery under the final `mission_v2` protocol. The original pixel and
entity-token LeWM pipelines remain available as the modeling foundation.

## Start here

| Material | What it contains |
|---|---|
| [Joint multi-agent planning](docs/planned-helper-audit.md) | Seven planned helper options, frozen joint models, matched 273-candidate interfaces, and 1,152 complete tasks |
| [Cooperative aerial transport](docs/uav-transport-world-model-audit.md) | Three full-dynamics UAVs, local outcome models, load redistribution, and 192 complete transport missions |
| [LeWM baselines and reproduction](docs/lewm-baselines.md) | Released checkpoints, pixel/state pipelines, readout fitting, original results, and training commands |
| [Push-T evidence](experiments/planned_helper/v1/) | Per-episode scores, paired comparisons, numerical checks, and portable frozen weights |
| [UAV evidence and frozen models](experiments/uav_transport/v2/README.md) | Existing run summaries, all mission results, source hashes, and six NumPy checkpoints |

## Latest results

### Joint contact planning

The planner selects both agents' options. Direct and expanded interfaces each
have **273 candidates**, preserving the primary-force/duration/helper programs.
The table includes all three prespecified contact-route allocations.

| Joint predictor | Direct interface | Expanded interface, allocations 0 / 1 / 2 |
|---|---:|---:|
| Frozen learned model | 0/108 | **71 / 68 / 81**, each out of 108 |
| True-option simulator | 0/36 | **32 / 34 / 34**, each out of 36 |

The bottleneck persists under joint helper planning and true dynamics.
**All 864 paired interface comparisons have zero regressions at both tested
resolutions.** Locally trained models are frozen before the composed tasks.
The study reports numerical replay, both agents' work, travel, and actual
prediction/simulation costs in the [full report](docs/planned-helper-audit.md).

### Cooperative aerial transport

Each mission executes **three waypoint legs and two unloading/recovery
operations**. Constrained completion enforces a **5 cm payload-excursion** and
**0.6 N donor-tension** budget during each operation. All 192 nominal missions
complete delivery; the table measures compliance with those load-transfer
quality constraints.

| Predictor/controller | Fixed support geometry | Expanded cooperative interface |
|---|---:|---:|
| Small frozen local model | 0/36 | **35/36** |
| Large frozen local model | 0/36 | **36/36** |
| Exact option simulator | 0/12 | **12/12** |
| Complete geometric feedback | 0/12 | **12/12** |

Local reconfiguration reduces peak payload excursion by **93.3%**, peak donor
tension by **65.3%**, and re-tension impulse by **90.7%** in the paired physical
audit. Held-out local selection retains its advantage with matched nine-option
budgets. The frozen models then select operations in complete missions at a
median **1.7–2.8 ms** per decision on the measured CPU setup.

All **192 full-mission refinements** pass the trajectory-error thresholds.
The nominal and refined constraint scores, arrival guard, and propulsion costs
are recorded in the [aerial report](docs/uav-transport-world-model-audit.md).

Learned denominators pool three fit seeds over 36 Push-T or 12 UAV physical
conditions. The reports keep physical cases, seed repeats, local selections,
and complete-task scores explicit.

## Environments and checks

The current interface studies run on CPU. Use Python 3.10 and create the
following environments when they are absent; existing environments can be reused.

```bash
# Torch/VMAS research environment; also trains the UAV outcome models.
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-novelty.txt

# Lightweight MuJoCo environment for UAV execution and NumPy inference.
mkdir -p data/uav_transport
uv venv --python 3.10 data/uav_transport/.venv
uv pip install --python data/uav_transport/.venv/bin/python -r requirements-uav.txt
```

Run from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m pytest -q tests/test_uav_*.py
```

To open the minimal MuJoCo viewer for the locally recorded small-model mission:

```bash
data/uav_transport/.venv/bin/python -m uav_transport.viewer
```

Drag to rotate, scroll to zoom, press **P** to pause/resume or **R** to restart.
The recording loops. An optional positional path selects another local mission
pickle; complete raw trajectories are kept in the local `data/` directory.

The combined branch passes **269 tests** in the Torch/VMAS environment and
**83 tests** in the UAV environment. Optional dependencies account for eight
and two skips respectively; the two Torch-training checks run in the first
environment. Full collection, training, evaluation and replay commands live in
the two study reports. Use fresh output prefixes to retain existing runs.

The [pixel LeWM setup](docs/lewm-baselines.md#install) uses the CUDA environment
and released [Hugging Face checkpoints](https://huggingface.co/fuisloy/vincair-multi-agent-wm).

## Research history and reading order

The audit sequence records how the current interaction-interface result was
established. Start with the two latest reports above, then use these references
for the underlying controls and measurements.

| Stage | Record | Role in the current result |
|---|---|---|
| Cooperation and exact state | [Boundary and pairing audit](docs/novelty-audit.md) | Complete simulator snapshots, solo/joint measurements, and fixed-marginal pairing |
| Physical support | [Physical-abstraction audit](docs/physical-abstraction-audit.md) | Hidden-state sensitivity, bounded support, local models, and transfer |
| Contact measurement | [Recovery and convergence](docs/rebrace-convergence-audit.md) | Reaction-force contact criteria and corrected historical composition scores |
| Stable control regime | [Stable physics and retraining](docs/stable-physics-local-wm-audit.md) | `dry_light`, equal-budget local training, and recovery handoff diagnosis |
| Recovery mechanism | [Attribution and simulator MPC](docs/recovery-attribution-mpc-audit.md) | Paired actuator interventions and physically executed reverse-contact routes |
| Learned composition | [Completed action interface](docs/completed-action-interface-audit.md) | Goal-free option learning and frozen complete-task evaluation |
| Predictor and search controls | [Predictor × interface factorial](docs/predictor-interface-factorial-audit.md) | Capacity, exact prediction, and matched candidate-budget controls |
| Joint multi-agent planning | [Planned helper](docs/planned-helper-audit.md) | Both agents become planner decision variables |
| Aviation application | [Three-UAV load recovery](docs/uav-transport-world-model-audit.md) | Load-sharing geometry, frozen joint predictions, and constrained transport missions |

The latest UAV protocol is **`mission_v2`**; earlier `mission_v1` results and
source archives remain available under `data/uav_transport/`. Historical
scores retain their original task definitions and corrections in their reports.
Parallel Navigation work is documented on
[`codex/e1-ranking-e4-integration`](https://github.com/ArcueidMP/vincair-multi-agent-wm/tree/codex/e1-ranking-e4-integration).

## Source and evidence layout

| Path | Role |
|---|---|
| `world.py`, `policies.py`, `collect.py` | Pixel MultiPush-T environment and scripted data collection |
| `jepa.py`, `module.py`, `train.py`, `plan.py`, `readout.py` | Pixel LeWM, training, CEM planning, and state readout |
| `vmas_pusht/{model,train,plan}.py` | Entity-token JEPA baseline and state-based planning |
| `vmas_pusht/action_interface.py`, `vmas_pusht/joint_*.py` | Executable contact options, joint helper planning, training, evaluation, and audits |
| `uav_transport/` | Three-UAV plant, feedback, snapshots, local models, mission execution, and replay |
| `docs/` | Current research reports and chronological audit records |
| `experiments/` | Versioned compact evidence, per-task results, hashes, and frozen models |
| `data/` | Gitignored raw datasets, complete trajectories, checkpoints, and historical source archives |
| `tests/`, `config/`, `scripts/` | Regression checks, baseline configurations, and release utilities |

The UAV evidence package organizes existing completed runs. Its index maps
copied files and summary fields back to the original SHA256-identified outputs.
The raw scientific records and frozen model values are preserved.

## Foundations

The original modeling and evaluation pipeline builds on
[LeWM](https://github.com/lucas-maes/le-wm),
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), and
[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator).
The aerial experiment uses MuJoCo dynamics; its physical formulation and
reference implementations are documented in the aerial report.
