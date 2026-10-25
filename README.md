# Multi-agent world models and executable interaction interfaces

**Physical interaction coverage is an independent determinant of multi-agent
planning capability.** This repository connects locally learned world models
with joint action selection in cooperative pushing and three-UAV aerial
transport. Contact transitions and load-sharing geometry make coordinated task
corrections available to a frozen predictive system.

This is the **`codex/novelty` research snapshot, 2026-10-25**. The current studies
cover joint primary/helper planning in VMAS Push-T and cooperative aerial load
recovery using the official Menagerie Crazyflie 2 airframe. CrazyMARL supplies
the external multi-agent RL environment and training implementation. The former
self-built UAV baseline is preserved in a verified, runnable archive. The
original pixel and entity-token LeWM pipelines remain available as the modeling
foundation.

## Start here

| Material | What it contains |
|---|---|
| [Joint multi-agent planning](docs/planned-helper-audit.md) | Seven planned helper options, frozen joint models, matched 273-candidate interfaces, and 1,152 complete tasks |
| [Public airframes and external UAV baselines](docs/uav-external-baselines.md) | Official Crazyflie 2 models, frozen local world models, upstream Lee flight feedback, and CrazyMARL integration |
| [Archived self-built UAV baseline](archives/uav_transport/self_built_v2/README.md) | Frozen v2 source, results and six weights; fresh execution and full-state replay reproduce bitwise |
| [LeWM baselines and reproduction](docs/lewm-baselines.md) | Released checkpoints, pixel/state pipelines, readout fitting, original results, and training commands |
| [Push-T evidence](experiments/planned_helper/v1/) | Per-episode scores, paired comparisons, numerical checks, and portable frozen weights |
| [Current UAV evidence and frozen models](experiments/uav_transport/menagerie_v1/README.md) | New official-airframe results, external controller checks, numerical replay, and six retrained NumPy checkpoints |

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

### Cooperative aerial transport with the official Crazyflie 2 airframe

Each mission executes **three waypoint legs and two unloading/recovery
operations**. Constrained completion enforces a **5 cm payload-excursion** and
**0.6 N donor-tension** budget during each operation. All 216 nominal missions
complete delivery; the table measures compliance with those load-transfer
quality constraints.

| Predictor/controller | Fixed support geometry | Expanded cooperative interface |
|---|---:|---:|
| Small frozen local model | 0/36 | **36/36** |
| Large frozen local model | 0/36 | **36/36** |
| Exact option simulator | 0/12 | **12/12** |
| Study geometric feedback | 0/12 | **12/12** |
| Upstream Lee flight feedback with shared load allocation | 0/12 | **10/12** |

Local reconfiguration reduces peak payload excursion by **93.4%**, peak donor
tension by **75.5%**, and re-tension impulse by **92.4%** in the paired physical
audit. Held-out local selection retains its advantage with matched nine-option
budgets. The frozen models then select operations in complete missions at a
median **1.49–1.55 ms** per decision on the measured CPU setup.

All **192 model/study-control mission refinements** pass the trajectory-error thresholds and
retain every completion score. The new airframe preserves official mass,
inertia, meshes and collision geometry. Source attribution, external-baseline
status and the fresh-data protocol are recorded in the
[current aerial report](docs/uav-external-baselines.md).

The additional 24 upstream-Lee missions retain all task scores at 1 ms;
23/24 pass the original trajectory tolerance, with the remaining case and
finer-grid diagnostics recorded in the report. CrazyMARL's official IPPO
training, actor updates and checkpoint reload run on CPU and RTX 5070 GPU.
These short integration runs produce explicitly unconverged checkpoints;
the full training configuration is retained separately.

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

To open the minimal MuJoCo viewer for the new official-airframe mission:

```bash
data/uav_transport/.venv/bin/python -m uav_transport.viewer \
  data/uav_transport_external/mission_menagerie_v1_wm_small_seed0_mass026_donor0_route0.pkl
```

Drag to rotate, scroll to zoom, press **P** to pause/resume or **R** to restart.
The recording loops. An optional positional path selects another local mission
pickle; complete raw trajectories are kept in the local `data/` directory.

The combined branch passes **274 tests** in the Torch/VMAS environment and
**108 tests** in the UAV environment. Optional dependencies account for twelve
and three skips respectively; the two Torch-training checks run in the first
environment. Full collection, training, evaluation and replay commands live in
the two study reports. Use fresh output prefixes to retain existing runs.

The isolated CrazyMARL environment passes its six integration tests. Set it up
with `python3 scripts/setup_crazymarl.py`; use `--cuda` with a separate `--venv`
for the validated GPU environment. The current aerial report gives execution
commands and source provenance.

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

The current UAV plant family is **`menagerie_crazyflie_2`**. New collection,
physical audits and mission execution select it with
`--plant menagerie_crazyflie_2`; the current report provides the commands.
The former self-built `mission_v2` study is independently preserved under
`archives/uav_transport/self_built_v2/`, with original raw records retained under
`data/uav_transport/`. Historical scores retain their original task definitions
and corrections in their reports.
Parallel Navigation work is documented on
[`codex/e1-ranking-e4-integration`](https://github.com/ArcueidMP/vincair-multi-agent-wm/tree/codex/e1-ranking-e4-integration).

## Source and evidence layout

| Path | Role |
|---|---|
| `world.py`, `policies.py`, `collect.py` | Pixel MultiPush-T environment and scripted data collection |
| `jepa.py`, `module.py`, `train.py`, `plan.py`, `readout.py` | Pixel LeWM, training, CEM planning, and state readout |
| `vmas_pusht/{model,train,plan}.py` | Entity-token JEPA baseline and state-based planning |
| `vmas_pusht/action_interface.py`, `vmas_pusht/joint_*.py` | Executable contact options, joint helper planning, training, evaluation, and audits |
| `uav_transport/` | Official-airframe transport, external controller/RL adapters, snapshots, local models, mission execution, and replay |
| `third_party/` | Pinned upstream Crazyflie meshes and native Lee controller files, manifests and licenses |
| `archives/uav_transport/self_built_v2/` | Immutable, independently executable historical UAV baseline |
| `docs/` | Current research reports and chronological audit records |
| `experiments/` | Versioned compact evidence, per-task results, hashes, and frozen models |
| `data/` | Gitignored raw datasets, complete trajectories, checkpoints, and historical source archives |
| `tests/`, `config/`, `scripts/` | Regression checks, baseline configurations, and release utilities |

Each UAV evidence package maps copied files and summary fields back to its
SHA256-identified outputs. Current and archived plant families retain their
own data, results and frozen models.

## Foundations

The original modeling and evaluation pipeline builds on
[LeWM](https://github.com/lucas-maes/le-wm),
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), and
[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator).
The aerial experiment uses MuJoCo dynamics; its physical formulation and
reference implementations are documented in the aerial report.
