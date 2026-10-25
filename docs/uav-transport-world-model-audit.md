# Cooperative Aerial Load Recovery with Frozen Local World Models

2026-10-25, `codex/novelty`.

[Branch overview](../README.md) · [Published evidence and models](../experiments/uav_transport/v2/README.md)

**Executable load-sharing geometry controls the quality of cooperative aerial
transport. Frozen local models can predict recovery outcomes and select joint
maneuvers for three UAVs, transferring from isolated operation data to complete
transport missions.**

This aviation experiment combines a physical multi-agent system, learned joint
operation models, and closed-loop motor execution. The sequence is transport,
controlled unloading, load redistribution, re-tension, and further transport.
Together with the [joint multi-agent Push-T study](planned-helper-audit.md), it
establishes executable interaction geometry as a planning capability across
ground contact and aerial load sharing. The flight task adds propulsion limits,
cable-force transients, and complete constrained transport missions.

## 1. Physical team and control architecture

The MuJoCo 3.3.0 plant contains three independently rotating/translating UAVs,
twelve rotor actuators, and a central-attachment spherical payload. The UAVs
share load through three fixed-length, tension-only spatial tendons. A motor
command passes through a 30 ms first-order response before applying local rotor
thrust and reaction torque. The controller runs every 20 ms; the initial physics
step is 2 ms.

| Property | Value |
|---|---|
| UAV mass | 0.034 kg each |
| Payload mass | 0.024–0.036 kg in the physical audit |
| Rotor force range | 0–0.16 N per rotor |
| Cable length | 0.6 m |
| Nominal formation radius | 0.25 m |
| Initial payload height | 1.2 m in the physical audit and missions |
| State coordinates | 28 configuration, 24 velocity, 12 motor activation |
| Observation/selection | Centralized team state and joint operation selection |

The common feedback controller performs nonnegative bounded tension allocation,
payload/formation feedback, geometric attitude control, and bounded rotor
mixing. Shared controller gains are fixed across all planning arms. Calibrated
plant parameters are supplied to every controller and predictor.

Geometry changes are executed by flight. Supporting UAVs follow constant-radius
arcs; the donor approaches the payload to create actual slack, then closes the
gap at bounded reference speed. Measured cable tension, relative closing speed,
payload error, and sustained contact determine recovery completion. Commands
preserve the cable lengths and attachment points.

The complete snapshot includes MuJoCo integration state, motor activation,
controls, solver warm starts, external forces, clocks, and measurement counters.
Feedback integrals and recovery-protocol memory are separately captured. These
states support exact counterfactual forks and saved-action replay.

Implementation: [plant](../uav_transport/plant.py), [control and recovery
protocol](../uav_transport/control.py), [snapshots](../uav_transport/snapshot.py),
and [physical outcome measurement](../uav_transport/evaluation.py).

## 2. A checkable cooperative mechanism

At the nominal 30 g payload, symmetric three-UAV hover requires approximately
0.10791 N of cable tension and 0.43398 N total thrust per UAV. After moving the
two continuing supports to opposing directions, each needs approximately
0.16187 N tension and 0.48540 N thrust, within its 0.64 N propulsion limit.

The geometry explains the intervention. Retaining two vertices of the original
triangle places the passive load equilibrium 0.125 m horizontally away from the
original payload position. Opposing supports balance the load at that position.

The registered physical audit covers 18 conditions: three payload masses, each
donor UAV, and two formation orientations. Each condition starts both operators
from the same complete settled state and uses the same feedback controller,
unloading requirement, force limits, and recovery duration.

| Mean physical outcome | Fixed support geometry | Reconfigured support geometry |
|---|---:|---:|
| Verified operator completion | 18/18 | 18/18 |
| Operation time | 9.0 s | 9.0 s |
| Peak payload excursion | 0.20346 m | **0.01364 m** |
| Peak donor cable tension | 1.2018 N | **0.4170 N** |
| Re-tension impulse | 0.10203 Ns | **0.00948 Ns** |
| Absolute actuator mechanical work | 0.13407 J | **0.08223 J** |

Reconfiguration reduces peak excursion **93.3%**, peak cable tension **65.3%**,
and re-tension impulse **90.7%**. Every pair improves all three quantities and
mechanical work. Squared-thrust effort rises 1.31%, providing a separate measure
of the propulsion tradeoff. Mechanical work is the integral of actuator force
times transmission velocity; squared thrust is an effort proxy.

All 36 nominal replays reproduce full integration state bitwise. Saved-action
refinements at 1 ms and 0.5 ms preserve every completion result and pass the
registered position, tension-peak, and impulse checks. Maximum body-position
discrepancy is 0.0006381 m; maximum peak-tension relative change is 9.61%; maximum
impulse discrepancy is 0.0000835 Ns. All trajectories remain finite and clear of
collisions, with rotor force at most 0.12995 N and UAV tilt at most 9.04 degrees.

Report: [registered physical-audit summary](../experiments/uav_transport/v2/summary.json).

## 3. The learned team-level world model

The learned model is a joint **operation-outcome model**. Its 143 inputs contain
the three UAV states relative to the payload, rotations and angular velocities,
motor activations, measured cable gaps/tensions, feedback integrals, calibrated
plant parameters, donor identity, and the declared recovery program. It predicts
eight outcomes:

1. Peak payload excursion.
2. Peak donor cable tension.
3. Operation duration.
4. Absolute actuator mechanical work.
5. Squared-thrust integral.
6. Terminal payload position error.
7. Terminal payload speed.
8. Local completion score.

The output describes a closed-loop physical operation. Its selected parameters
drive all three UAV controllers, which produce twelve bounded rotor commands.
The local holding reference is the current payload position; mission waypoints
are supplied to the flight controller outside the model input.

The dataset contains **888 individually executed local operations** from 44
independent physical contexts, with all counterfactual candidates sharing their
context's complete start state and controller memory:

| Split | Contexts | Operation examples |
|---|---:|---:|
| Training | 24 | 432 |
| Validation | 8 | 144 |
| Held-out contexts, original action bank | 12 | 216 |
| New action durations on those held-out contexts | same 12 | 96 |

Physical preludes supply motion history and controller integrals. Each candidate
specifies support-geometry mode, rearrangement time, and re-tension time; the
required donor and slack distance belong to the operation context. Training
uses individual recovery operations, with complete multi-waypoint missions
reserved for subsequent evaluation.

Two MLP sizes, **5,928 and 35,976 parameters**, each use three fit seeds. Input
normalization uses training data; validation selects checkpoints and provides
95th-percentile positive-underprediction margins for excursion and tension.
All six networks are frozen before mission execution. NumPy inference is
checked against the training network and requires CPU only.

The planner ranks predicted constraint-feasible candidates by duration and
effort. When that set is empty, it chooses the smallest predicted violation.
The recorded decision includes every prediction, feasibility margin, candidate
rank, and selection path. Completion labels in this dataset are all positive;
excursion, tension, time, and effort supply the principal decision signal.

Implementation and data: [local models](../uav_transport/local_model.py),
[data collection](../uav_transport/collect.py),
[dataset summary](../experiments/uav_transport/v2/summary.json),
[frozen model manifest](../experiments/uav_transport/v2/models/models_v1_manifest.json).

## 4. Held-out local decisions at matched candidate budgets

The direct bank contains nine duration pairs on the original support geometry.
The expanded bank supplies the same pairs on both geometries, for 18 candidates.
Two prespecified nine-candidate allocations preserve every duration pair once
and alternate which geometry receives it. All evaluations use the held-out
physical outcomes; the models remain frozen.

An accepted local choice completes its operation within 18 s while keeping peak
payload excursion within 0.05 m and donor tension within 0.6 N.

| Predictor | Direct9 | Expanded18 | Allocated9, shift 0 | Allocated9, shift 1 |
|---|---:|---:|---:|---:|
| Small, three seeds | 0/36 | **34/36** | **36/36** | **34/36** |
| Large, three seeds | 0/36 | **35/36** | **36/36** | **35/36** |
| Exact counterfactual outcomes | 0/12 | **12/12** | **12/12** | **12/12** |

Every direct bank lacks a choice satisfying the declared excursion/tension
budgets. Every expanded and allocated bank contains one. Thus the geometry
intervention retains its value when candidate counts and duration-parameter
coverage are matched. These are local-choice results; complete missions are
evaluated separately below.

The three learned selection misses all use the same rearrange/re-tension timing
class (1 s / 3 s) and exceed the donor-tension limit, with actual peaks of
0.644–0.706 N. Both allocations were fixed before examining these outcomes.
The audit preserves all candidate predictions, ranking errors, and constraint
margin coverage.

Report: [held-out action-selection summary](../experiments/uav_transport/v2/summary.json).

## 5. Complete aviation missions

Each mission contains three waypoint legs and two unloading/recovery operations,
with different donor UAVs in the two operations. The 12 physical conditions
combine two calibrated payload masses, all three donor identities, and two
waypoint/formation templates. The small and large predictors each use all three
frozen seeds. Complete feedback rules and exact-simulator selectors run once
per physical condition, giving 192 mission executions per registered protocol.

Delivery requires all waypoint dwells, both measured recovery completions, at
least 0.1 s of real donor slack per operation, finite dynamics, and no collisions
or warnings. Constrained completion additionally requires each operation's
payload excursion to stay within 5 cm and donor tension within 0.6 N. These
quality requirements make the load-transfer transient part of mission success.

All planning arms share the flight and recovery controllers, calibrated state,
rotor limits, 6 s leg budgets, and 18 s operation budgets. The complete geometric
feedback rule is included alongside the fixed-geometry ablation. Planning wall
time and physical execution time are recorded separately.

The final registered protocol uses a stricter **20 mm / 0.040 m/s internal
arrival guard** for 0.3 s. The external waypoint contract remains **25 mm /
0.050 m/s**. Additional settling steps consume the original leg budget. The
earlier protocol is retained as `mission_v1`; its sources are archived under
`data/uav_transport/mission_v1_sources/`. The final `mission_v2` uses exactly the
same model weights, candidate bank, controller gains, task constraints, and
planning cost.

| Predictor/controller | Deliveries, fixed geometry | Deliveries, expanded geometry | Constrained completion, fixed → expanded |
|---|---:|---:|---:|
| Small frozen local model | 36/36 | 36/36 | **0/36 → 35/36** |
| Large frozen local model | 36/36 | 36/36 | **0/36 → 36/36** |
| Exact option simulator | 12/12 | 12/12 | **0/12 → 12/12** |
| Complete geometric feedback rule | 12/12 | 12/12 | **0/12 → 12/12** |

The learned rows contain 12 physical cases under each of three fit seeds. The
small-model constrained counts are 12/12, 11/12, and 12/12; the larger model
achieves 12/12 under every seed. Both models select physical reconfiguration in
every expanded-interface operation. The small-model exceedance reaches a
0.62590 N donor-tension peak against the 0.6 N budget, while completing all
waypoints and recovery gates.

This makes the aviation contribution concrete: **the interface determines
whether the team can carry out load transfers while maintaining cargo-motion
and cable-force limits. Local learned predictions successfully select such
cooperative operations in complete missions assembled after training.**

### Decision provenance and computation

All 96 direct/expanded mission pairs begin with bitwise-identical full physical
and integration states, feedback memory, and the entire flight prefix before
the first operation. All **288 learned decisions** reproduce frozen model
predictions and the selected argmin exactly; their future-simulator query count
is zero. For all **48 exact-simulator decisions**, the selected eight-outcome
prediction matches its subsequent actual execution exactly.

Expanded learned selection has median latency **2.78 ms** for the small model
and **1.68 ms** for the larger model; the corresponding 95th percentiles are
3.01/3.08 ms. All recorded learned decisions fit within the 20 ms control period.
The exact expanded option selector has a **28.48 s** median per decision.
These are measured wall times under eight-worker CPU execution. The simulator
selector evaluates counterfactual options while the parent physical clock is
held; its compute is reported separately from execution duration.

| Expanded policy | Mean mission duration | Mean mission maximum donor tension | Mean absolute actuator work |
|---|---:|---:|---:|
| Small local model | 31.92 s | 0.35666 N | 0.39883 J |
| Large local model | 32.52 s | 0.32171 N | 0.39235 J |
| Exact option simulator | 27.81 s | 0.40883 N | 0.46373 J |
| Complete geometric rule | 30.81 s | 0.41697 N | 0.39783 J |

The models achieve millisecond selection and favor lower predicted-force
transients. The exact selector chooses the fastest constraint-feasible programs
in its bank. The table preserves the resulting time/force/work tradeoffs.

The validation margins are intentionally frozen with the model. Minimum-
predicted-violation selection is used in 54/72 small-model and 66/72 large-model
expanded decisions; those choices still follow the learned outcome ranking.
The observed task counts above come from actual dynamics and measured limits.

Reports: [final mission matrix](../experiments/uav_transport/v2/mission_v2.json) and
[paired-state, prediction-provenance, cost, and latency summary](../experiments/uav_transport/v2/summary.json).

## 6. Whole-mission numerical verification

The final protocol is replayed from its complete initial state, holding every
recorded rotor command, disturbance, phase boundary, and stopping time fixed.
All **192 nominal missions reproduce their physical/integration states and
independently derived scores exactly**. All **192 finer 1 ms replays** pass the
registered 2 mm body-position and 0.02 rad attitude discrepancy thresholds;
observed maxima are **1.5302 mm / 0.0006080 rad**.

Every mission delivers successfully at both resolutions, and every waypoint
dwell retains its success. At 1 ms, expanded small and large models each achieve
36/36 constrained completions; exact simulation and the reconfiguration rule
each achieve 12/12. Every fixed-geometry arm remains at zero constrained
completions.

The single changed constrained label is the small-model tension-boundary case:
its peak is 0.62590 N at 2 ms, 0.58183 N at 1 ms, and 0.58581 N in a targeted
0.5 ms replay. The primary table retains the nominal 35/36 count and records
the refinement result separately. The stricter arrival guard removes the
waypoint-window sensitivity observed in the earlier protocol while preserving
the original external task requirements and charging every extra control step.

Reports: [final replay summary](../experiments/uav_transport/v2/summary.json),
earlier protocol control (`data/uav_transport/mission_v1_replay.json`), and
[targeted tension-boundary replay](../experiments/uav_transport/v2/mission_v2_boundary_dt0005.log).

**The resulting system combines physical multi-agent cooperation with learned
joint outcome prediction: locally trained, frozen models select executed
load-sharing transitions, complete new transport sequences, and retain their
interface advantage under numerical refinement.**

## Reproduction

```bash
uv venv --python .venv/bin/python data/uav_transport/.venv
uv pip install --python data/uav_transport/.venv/bin/python -r requirements-uav.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.physics_audit --out data/uav_transport/physics_v1 --workers 4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.collect --out data/uav_transport/local_v1 --workers 8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m uav_transport.train --data data/uav_transport/local_v1.npz \
  --out data/uav_transport/models_v1 --workers 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.selection_audit --data data/uav_transport/local_v1.npz \
  --metadata data/uav_transport/local_v1.json \
  --models data/uav_transport/models_v1_manifest.json --out data/uav_transport/selection_v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.mission --models data/uav_transport/models_v1_manifest.json \
  --out data/uav_transport/mission_v2 --workers 8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.mission_analysis --manifest data/uav_transport/mission_v2.json \
  --models data/uav_transport/models_v1_manifest.json --out data/uav_transport/mission_v2_analysis
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.mission_replay --manifest data/uav_transport/mission_v2.json \
  --fine-dt .001 --workers 8 --out data/uav_transport/mission_v2_replay
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m pytest -q tests/test_uav_*.py
```

Train each width/seed with the existing Torch environment, then use the frozen
NPZ models from the lightweight simulator environment. Fresh output prefixes
preserve earlier artifacts. [Compact evidence and all six frozen model files](../experiments/uav_transport/v2/README.md)
are published with source hashes. Full raw trajectories, candidate arrays,
historical protocol archives, and Chinese working notes remain under the local
`data/uav_transport/` tree. The evidence guide also provides a direct model-restore
path for mission evaluation.

The combined branch passes **269 tests** in the Torch/VMAS environment, with
eight optional MuJoCo checks skipped. The UAV environment passes **83 tests**;
its two optional Torch-training tests run in the first environment. The original
UAV run provenance retains its stage-specific 257/83 test counts. The training launcher
also reproduces both seed-0 model sizes' complete weight and normalization
arrays bitwise.

## Primary references

The plant uses [MuJoCo's constraint and actuator dynamics](https://mujoco.readthedocs.io/en/stable/computation/index.html).
Its vehicle-scale mass, inertia, and rotor-force magnitudes are grounded in the
[CrazyMARL reference implementation](https://github.com/IMRCLab/CrazyMARL), which
also studies multi-UAV cable transport and slack/taut transitions.
[RotorTM](https://github.com/arplaboratory/RotorTM) provides a complementary
hybrid-dynamics formulation of aerial transportation. The present experiments
focus on paired interface interventions, frozen local outcome prediction, and
composition into complete flight tasks.
