# UAV transport with public airframes and external baselines

**The cooperative action-interface result carries over to the official
Menagerie Crazyflie 2 airframe.** Locally trained, frozen small and large world
models each complete **36/36** composed transport missions within the registered
load-transfer limits when cooperative reconfiguration is available. Both
achieve **0/36** within those limits with the original support geometry. Exact
option prediction gives the same **0/12 to 12/12** transition.

This is the current UAV track on `codex/novelty`. The former self-built v2
baseline has been frozen in an independently runnable
[archive](../archives/uav_transport/self_built_v2/README.md). Its source,
results and weights remain attached to that historical plant.

## Public components and their roles

| Component | Pinned source | Role |
|---|---|---|
| Crazyflie 2 airframe | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f3b6013629c7b94f8ee93fd977b161fa8350f9bc/bitcraze_crazyflie_2) | Official mesh, collision decomposition, mass and inertia in the transport plant |
| Multi-agent RL environment and algorithm | [CrazyMARL](https://github.com/IMRCLab/CrazyMARL/tree/fca7b6e05f3b5e816be7c982c91d69eb92925f91) | Original three-quadrotor MJX environment and IPPO training implementation |
| External position/attitude feedback | [IMRCLab Crazyflie firmware](https://github.com/IMRCLab/crazyflie-firmware/blob/b08d9f3cbcc581d8e7f988cabad22c26d35f4a58/src/modules/src/controller_lee.c) | Native upstream Lee controller, with the shared payload allocator and recovery references |
| Local operation world models | This repository | Predict joint physical outcomes and select cooperative operations |

The firmware revision is the dependency pinned by
[coltrans-planning](https://github.com/IMRCLab/coltrans-planning/tree/146b205e33e20ac26975993d32ec89b140754550/deps/crazyflie-firmware).
Upstream file bytes and license notices are retained. Menagerie is MIT-licensed;
CrazyMARL is Apache-2.0. The native controller bundle includes MIT-noticed C/math
files and GPL-3.0 firmware headers, with the original GPL license included.

## Airframe and physical execution

All three aircraft preserve the compiled upstream values exactly:

- Mass: **0.027 kg** per aircraft.
- Diagonal inertia: **(2.3951e-5, 2.3951e-5, 3.2347e-5) kg m²**.
- Seven visual meshes and **32 collision pieces** per aircraft, including
  their transforms, friction and collision masks.
- Center of mass, inertial frame and IMU sites.

The transport assembly adds a central-attachment spherical payload, three
tension-only fixed-length cables, four bounded rotor forces per aircraft, and
30 ms first-order motor dynamics. Per-rotor force remains limited to 0.16 N;
the arm radius and reaction-torque ratio remain declared study parameters.
The upstream description uses body-wrench actuators with explicitly arbitrary
ranges; the transport assembly uses the rotor-level actuation above. Air
density and viscosity retain the official description's values. Integration
uses the registered `implicitfast` scheme, 2 ms physics steps and 20 ms control
periods.

`MenagerieTransportEnv` preserves the existing observation, resource and
complete-state interfaces. Its model hash covers both the composed XML and
the pinned asset manifest. Clones use independent model/data copies. Recorded
plant-family identifiers route replay and visualization to the matching model.
Model execution rejects weights trained on the other plant family.

## Physical audit

The paired audit uses three payload masses, three unloading aircraft and two
formation headings: **18 physical conditions, 36 operator executions**. Each
pair begins from the same complete, physically settled state and controller
memory. Both operators use the same feedback controller, actuator limits and
9 s execution time.

| Mean local outcome | Original support geometry | Cooperative reconfiguration |
|---|---:|---:|
| Peak payload excursion | 0.21199 m | **0.01402 m** |
| Peak unloading-cable tension | 1.29924 N | **0.31825 N** |
| Re-tension impulse | 0.11804 N s | **0.00900 N s** |
| Absolute actuator mechanical work | 0.11815 J | **0.06877 J** |
| Squared-thrust integral | 0.91201 N² s | 0.92920 N² s |

All 18 pairs reduce excursion, peak tension, impulse and mechanical work.
Squared-thrust effort increases by 1.88%. Mechanical work is the substep
integral of actuator force times transmission velocity.

All 36 nominal full-state replays are bitwise identical. Saved-action replay at
1 ms and 0.5 ms preserves every completion score and passes every registered
continuous check. Maximum position discrepancy is **0.546 mm**, maximum peak
tension discrepancy is 9.73%, and maximum impulse discrepancy is 8.49e-5 N s.

## Fresh local training and frozen composition

The new plant supplies **888 local operator executions**: 432 training, 144
validation, 216 held-out-context and 96 held-out-action examples. Splits are
grouped by complete physical context. Data collection uses local recovery
operations after independent physical histories; composed mission trajectories
and targets are excluded from fitting.

The 143-input, eight-outcome models retain widths 32 and 128, with **5,928** and
**35,976** parameters. Three fit seeds per size use the same 1,500-step training
budget and validation checkpoint selection. All six models and their
validation-only uncertainty margins are frozen before mission execution.

Held-out local selection gives:

| Predictor | Direct nine options | Expanded 18 options | Matched nine options, allocation 0 / 1 |
|---|---:|---:|---:|
| Small local model | 0/36 | **36/36** | **36/36 / 36/36** |
| Large local model | 0/36 | **36/36** | **36/36 / 36/36** |
| Exact local outcomes | 0/12 | **12/12** | **12/12 / 12/12** |

Each matched allocation preserves every duration pair once and changes only
which support-geometry mode occupies that candidate slot. Both allocations
were specified before examining their results.

Complete missions execute three waypoint legs and two unloading/recovery
operations. The 12 physical cases cross two payload masses, three unloading
aircraft and two waypoint routes. Learned denominators pool three fit seeds
over those 12 conditions. Constrained completion additionally requires every
recovery to stay within **5 cm payload excursion and 0.6 N unloading-cable
tension**, with actual slack, restored support and all waypoint dwell periods.

| Predictor/controller on the Menagerie plant | Original support geometry | Expanded interface |
|---|---:|---:|
| Small frozen local model | 0/36 | **36/36** |
| Large frozen local model | 0/36 | **36/36** |
| True-option simulator | 0/12 | **12/12** |
| Study geometric feedback | 0/12 | **12/12** |
| Upstream Lee flight feedback with shared load allocation | 0/12 | **10/12** |

All **192 model/study-control missions and 24 external-Lee missions** deliver
their payload. The table measures the additional
load-transfer constraints. The expanded small/large selectors have median
latencies **1.49/1.55 ms**; exact-option selection has median latency 28.19 s.
Physical execution time and computation are reported separately. When no
candidate passes the validation-margin screen, the frozen model's predicted
constraint-violation score determines the action; this happens in 66/72 small
and 58/72 large expanded decisions. These are measured task results with
validation-derived margins.

All **96 paired pre-intervention trajectories** match bitwise. All 288 learned
decisions reproduce from their frozen predictions and argmins; all 48 oracle
decisions match actual simulated outcomes. Whole-mission nominal replay is
bitwise identical for all 192 runs. At 1 ms, all 192 pass the trajectory checks
and retain every delivery and constraint score. Maximum position/attitude
differences are **0.810 mm / 0.000693 rad**.

## External controller and CrazyMARL integration

The external controller calls the unchanged upstream `controller_lee.c` through
a small native wrapper. Each aircraft owns its complete `controllerLee_t`
state, including separate position and attitude integrals. The adapter supplies
the calibrated airframe mass/inertia, converts quaternion/gyro units, and sets
the scheduler to the shared 50 Hz control period. Upstream feedback gains are
retained. The payload force allocator, formation/recovery references and
bounded rotor mixer are shared across the flight-controller comparison.
This identifies the external component as Lee position/attitude feedback.

The full external-controller comparison executes **24 complete missions** with
unchanged upstream gains and no gain search. Both geometry arms deliver 12/12;
constrained completion changes from **0/12 to 10/12**. Maximum operational
payload excursion changes from 0.21794 m to 0.01477 m, maximum unloading-cable
tension from 2.11714 N to 0.70650 N, and mean actuator work from 0.49318 J to
0.37504 J. The two expanded-arm failures exceed the tension limit. All 12 paired
prefixes, including complete native-controller memory, match bitwise. There
are no collisions, numerical warnings or rotor-limit violations.

All 24 nominal external-controller replays match bitwise. At 1 ms, every task
score is unchanged and **23/24** trajectories pass the original 2 mm/0.02 rad
thresholds. The retained exception is `mass034_donor0_route0`, reconfigured,
with a 2.210 mm maximum aircraft-position difference. Additional 0.5 ms and
0.25 ms replays preserve its task score; those two finer grids differ by
0.495 mm. Their differences from the original 2 ms trajectory are 0.682 mm
and 0.802 mm respectively. The original 1 ms failure remains in the evidence.

The external-feedback runner accepts the two prescribed recovery-protocol
arms. Learned-model and true-option studies retain their training feedback
controller; the public mission entry point checks that separation.

CrazyMARL runs in a separate dependency environment. Its original three-UAV
environment, observation construction, dynamics, rewards, actor/critic and PPO
update bodies are preserved. A checked AST adapter omits an unused ONNX export
import. The pinned JAX CPU wheel requires a recorded, hash-guarded GNU_STACK
metadata repair on this host; no system loader setting is changed.

CPU and RTX 5070 GPU runs each execute **64 environment transitions, two PPO
updates and four optimizer steps**. The 48,457-parameter count includes actor
and critic. Actor weights change, physical states remain finite in the native
evaluation probes, and saved checkpoints reload exactly. GPU execution keeps
the CPU backend available for the upstream logging callback. Both final runs
use the same pinned adapter and separate CPU/CUDA environments.

The official repository's [checkpoint request](https://github.com/IMRCLab/CrazyMARL/issues/1)
and the local release/artifact audit identify the remaining policy-training
step. Executed small-budget IPPO runs are labeled `smoke_unconverged`; their
weights demonstrate training and checkpoint integration. The original
two-billion-step training configuration is retained for the full benchmark.
These CrazyMARL runs use its native position-tracking task. A formal IPPO
comparison on unloading/recovery requires matching the task and its observation
and action interfaces, then training to convergence under a declared budget.

## Reproduction

Use the existing UAV and Torch environments described in the project README.
Every scientific run requires a fresh output prefix; preserve the completed
artifacts under `data/uav_transport_external/`.

```bash
# Official-airframe physical audit and local data.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.physics_audit --plant menagerie_crazyflie_2 \
  --output data/uav_transport_external/repro_physics --workers 4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.collect --plant menagerie_crazyflie_2 \
  --out data/uav_transport_external/repro_local --workers 6

# Fit both sizes and all three seeds, then freeze them.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m uav_transport.train --data data/uav_transport_external/repro_local.npz \
  --out data/uav_transport_external/repro_models --workers 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.mission --plant menagerie_crazyflie_2 \
  --models data/uav_transport_external/repro_models_manifest.json \
  --out data/uav_transport_external/repro_missions --workers 8

# Refine the complete saved-action trajectories.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.mission_replay \
  --manifest data/uav_transport_external/repro_missions.json \
  --out data/uav_transport_external/repro_replay --workers 8

# Separate upstream CrazyMARL environment and real IPPO training-update smoke.
python3 scripts/setup_crazymarl.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/external/.venv-crazymarl/bin/python \
  -m uav_transport.crazymarl_baseline --out data/uav_transport_external/repro_ippo
```

The external flight-controller matrix and optional CUDA integration run use:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.external_experiment \
  --out data/uav_transport_external/repro_lee --workers 4
python3 scripts/setup_crazymarl.py --cuda \
  --venv data/external/.venv-crazymarl-gpu \
  --provenance data/uav_transport_external/crazymarl_gpu_runtime_compat.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/external/.venv-crazymarl-gpu/bin/python \
  -m uav_transport.crazymarl_baseline --platform gpu \
  --out data/uav_transport_external/repro_ippo_gpu
```

The combined source passes 274 tests in the Torch/VMAS environment and 108 in
the UAV environment, with 12 and three optional-dependency skips respectively.
The isolated CrazyMARL environment passes all six integration tests. The
historical archive independently passes its original 83 UAV tests.

Current scientific records use prefixes `physics_menagerie_v1`,
`local_menagerie_v1`, `models_menagerie_v1`, `selection_menagerie_v1`,
`mission_menagerie_v1`, `mission_menagerie_v1_analysis` and
`mission_menagerie_v1_replay`. Full raw trajectories and checkpoint provenance
remain in the ignored data tree; copied records and summaries are available in
[compact evidence package](../experiments/uav_transport/menagerie_v1/README.md).
