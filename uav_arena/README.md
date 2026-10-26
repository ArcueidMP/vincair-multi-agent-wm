# UAV world-model arena (working name)

A diagnostic benchmark for world models on a physically coupled multi-agent task. Three official
Crazyflie 2 airframes carry one cable-suspended payload from a given start to a given goal past
obstacles, while the decision maker sees delayed observations. The simulator is the ground truth:
the benchmark measures how close each model's predictions and decisions come to it, under which
conditions, and where they break.

The benchmark does not assume that learning is needed. A non-privileged physics model of the same
plant is a reference row, and learned models are reported against it as well as against the oracle.

Status:
- The environment core (`env.py`, `scenarios.py`, `geometry.py`, `lowlevel.py`) and the shared
  planning layer (`costs.py`, `planning.py`, `run.py`) are implemented and tested, with two
  reference rows: the oracle U and the physics reference.
- CrazyMARL runs on arena tasks (`crazymarl_task.py`, `crazymarl_env.py`, `crazymarl_train.py`,
  `crazymarl_policy.py`). Seed 0 is trained and evaluated on `dev`; seeds 1 and 2 are pending.
- The world-model participants, the layered evaluation and the leaderboard are next (see Roadmap).
- **Known issue (fix before reporting new results):** `run.py` reuses one planner for every episode,
  and swm's MPPI keeps one random generator, so an episode's samples depend on the episodes run
  before it. The `dev` numbers below are valid as one run, but a single episode does not replay
  identically on its own. Reseeding the solver at every episode start fixes this; U and physics
  then need a rerun.
- Related work, grouped by relation to the arena, with BibTeX:
  [`docs/uav-arena-related-work.md`](../docs/uav-arena-related-work.md).
- The earlier experiment card v2.0 is archived in
  [`archives/uav_transport/experiment_card_v2/`](../archives/uav_transport/experiment_card_v2/README.md)
  and does not apply to this benchmark.

## Task and conditions

- **Team and plant:** `uav_transport.menagerie` scene (pinned Menagerie Crazyflie 2 airframes,
  0.6 m tension-only cables, central payload, first-order motors, 2 ms physics).
- **Payload mass:** drawn per seed from 10–40 g and hidden from every method except the oracle.
  The onboard cable-load feedforward assumes 25 g; the onboard integral absorbs the difference.
- **Episode:** hover formation at the start (payload 1.0–1.4 m high), goal 3.2 m away in x,
  20 s limit. Success: payload within 0.15 m of the goal and slower than 0.2 m/s for 5
  consecutive decisions (0.5 s).
- **Factor 1, obstacles:** `static` = 4 pillars (vertical cylinders, radius 0.10–0.18 m) placed
  between start and goal, always leaving a corridor for the team (planar grid search with team
  radius 0.41 m). `static+moving` = the same pillars plus 2 spheres (radius configurable,
  default 0.15 m) moving back and forth across the corridor at 0.25–0.45 m/s on seeded segments.
- **Factor 2, latency:** the decision maker sees the state from d ∈ {0, 50, 100, 200} ms ago.
  Decisions stay at 10 Hz; onboard feedback is unaffected.
- **Pairing:** for a given seed the layout and the payload mass are identical in all 8
  conditions; moving obstacles are identical across latencies. Suites: `dev` seeds 0–19,
  `test` seeds 1000–1049 (`scenarios.suite`).

## Clocks and information

| Stream | Rate | State |
|---|---|---|
| Physics | 500 Hz (2 ms) | true |
| Onboard feedback: upstream `controller_lee.c` (IMRCLab firmware, pinned commit) per aircraft, the study's bounded mixer, nominal 25 g cable-load feedforward | 500 Hz | true, fresh |
| Decision maker (planner or policy) | 10 Hz | recorded `latency` ago |

- **Action:** one world-frame position setpoint per aircraft and decision. Onboard, each
  aircraft's setpoint moves linearly from the previous one to the new one over the 100 ms period,
  with the matching velocity feedforward, as the firmware's trajectory commander does. The onboard
  geofence clips each new setpoint to 0.5 m from the aircraft and to at least 0.3 m above the floor.
  - Holding position-only setpoints made the aircraft trail them by about 0.45 m in transit and
    wound up the Lee position integral. Ignoring that hidden integral then cost 30–37 cm of payload
    prediction error at 1.5 s, an effect unrelated to coupling. With segments the trail is about
    0.1 m and the same test costs about 8 cm, mostly the payload-mass compensation.
- **Observation:** per aircraft position, velocity, two rotation columns and body rates; payload
  position and velocity; obstacles' current (equally delayed) positions, velocities and sizes; the
  goal; and the last four setpoints in force with their times. The first is the hover setpoint
  held before t = 0; four are enough to replay a 200 ms delay.
- **Obstacle forecasts:** every method that needs them uses the same constant-velocity
  extrapolation from the observed state. No method sees future obstacle paths.
- **Not observed:** payload mass, cable tensions, and the onboard motor and controller states.
  Only the oracle U uses them.
- Before t = 0 the team hovers for 5 s under onboard feedback so every episode starts settled.

## Events and scoring

Checked every 10 ms on the true state:
- `collision_obstacle`: any aircraft (radius 0.06 m), the payload or any cable segment enters an
  obstacle. Obstacles are non-physical volumes; MuJoCo tendons cannot collide, so cable crossings
  are computed as segment distances in `geometry.py`.
- `collision_team`: two aircraft closer than 0.1 m.
- `crash`: an aircraft below 0.05 m, or the payload touching the floor.
- `out_of_bounds`: leaving x ∈ [−3, 3], y ∈ [−2.5, 2.5], z ∈ [0, 3].
- `nonfinite`.

Every step returns a reward for methods that learn from one: 10 × (decrease in payload-to-goal
distance) − 0.01, plus +10 on success and −10 on any failure. Episode records keep the outcome,
time, minimum clearance per team part, and a 10 ms log of the true state.

## Interface

```python
from uav_arena.scenarios import suite
from uav_arena.env import ArenaEnv

for scenario in suite("dev"):
    env = ArenaEnv(scenario)
    obs = env.reset()
    while not env.done:
        obs, reward, done, info = env.step(setpoints)   # (3, 3) world-frame segment end points
    snapshot = env.capture()                            # exact restore: env.restore(snapshot)
```

`capture(log_tail=...)` keeps only the recent log, which is enough for planning forks. Restores
replay bitwise.

## Planning

Every model-based participant plans with the same solver, action parameterization and cost; only
the predictor differs.

- **Solvers:** the [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) 0.1.1
  `MPPISolver` (512 samples, 15 decisions = 1.5 s, one iteration, warm start) and
  `LagrangianSolver`. Costs implement swm's `Costable` protocol (`get_cost`, plus
  `get_constraints` for the Lagrangian).
- **Action:** per-aircraft setpoint increments of at most 0.12 m per decision and axis, added to
  the last commanded setpoints.
- **Cost (`costs.py`):** payload distance to the goal (mean over the horizon plus terminal),
  payload speed near the goal, and increment size.
- **Constraints:** worst-step clearance to pillars and to movers (aircraft, payload and cables,
  0.10 m margin), aircraft separation of at least 0.18 m, payload height of at least 0.30 m, and
  every aircraft at least 0.30 m above the payload (0.55 m in hover). Without the last one, even U
  sometimes planned aggressive segments that saturated an aircraft's rotors under a swinging
  payload: the aircraft sank below the payload and crashed (3 of 40 dev episodes).
  MPPI sees them as quadratic penalties. The Lagrangian sees them as constraints; because swm
  0.1.1 returns its lowest-cost sample without checking constraints, its cost keeps the penalties.
- **Predictors:**
  - **U, the oracle (`TrueSimPredictor`):** forks the true simulator at the current true state in
    a process pool. It is privileged: it also knows the payload mass and the onboard motor and
    controller states.
  - **Physics reference (`PhysicsPredictor`):** the same simulator with the nominal 25 g payload,
    started from the delayed observation. A shadow copy of the known onboard flight stack carries
    the unobserved motor and controller states between decisions, and the delay is bridged by
    replaying the setpoint segments already sent (a Smith predictor). With the true mass it
    reproduces U's predictions (tested).
  - **Kinematic model (`KinematicPredictor`):** aircraft at their setpoints, payload under the
    formation. A differentiable test fixture for the Lagrangian path, not a participant.
- **Lagrangian limits:** the Lagrangian solver needs gradients through the prediction, so it
  applies only to differentiable predictors (world models), never to the simulator.

## Participants

| Row | Kind | Information | Status |
|---|---|---|---|
| U | MPPI on the true simulator | privileged: true state, payload mass, onboard states | implemented |
| Physics | MPPI on the nominal simulator | observation only | implemented |
| Physics + mass estimation | the physics row with online payload-mass estimation from the observed history | observation history | next |
| CrazyMARL | external, model-free: official IPPO trainer, decentralized 250 Hz motor commands, retrained on arena tasks | own state fresh; payload, teammates and obstacles delayed | training |
| Entity-token JEPA (`full`, `no_cross`, `flat`), autoregressive state model, PETS ensemble | world models in the shared MPPI | observation history | planned |
| TD-MPC2 (official, trained online), DreamerV3 | external model-based RL, evaluated as published, plus state probes for layers 1–3 | observation | planned |

## CrazyMARL on arena tasks

CrazyMARL's published setting has no obstacles and no observation delay, and its policy flies the
rotors directly, so it is retrained on arena tasks rather than run as released (it releases no
checkpoint either). The method stays as published; only the task changes.

- **Kept:** the pinned official IPPO trainer, networks and hyperparameters (actor 64-64-64 tanh,
  critic 128-128-128), parameter sharing, decentralized local observations, per-rotor thrust
  commands at 250 Hz, and the upstream motor model, disturbance randomization, safety checks and
  reward terms (`crazymarl_env.ArenaMultiQuadEnv` subclasses the official `MultiQuadEnv`).
- **Arena task (`crazymarl_task.py`):** the arena airframe mass and inertia, 0.6 m cables, the
  nominal 25 g payload, the settled-hover start and goal transit with a progress reward
  (0.25 × approach speed, clipped at ±2 m/s), obstacle features (pillars relative to the payload,
  nearest first; movers' relative position, velocity and size) with collisions ending the episode,
  and the arena's information delays: each aircraft's own state is fresh, while the payload,
  teammates and obstacles arrive 0, 48, 100 or 200 ms late (latency is an input). Per-agent
  observations grow from 34 to 61 entries.
- **Training layouts:** 2,048 seeds × 2 obstacle settings from the arena generator, disjoint from
  `dev` and `test`; payload mass is not randomized in training, as upstream.
- **Budget:** 2×10⁹ environment steps per seed. The official 16,384 parallel environments do not
  fit the 8 GB GPU, so 8,192 are used (about 270k steps/s, 2.1 h per seed).
- **Evaluation (`crazymarl_policy.py`):** the mean action of the exported actor flies the arena
  plant through `ArenaEnv.step_motors`, mapping upstream rotor order onto the arena's and actions
  onto 0–0.15 N per rotor (the centre of the training randomization). Outcomes use the arena's
  events and success rule.

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -c "from uav_arena.crazymarl_task import build_task; build_task('data/uav_arena/crazymarl/task.npz')"
PYTHONPATH=$PWD data/external/.venv-crazymarl-gpu/bin/python -m uav_arena.crazymarl_train \
  --checkout data/external/CrazyMARL --out data/uav_arena/crazymarl/ippo_s0 --num-envs 8192 --seed 0
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.crazymarl_policy \
  --actor data/uav_arena/crazymarl/ippo_s0.npz --suite dev --seeds 0:5 --out data/uav_arena/results/crazymarl_dev_s0-5.json
```

## Results on dev (seeds 0–4, all 8 conditions)

Shared MPPI (512 samples, 1.5 s horizon). Success count out of 5, mean time of the successes:

| Condition | U (oracle) | Physics (nominal 25 g) | CrazyMARL (seed 0) |
|---|---|---|---|
| static, 0 ms | 5/5 (6.8 s) | 5/5 (8.4 s) | 3/5 (4.7 s) |
| static, 50 ms | 5/5 (8.3 s) | 4/5 (7.0 s) | 3/5 (4.8 s) |
| static, 100 ms | 5/5 (6.1 s) | 4/5 (6.0 s) | 3/5 (4.9 s) |
| static, 200 ms | 5/5 (6.1 s) | 4/5 (6.6 s) | 3/5 (5.2 s) |
| static+moving, 0 ms | 5/5 (6.8 s) | 4/5 (5.9 s) | 3/5 (4.8 s) |
| static+moving, 50 ms | 5/5 (7.1 s) | 3/5 (7.4 s) | 3/5 (4.9 s) |
| static+moving, 100 ms | 5/5 (7.7 s) | 2/5 (5.8 s) | 3/5 (4.9 s) |
| static+moving, 200 ms | 4/5 (6.7 s) | 2/5 (9.8 s) | 3/5 (5.2 s) |
| **Total** | **39/40** | **28/40** | **24/40** |

- U's one failure is a mover collision at 200 ms: the constant-velocity forecast misses a turn.
- Every physics failure is on one of the three heavy payloads (seeds 1, 2 and 4: 38.4, 35.5 and
  35.9 g against the nominal 25 g). Most are aircraft-separation failures (9 of 12): the heavier
  load pulls the cables inward harder than the nominal model predicts. Failures grow with latency
  and moving obstacles. On the light seeds (0 and 3: 27.3 and 23.1 g) it never fails.
- With the true mass the physics reference reproduces U's predictions, so this gap is the hidden
  payload mass. It is a parametric mismatch: a physics model with online mass estimation is the
  reference a learned model must beat before any claim that learning helps.
- CrazyMARL (one training seed so far; its return plateaued from about 40% of the 2×10⁹ steps)
  succeeds on layouts 0, 1 and 3 in every condition, faster than U, and hits a pillar within 2 s on
  layouts 2 and 4 in every condition. Latency and moving obstacles do not change its outcomes.

## Evaluation layers (planned)

1. **Open-loop prediction:** multi-step error from decision states, including bridging the latency.
2. **Cross-agent counterfactuals:** change one aircraft's setpoints and compare predicted and true
   changes for the payload and the other two aircraft.
3. **Decision level:** ranking of candidate plans and elite regret against the true simulator
   under the arena cost.
4. **Closed loop:** success, failure types, time to goal and clearance per condition.

The headline analysis is whether the cross-agent layer explains closed-loop differences that the
other layers miss, and how that changes with latency and moving obstacles.

## Roadmap

1. ~~Environment, shared MPPI, U and physics references, setpoint segments, hidden payload mass~~
   (done).
2. CrazyMARL on arena tasks: ~~training environment, motor-level arena interface~~ (done);
   training three seeds and evaluation on `dev` and `test`.
3. Physics with online payload-mass estimation: the adaptive reference for the hidden mass.
4. Data collection; entity-token JEPA variants, the autoregressive model and PETS.
5. Layer 1–3 diagnostics harness.
6. TD-MPC2 and DreamerV3 adapters (a Gymnasium wrapper; a JAX venv for DreamerV3).
7. Closed-loop runs on `test`, the leaderboard tables and the layer-consistency analysis.

## Replay and video

`replay.py` records one episode (MuJoCo positions and movers every 10 ms, plus the scenario) and
replays it kinematically, in a MuJoCo window or as an MP4. The payload's path is drawn as a trail
and cables are drawn thicker than they are simulated.

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.replay record --participant oracle \
  --seed 0 --condition static+moving/100ms --out data/uav_arena/replays/oracle_s0_moving100.npz
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.replay view data/uav_arena/replays/oracle_s0_moving100.npz
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.replay video data/uav_arena/replays/oracle_s0_moving100.npz \
  --out data/uav_arena/videos/oracle_s0_moving100.mp4
```

- Participants: `oracle`, `physics`, `crazymarl` (with `--actor`), and `transfer`, an open-loop
  straight minimum-jerk line for quick looks.
- Viewer keys: P pauses, R restarts; the mouse orbits and zooms. The replay loops.
- Videos are H.264 through the system `ffmpeg` (OpenCV's MPEG-4 writer otherwise), 1280×720 at
  50 fps by default, with the participant, condition, payload mass, time and outcome overlaid.
  `--speed` changes the playback rate.

## Environment

The planning layer needs PyTorch and stable-worldmodel:

```bash
uv venv --python 3.10 data/uav_arena/.venv
uv pip install --python data/uav_arena/.venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision
uv pip install --python data/uav_arena/.venv/bin/python stable-worldmodel==0.1.1 mujoco==3.3.0 numpy==2.2.6 scipy==1.15.3 pytest imageio opencv-python-headless
```

- `import stable_worldmodel` 0.1.1 imports every submodule, and its base install misses
  `imageio` and `cv2`, so both are added above.
- Do not install the `[env]` extra: it pulls in `dm_control`, which can replace MuJoCo 3.3.0.
- The environment-core tests run without PyTorch, in `data/uav_transport/.venv`.

## Reproduce

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m pytest -q tests/test_uav_arena.py tests/test_uav_arena_planning.py
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run --predictor oracle --suite dev --seeds 0:5 \
  --workers 20 --out data/uav_arena/results/oracle_mppi_dev_s0-5.json
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run --predictor physics --suite dev --seeds 0:5 \
  --workers 20 --out data/uav_arena/results/physics_mppi_dev_s0-5.json
```

The batched controller compiles once into `data/uav_arena/native_build/`
(override with `UAV_ARENA_BUILD_CACHE`).
