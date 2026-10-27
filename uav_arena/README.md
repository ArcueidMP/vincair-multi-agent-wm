# UAV world-model arena (working name)

A diagnostic benchmark for world models on a physically coupled multi-agent task. Three official
Crazyflie 2 airframes carry one cable-suspended payload from a given start to a given goal past
obstacles, while the decision maker sees delayed observations. The simulator is the ground truth:
the benchmark measures how close each model's predictions and decisions come to it, under which
conditions, and where they break.

The benchmark does not assume that learning is needed. A non-privileged physics model of the same
plant is a reference row, and learned models are reported against it as well as against the oracle.

Status (tag `arena-v0.2`):
- The environment core (`env.py`, `scenarios.py`, `geometry.py`, `lowlevel.py`) and the shared
  planning layer (`costs.py`, `planning.py`, `run.py`) are implemented and tested, with three
  reference rows: the oracle U, the physics reference, and physics with online payload-mass
  estimation.
- World models join under the participant contract, [PARTICIPANTS.md](PARTICIPANTS.md): the
  predictor interface, information rules, seed splits, data, budgets and model selection.
  `collect.py` collects training data and checks it against the committed reference manifests,
  `windows.py` turns it into training windows, and `openloop.py` scores predictions on it
  (evaluation layer 1).
- Rows, owners and deliverables are on the task board, [TASKS.md](TASKS.md);
  `participants/example/` is a working example participant.
- CrazyMARL runs on arena tasks (`crazymarl_task.py`, `crazymarl_env.py`, `crazymarl_train.py`,
  `crazymarl_policy.py`). Seeds 0–2 are trained and evaluated on `dev`.
- The world-model rows, evaluation layers 2–3 and the leaderboard are next (see Roadmap).
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
- **Sampling seeds:** the solver is reseeded at every episode start from (planner seed, layout
  seed), so an episode replays identically on its own. Every condition and participant on a layout
  draws the same samples (common random numbers). Before `arena-v0.2` one generator ran through
  all episodes, so results depended on episode order.
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
  - **Physics + mass estimation (`AdaptivePhysicsPredictor`):** the physics reference plus online
    payload-mass estimation by multiple models: a bank of 16 shadows, one per candidate mass on a
    2 g grid over the prior (10–40 g). Each new observation scores every shadow's prediction of it
    (positions, velocities scaled by 0.1 s and rotation columns scaled by 0.05 m). The estimate is
    the candidate with the least error over the last 0.5 s, and planning rolls out its shadow and
    mass. The first observation is scored against each candidate's settled hover, so the initial
    estimate comes from the hover tilt that the cable tension sets. With a mass on the grid it
    reproduces U's predictions (tested). Estimating a cable-suspended payload's mass online has a
    precedent in adaptive cable-payload control (Hajieghrary et al., IROS 2026); this estimator is a
    different, multiple-model one.
  - **Kinematic model (`KinematicPredictor`):** aircraft at their setpoints, payload under the
    formation. A differentiable test fixture for the Lagrangian path, not a participant.
- **Lagrangian limits:** the Lagrangian solver needs gradients through the prediction, so it
  applies only to differentiable predictors (world models), never to the simulator.

## Participants

| Row | Kind | Information | Status |
|---|---|---|---|
| U | MPPI on the true simulator | privileged: true state, payload mass, onboard states | implemented |
| Physics | MPPI on the nominal simulator | observation only | implemented |
| Physics + mass estimation | the physics row with online payload-mass estimation (multiple models, 2 g grid) | observation history | implemented |
| CrazyMARL | external, model-free: official IPPO trainer, decentralized 250 Hz motor commands, retrained on arena tasks | own state fresh; payload, teammates and obstacles delayed | seeds 0–2 evaluated on `dev` |
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
- **Evaluation (`crazymarl_policy.py`):** the mean action of each seed's final actor (the official
  training script also exports the final one) flies the arena plant through
  `ArenaEnv.step_motors`, mapping upstream rotor order onto the arena's and actions onto 0–0.15 N
  per rotor (the centre of the training randomization). Outcomes use the arena's events and
  success rule.

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -c "from uav_arena.crazymarl_task import build_task; build_task('data/uav_arena/crazymarl/task.npz')"
for s in 0 1 2; do
  PYTHONPATH=$PWD data/external/.venv-crazymarl-gpu/bin/python -m uav_arena.crazymarl_train \
    --checkout data/external/CrazyMARL --out data/uav_arena/crazymarl/ippo_s$s --num-envs 8192 --seed $s
  PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.crazymarl_policy --actor data/uav_arena/crazymarl/ippo_s$s.npz \
    --suite dev --seeds 0:5 --out data/uav_arena/results/crazymarl_s${s}_dev_s0-5.json
done
```

## Results on dev (seeds 0–4, all 8 conditions)

At `arena-v0.2`. The three planning rows use the shared MPPI (512 samples, 1.5 s horizon),
reseeded per episode; CrazyMARL flies the final actor of each training seed. Success count out of
5 (for CrazyMARL, one count per training seed) and mean time of the successes:

| Condition | U (oracle) | Physics (nominal 25 g) | Physics + mass estimation | CrazyMARL (seeds 0, 1, 2) |
|---|---|---|---|---|
| static, 0 ms | 5/5 (6.1 s) | 4/5 (7.2 s) | 5/5 (6.5 s) | 3, 2, 2 (4.8 s) |
| static, 50 ms | 5/5 (6.1 s) | 3/5 (7.1 s) | 5/5 (5.9 s) | 3, 2, 2 (4.9 s) |
| static, 100 ms | 5/5 (6.1 s) | 3/5 (6.2 s) | 5/5 (5.7 s) | 3, 2, 2 (5.0 s) |
| static, 200 ms | 5/5 (6.1 s) | 4/5 (6.4 s) | 5/5 (6.2 s) | 3, 2, 2 (5.4 s) |
| static+moving, 0 ms | 5/5 (6.8 s) | 3/5 (7.7 s) | 5/5 (6.1 s) | 3, 2, 2 (4.9 s) |
| static+moving, 50 ms | 5/5 (7.4 s) | 4/5 (9.3 s) | 5/5 (7.1 s) | 3, 2, 2 (5.0 s) |
| static+moving, 100 ms | 5/5 (7.6 s) | 4/5 (8.0 s) | 5/5 (6.8 s) | 3, 2, 2 (5.1 s) |
| static+moving, 200 ms | 5/5 (7.6 s) | 3/5 (6.1 s) | 5/5 (6.8 s) | 3, 2, 2 (5.5 s) |
| **Total** | **40/40** | **28/40** | **40/40** | **24, 16, 16 of 40** |

- **Physics + mass estimation matches U.** It scores 40/40, and no episode's outcome differs from
  U's. Paired success times differ by −0.3 s on average (median −0.5 s). Mean minimum clearance
  is 13.0 cm against U's 12.3 cm.
- **The estimator.** Its first estimate, from the hover tilt, is within 0.9 g of the true mass
  (mean 0.5 g). In 9 of 40 episodes it later switches between the two grid neighbours of the true
  mass, never further: the largest error is 1.6 g.
- **What this means.** The gap between U and the nominal physics reference is the hidden payload
  mass alone, and a classical estimator closes it. On `dev`, no closed-loop headroom is left above
  physics + mass estimation. A learned model can match it, or show where and why it falls short;
  the evaluation layers measure that.
- **The nominal physics reference** fails 12 times, 11 of them on the three heavy payloads (seeds
  1, 2 and 4: 38.4, 35.5 and 35.9 g against the nominal 25 g). Most are aircraft-separation
  failures (9 of 12): the heavier load pulls the cables inward harder than the nominal model
  predicts. The twelfth is a timeout on seed 0 (27.3 g). Its mean minimum clearance is 10.4 cm.
- **CrazyMARL** has one outcome per layout and training seed, the same in all 8 conditions. Seed 0
  succeeds on layouts 0, 1 and 3; seeds 1 and 2 on layouts 1 and 3 only. Every failure is a
  pillar strike within 2.7 s, in the moving-obstacle conditions too (the movers stay at least
  33 cm away). Latency changes only when the strike happens: at 200 ms it comes 0.06–0.47 s later
  than at 0 ms. Paired with U on the same episodes, its success times differ from U's by −1.5,
  +0.1 and −0.4 s on average for seeds 0–2 (negative is faster). Seed 0's lead comes from
  layout 0, where U needs 7.4–11.9 s and seed 0 5.3–5.9 s.
- **CrazyMARL training.** All three seeds train on the same task file, trainer and pinned upstream
  commit. Their returns level off after about 40% of the 2×10⁹ steps, then swing between about
  2,800 and 4,600 (averaged over 50 updates) and end below their peaks; the final logged returns
  are 3,676, 3,540 and 3,424. No intermediate checkpoints are kept, so the final actor is the one
  evaluated, as in the official training script. Seed 0 was evaluated before `arena-v0.2`; rerun
  at the tag, its 40 records are identical.
- **Before `arena-v0.2`** (one solver generator across episodes), U scored 39/40 and physics
  28/40. U's one failure, a mover collision at 200 ms, did not recur.

## Open-loop references (layer 1, validation set)

`openloop.py` on the contract's validation set (100 `explore-v1` episodes, seeds
299000–299099, every decision scored). Mean position error over the 15-decision (1.5 s) horizon,
payload / aircraft:

| Latency | Physics (nominal 25 g) | Physics + mass estimation |
|---|---|---|
| 0 ms | 2.81 / 5.61 cm | 0.15 / 0.33 cm |
| 50 ms | 2.88 / 5.72 cm | 0.15 / 0.34 cm |
| 100 ms | 2.96 / 5.84 cm | 0.16 / 0.34 cm |
| 200 ms | 3.08 / 6.01 cm | 0.16 / 0.35 cm |
| **Selection score** | **2.93 cm** | **0.16 cm** |

The selection score is the contract's model-selection metric: the payload error averaged over the
horizon and the four latencies. U has no row: it needs the simulator's true state, which recorded
data does not hold. With the true mass the physics reference scores below 10⁻⁶ m (tested).

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
2. CrazyMARL on arena tasks: ~~training environment, motor-level arena interface, seeds 0–2 on
   `dev`~~ (done); evaluation on `test`.
3. ~~Physics with online payload-mass estimation~~ (done).
4. ~~Participant contract, data collector with reference manifests, training windows, layer-1
   open-loop metric~~ (done, `arena-v0.2`);
   entity-token JEPA variants, the autoregressive model and PETS.
5. Layer 2–3 diagnostics harness.
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
- The onboard controller is the pinned upstream C source, compiled on first use, so the arena
  needs a C compiler (`cc`) and a POSIX system. It has been run on Linux x86-64 (Ubuntu 26.04,
  gcc 15.2); on Windows, use WSL2.

## Reproduce

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m pytest -q tests/test_uav_arena.py tests/test_uav_arena_planning.py \
  tests/test_uav_arena_collect.py tests/test_uav_arena_windows.py tests/test_uav_arena_openloop.py \
  tests/test_uav_arena_example.py tests/test_uav_arena_replay.py tests/test_uav_arena_crazymarl.py
for p in oracle physics adaptive; do
  PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run --predictor $p --suite dev --seeds 0:5 \
    --workers 20 --out data/uav_arena/results/${p}_mppi_dev_s0-5_v3.json
done
```

The `dev` table above ran the three rows in parallel with 7 workers each. The outcome does not
depend on the worker count: two of its episodes, rerun alone with 3 workers, reproduced their
records exactly.

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.collect --seeds 200000:202000 \
  --out data/uav_arena/datasets/explore_v1
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.collect --seeds 299000:299100 \
  --out data/uav_arena/datasets/explore_v1_val
for d in explore_v1 explore_v1_val; do
  PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.collect --verify data/uav_arena/datasets/$d
done
for p in physics adaptive; do
  PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop --predictor $p \
    --data data/uav_arena/datasets/explore_v1_val --out data/uav_arena/results/${p}_openloop_val_v1.json
done
```

Both datasets take about a minute on 24 cores (1.6 GB and 80 MB), and `--verify` must report
every episode as a match with the reference manifests. The batched controller compiles once into
`data/uav_arena/native_build/` (override with `UAV_ARENA_BUILD_CACHE`).
