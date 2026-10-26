# UAV world-model arena (working name)

A benchmark that asks which learned world models can be trusted to plan a physically coupled
multi-agent task, under which conditions, and where they break. Three official Crazyflie 2
airframes carry one cable-suspended payload from a given start to a given goal past obstacles,
while the decision maker sees delayed observations.

Status (2026-10-27):
- The environment core (`env.py`, `scenarios.py`, `geometry.py`, `lowlevel.py`) is implemented
  and tested.
- The shared planning layer (`costs.py`, `planning.py`, `run.py`) is implemented and tested. It
  uses stable-worldmodel 0.1.1 solvers, the oracle U and a kinematic reference.
- World-model participants, the layered evaluation and the leaderboard are next (see Roadmap).

## Task and conditions

- **Team and plant:** `uav_transport.menagerie` scene (pinned Menagerie Crazyflie 2 airframes,
  0.6 m tension-only cables, 30 g central payload, first-order motors, 2 ms physics).
- **Episode:** hover formation at the start (payload 1.0–1.4 m high), goal 3.2 m away in x,
  20 s limit. Success: payload within 0.15 m of the goal and slower than 0.2 m/s for 5
  consecutive decisions (0.5 s).
- **Factor 1, obstacles:** `static` = 4 pillars (vertical cylinders, radius 0.10–0.18 m) placed
  between start and goal, always leaving a corridor for the team (planar grid search with team
  radius 0.41 m). `static+moving` = the same pillars plus 2 spheres (radius configurable,
  default 0.15 m) moving back and forth across the corridor at 0.25–0.45 m/s on seeded segments.
- **Factor 2, latency:** the decision maker sees the state from d ∈ {0, 50, 100, 200} ms ago.
  Decisions stay at 10 Hz; onboard feedback is unaffected.
- **Pairing:** for a given seed the layout is identical in all 8 conditions; moving obstacles
  are identical across latencies. Suites: `dev` seeds 0–19, `test` seeds 1000–1049
  (`scenarios.suite`).

## Clocks and information

| Stream | Rate | State |
|---|---|---|
| Physics | 500 Hz (2 ms) | true |
| Onboard feedback: upstream `controller_lee.c` (IMRCLab firmware, pinned commit) per aircraft, bounded mixer, nominal 25 g cable-load feedforward | 500 Hz | true, fresh |
| Decision maker (planner or policy) | 10 Hz | recorded `latency` ago |

- **Action:** one world-frame position setpoint per aircraft, held for 100 ms. The onboard
  geofence clips each setpoint to 0.5 m from the aircraft and to at least 0.3 m above the floor.
- **Observation:** per aircraft position, velocity, two rotation columns and body rates; payload
  position and velocity; obstacles' current (equally delayed) positions, velocities and sizes; the
  goal; and the last three setpoints the decision maker sent, with their times.
- **Obstacle forecasts:** methods that need obstacle forecasts all use the same constant-velocity
  extrapolation from the observed state (to be provided with the planner). No method sees future
  obstacle paths.
- **Not observed:** payload mass and cable tensions. The oracle U (planned) sees the true state.
- Before t = 0 the team hovers for 5 s under onboard feedback so every episode starts settled.

## Events and scoring (implemented)

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
        obs, reward, done, info = env.step(setpoints)   # (3, 3) world-frame positions
    snapshot = env.capture()                            # exact restore: env.restore(snapshot)
```

`capture(log_tail=...)` keeps only the recent log, which is enough for planning forks. Restores
replay bitwise.

## Planning (implemented)

Every participant plans with the same solver, action parameterization and cost; only the
predictor differs.

- **Solvers:** the [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) 0.1.1
  `MPPISolver` (512 samples, 15 decisions = 1.5 s, one iteration, warm start) and
  `LagrangianSolver`. Costs implement swm's `Costable` protocol (`get_cost`, plus
  `get_constraints` for the Lagrangian).
- **Action:** per-aircraft setpoint increments of at most 0.12 m per decision and axis, added to
  the last commanded setpoints.
- **Cost (`costs.py`):** payload distance to the goal (mean over the horizon plus terminal),
  payload speed near the goal, and increment size.
- **Constraints:** worst-step clearance to pillars and to movers (aircraft, payload and cables,
  0.10 m margin), aircraft separation of at least 0.18 m, and payload height of at least 0.30 m.
  - MPPI sees the constraints as quadratic penalties.
  - The Lagrangian sees them as constraints. Because swm 0.1.1 returns its lowest-cost sample
    without checking constraints, its cost keeps the penalties as well.
- **Obstacle forecasts:** every method extrapolates the observed (equally delayed) obstacle state
  at constant velocity.
- **Predictors:**
  - **U, the oracle (`TrueSimPredictor`):** forks the true simulator at the current state in a
    process pool. Bridging the delay with an exact model is equivalent to this.
  - **Kinematic reference (`KinematicPredictor`):** aircraft reach their setpoints instantly and
    the payload hangs under the formation. It is differentiable, so it is the test bed for the
    Lagrangian path.
- **Lagrangian limits:** the Lagrangian solver needs gradients through the prediction. It
  therefore applies only to differentiable predictors (world models), never to the simulator.

## Evaluation layers (planned)

1. **Open-loop prediction:** multi-step error from decision states, including bridging the latency.
2. **Cross-agent counterfactuals:** change one aircraft's setpoints and compare predicted and true
   changes for the payload and the other two aircraft.
3. **Decision level:** ranking of candidate plans and elite regret against the true simulator
   under the arena cost.
4. **Closed loop:** success, failure types, time to goal and clearance per condition.

The headline analysis is whether these layers rank world models consistently, and how that
changes with latency and moving obstacles.

## Participants (planned)

- **U:** the same MPPI on the true simulator (snapshot rollouts); the oracle reference.
- **Entity-token JEPA world models** (`vmas_pusht/model.py` adapted to 3 aircraft + payload;
  `full`, `no_cross`, `flat`) and a non-JEPA autoregressive state model, all inside one shared MPPI.
- **TD-MPC2 and DreamerV3**, trained on the arena reward and evaluated as published, plus state
  probes for layers 1–3.

## Roadmap

1. ~~Shared MPPI with the arena cost, the U oracle~~ (done); solvability checks on `dev`.
2. Data collection; entity-token JEPA, `no_cross`/`flat` and the autoregressive model.
3. Layer 1–3 diagnostics harness.
4. TD-MPC2 and DreamerV3 adapters (needs a CUDA PyTorch venv and a JAX venv).
5. Closed-loop runs on `test`, the leaderboard tables and the layer-consistency analysis.

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
```

The batched controller compiles once into `data/uav_arena/native_build/`
(override with `UAV_ARENA_BUILD_CACHE`).
