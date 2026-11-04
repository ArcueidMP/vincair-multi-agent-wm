# UAV world-model arena (working name)

A diagnostic benchmark for world models on a physically coupled multi-agent task. Three official
Crazyflie 2 airframes carry one cable-suspended payload from a given start to a given goal past
obstacles, while the decision maker sees delayed observations. The simulator is the ground truth:
the benchmark measures how close each model's predictions and decisions come to it, under which
conditions, and where they break.

The benchmark does not assume that learning is needed. A non-privileged physics model of the same
plant is a reference row, and learned models are reported against it as well as against the oracle.

Status (`arena-v0.3`, tagged on 2026-10-28, before the first `test` run):
- The environment core (`env.py`, `scenarios.py`, `geometry.py`, `lowlevel.py`) and the shared
  planning layer (`costs.py`, `planning.py`, `run.py`) are implemented and tested, with three
  reference rows: the oracle U, the physics reference, and physics with online payload-mass
  estimation. These files are unchanged since `arena-v0.2`.
- World models join under the participant contract, [PARTICIPANTS.md](PARTICIPANTS.md): the
  predictor interface, the rules for stochastic predictors, information rules, seed splits, the two
  data budgets and model selection. `collect.py` collects training data and checks it against the
  committed reference manifests, `windows.py` turns it into training windows, and `openloop.py`
  scores predictions on it (evaluation layer 1). The standard data are `explore-v2`, which replaced
  `explore-v1` in `arena-v0.3` because U's planner commands actions `explore-v1` never flew;
  `coverage.py` compares a dataset with U's flights.
- The analysis plan, [ANALYSIS.md](ANALYSIS.md), is frozen with the tag, before any `test` run:
  the headline test, its decision rule and the secondary analyses. `analysis.py` computes them.
- The evaluation harness for layers 1–3, [HARNESS.md](HARNESS.md), is built (`record.py`,
  `harness.py`), tested and validated on `dev`. It is maintained outside the frozen contract and
  scores every model on decision states recorded from U's closed-loop runs.
- Rows, owners and deliverables are on the task board, [TASKS.md](TASKS.md);
  `participants/example/` is a working example participant.
- CrazyMARL runs on arena tasks (`crazymarl_task.py`, `crazymarl_env.py`, `crazymarl_train.py`,
  `crazymarl_policy.py`). Seeds 0–2 are trained, evaluated on `dev` and run on `test`.
- Methods that train online, TD-MPC2 and DreamerV3, get a Gymnasium wrapper (`gym_env.py`,
  `gym_eval.py`; Gymnasium wrapper, below) and adapters to their pinned official code, with a venv
  each (TD-MPC2 and DreamerV3, below). Their training runs come next: 2M environment steps and
  seeds 42, 67 and 2027 for each method.
- The reference rows and the example participant ran on `test` on 2026-10-28 and 29, U's through
  the recorder, which also recorded the questions, the CrazyMARL rows on 2026-11-01, W-ar's and
  PETS's six models each on 2026-11-02, and JEPA's 18 on 2026-11-03 and 04 (Results on test,
  below).
- The pre-registered test ran on 2026-11-04 over the 30 primary models and is supported:
  ρ(L2, J | L1) = −0.390 (−0.602 to −0.274). Next: the TD-MPC2 and DreamerV3 rows (see Roadmap).
- Related work, grouped by relation to the arena, with BibTeX:
  [`docs/uav-arena-related-work.md`](../docs/uav-arena-related-work.md). Its section 8 covers the
  evaluation design.

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
| U | MPPI on the true simulator | privileged: true state, payload mass, onboard states | implemented; run on `test` |
| Physics | MPPI on the nominal simulator | observation only | implemented; run on `test` |
| Physics + mass estimation | the physics row with online payload-mass estimation (multiple models, 2 g grid) | observation history | implemented; run on `test` |
| CrazyMARL | external, model-free: official IPPO trainer, decentralized 250 Hz motor commands, retrained on arena tasks | own state fresh; payload, teammates and obstacles delayed | seeds 0–2 run on `test` |
| Autoregressive state model (W-ar) | world model in the shared MPPI | observation history | six formal models run on `test` |
| PETS ensemble | world model in the shared MPPI | observation history | six formal models run on `test` |
| Entity-token JEPA (`full`, `no_cross`, `flat`) | world models in the shared MPPI | observation history | 18 formal models run on `test` |
| TD-MPC2 (official, trained online), DreamerV3 | external model-based RL, evaluated as published, plus state probes for layers 1–3 | observation | adapters built and smoke-tested; training planned (2M steps, 3 seeds) |

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

## Gymnasium wrapper

`gym_env.ArenaGymEnv` serves the methods that train online in the environment, TD-MPC2 and
DreamerV3. Each of its steps is an `ArenaEnv` step, bit for bit (tested). Settings, decided on
2026-10-30:
- **Episodes.** Each reset draws a training layout from seeds 200000–298999 (PARTICIPANTS.md,
  Section 4) and one of the 8 conditions uniformly. `reset(options=dict(seed=..., condition=...))`
  sets a `dev` or `test` episode. An event or a success ends an episode (`terminated`); the 20 s time
  limit truncates it.
- **Action** (9,) in [−1, 1]: each aircraft's setpoint increment per axis in units of 0.12 m, added to
  the setpoint in force, as the shared MPPI parameterizes its candidates.
- **Observation** (118,): the decision maker's current observation, flattened by `gym_env.flatten`:
  payload and aircraft states seen `latency` late, the goal, pillars, movers, the setpoints in force,
  the latency, and the share of the time limit left. It never holds the payload mass or the undelayed
  state, and no history beyond the setpoints in force: the official code takes the current state.
- **Reward:** `ArenaEnv`'s (Events and scoring), which also charges −10 at the time limit.
- **Evaluation:** `gym_eval.py` flies a policy, an object with `reset(seed)` and `act(observation)`,
  on a suite and writes `run.py`'s records.
- **Versions.** The physics reproduces the reference bit for bit only with MuJoCo 3.3.0 and NumPy 1.26
  or later; Python 3.10 and 3.11 both do. With NumPy 1.24.4, TD-MPC2's pin, the settled hover already
  differs by 10⁻¹⁰ m, and `collect.py --verify` reports every regenerated episode as different.
  Check every venv and machine before training or evaluating; it must report `"differ": 0`:

```bash
PYTHONPATH=. python -m uav_arena.collect --seeds 200000:200010 --out /tmp/arena_check
PYTHONPATH=. python -m uav_arena.collect --verify /tmp/arena_check
```

## TD-MPC2 and DreamerV3

Both run as published: the official code at a pinned commit, never modified (each launcher checks
the commit and that no tracked file changed), with its default settings, trained online through the
Gymnasium wrapper and evaluated with its own planner or actor through `gym_eval.py`. Each registers
the arena task the way the code registers its own tasks (`upstream.py` holds the shared checks).
- **TD-MPC2** (`tdmpc2_train.py`, `tdmpc2_policy.py`; nicklashansen/tdmpc2 at 332f19e). Upstream's
  `train.py` with `config.yaml`'s defaults (5M parameters; planning with 512 samples over 3 steps)
  on the task `uav-arena`, which joins `envs.make_env` as `envs/mujoco.py` adds Gymnasium tasks.
  Besides the budget (below), the arena changes two settings: `episodic=true`, because arena events
  end episodes and the default rejects terminations, and `rho=0.7`, upstream's setting for tasks
  with terminations (`envs/mujoco.py`, issue #62). Without wandb upstream saves no agent, so the
  launcher saves the final one and writes upstream's train and eval logs as JSON lines. Evaluation
  runs upstream's planner in eval mode, with torch reseeded by the layout seed each episode.
- **DreamerV3** (`dreamerv3_train.py`, `dreamerv3_policy.py`; danijar/dreamerv3 at e01491f).
  Upstream's `main.py` with its `dmc_proprio` preset, the one for proprioceptive continuous control
  (model size 1M, train ratio 1024, 16 environments; the budget below replaces its 1.1M
  environment steps), on the task `arena_train`. A timeout is the last step of an episode but not a
  terminal one. Upstream checkpoints every 15 minutes and continues from the latest checkpoint when
  the same command runs again with `--resume`; the launcher adds the final agent. Evaluation runs
  upstream's actor in eval mode, which samples; its sampling seeds count the actions taken, and each
  episode restarts the count at the layout seed. The actor's samples can leave [−1, 1]; training
  clips them (`ClipAction`), and so does the wrapper.
- **Budget and seeds** (decided on 2026-10-30, before any training run): 2M environment steps for
  each method, the same for both so that their rows compare directly, as TD-MPC2's own DreamerV3
  results do. That lies within the papers' single-task budgets: TD-MPC2's published runs train
  Meta-World and MyoSuite for 2M steps, most DMControl and ManiSkill2 tasks for 4M and the hardest
  for 14M (`config.yaml`'s 10M is a default none of them uses), and DreamerV3's own DMControl
  proprioceptive scores reach 0.5M. It is about 12 times the world models' standard budget of
  170,304 decisions, with data each method collects with its own policy. Each method trains seeds
  42, 67 and 2027: 3 seeds, as TD-MPC2 reports per task and as the arena's other trained rows use
  (DreamerV3 reports 5 for DMControl). Every seed runs once on `test` and is reported with its
  training curve.
- **Venvs** (`scripts/setup_online_rl.py`, `requirements-tdmpc2.txt`, `requirements-dreamerv3.txt`):
  one per method, Python 3.11, with the official pins except MuJoCo 3.3.0 and NumPy 1.26.4 for the
  arena's physics, a cu128 build of torch 2.7.1, and JAX 0.5.2 with CUDA 12.9 wheels instead of 0.4.33,
  whose CUDA 12.2 compiler lacks this laptop's sm_120. Both builds also run on sm_89 (RTX 4090):
  torch through its sm_86 kernels, JAX by compiling for the device.
  As `setup_crazymarl.py` does, the script clears jaxlib 0.5.1's executable-stack request, which
  current glibc refuses to load. In both venvs `collect --verify` reports `"differ": 0`.
- **Checked on 2026-10-30** (`tests/test_uav_arena_online.py`, in each venv): every adapter step
  equals the wrapper's bit for bit; the time limit ends an episode without terminating it; a short
  training saves an agent whose evaluation replays an episode exactly.
- **Measured on this laptop** (RTX 5070 Laptop, 2026-10-30, short runs with the default settings):
  TD-MPC2's first `torch.compile` takes about 6 minutes; then it trains at about 50 environment
  steps a second with one CPU core busy, 5.6 h per million steps (2M: about 11 h). DreamerV3
  compiles in about a minute and trains at about 55 steps a second (2M: about 10 h). The 6 training
  runs take about 64 h one after another. Evaluation takes 10 ms per decision for TD-MPC2's planner
  and 1.5 ms for DreamerV3's actor.

```bash
python3 scripts/setup_online_rl.py tdmpc2
python3 scripts/setup_online_rl.py dreamerv3
for s in 42 67 2027; do
  PYTHONPATH=$PWD data/external/.venv-tdmpc2/bin/python -m uav_arena.tdmpc2_train \
    --out data/uav_arena/tdmpc2/s$s --seed $s --steps 2000000
  PYTHONPATH=$PWD data/external/.venv-dreamerv3/bin/python -m uav_arena.dreamerv3_train \
    --out data/uav_arena/dreamerv3/s$s --seed $s --steps 2000000
  PYTHONPATH=$PWD data/external/.venv-tdmpc2/bin/python -m uav_arena.gym_eval --policy uav_arena.tdmpc2_policy:make \
    --policy-options "{\"run\": \"data/uav_arena/tdmpc2/s$s\"}" --suite dev --seeds 0:5 \
    --out data/uav_arena/results/tdmpc2_s${s}_dev_s0-5.json
  PYTHONPATH=$PWD data/external/.venv-dreamerv3/bin/python -m uav_arena.gym_eval --policy uav_arena.dreamerv3_policy:make \
    --policy-options "{\"run\": \"data/uav_arena/dreamerv3/s$s\"}" --suite dev --seeds 0:5 \
    --out data/uav_arena/results/dreamerv3_s${s}_dev_s0-5.json
done
```

## Results on test (seeds 1000–1049, all 8 conditions)

At `arena-v0.3`, run on 2026-10-28 and 29 from a clean checkout of the tag (64134e7), one step at
a time with 20 workers for the simulator rollouts. U flew through the recorder, so its run is both
its row and the source of the 8,660 questions on which the harness scored every row. The example
MLP is the `explore-v2` checkpoint scored on `dev` (`participants/example/mlp.pt`, SHA-256
`110c93c4…`). CrazyMARL ran on 2026-11-01 from the same checkout and environment, once it had
reproduced its `dev` records exactly; it flies the final actor of each training seed. Success count
out of 50 (for CrazyMARL, one count per training seed) and mean time of the successes:

| Condition | U (oracle) | Physics (nominal 25 g) | Physics + mass estimation | Example MLP | CrazyMARL (seeds 0, 1, 2) |
|---|---|---|---|---|---|
| static, 0 ms | 38/50 (7.9 s) | 33/50 (7.6 s) | 37/50 (7.9 s) | 9/50 (5.8 s) | 20, 27, 20 (4.5 s) |
| static, 50 ms | 38/50 (7.9 s) | 27/50 (7.3 s) | 39/50 (7.8 s) | 12/50 (8.0 s) | 20, 27, 21 (4.7 s) |
| static, 100 ms | 38/50 (7.9 s) | 27/50 (7.5 s) | 41/50 (8.4 s) | 7/50 (6.2 s) | 20, 26, 19 (5.1 s) |
| static, 200 ms | 38/50 (7.9 s) | 25/50 (6.1 s) | 40/50 (8.0 s) | 10/50 (6.0 s) | 18, 21, 18 (5.6 s) |
| static+moving, 0 ms | 38/50 (7.6 s) | 30/50 (8.5 s) | 39/50 (8.8 s) | 7/50 (6.2 s) | 16, 25, 18 (4.6 s) |
| static+moving, 50 ms | 40/50 (8.2 s) | 26/50 (8.0 s) | 39/50 (8.8 s) | 6/50 (7.2 s) | 16, 26, 17 (4.8 s) |
| static+moving, 100 ms | 36/50 (7.5 s) | 23/50 (7.1 s) | 39/50 (8.9 s) | 4/50 (6.6 s) | 16, 24, 17 (5.1 s) |
| static+moving, 200 ms | 39/50 (8.0 s) | 22/50 (6.8 s) | 37/50 (7.7 s) | 6/50 (6.6 s) | 16, 22, 17 (5.6 s) |
| **Total** | **305/400** | **213/400** | **311/400** | **61/400** | **142, 198, 147 of 400** |
| Success rate (95% interval) | 76.25% (65.00–86.75%) | 53.25% (42.50–64.25%) | 77.75% (67.50–87.00%) | 15.25% (9.25–22.25%) | 35.50% (23.25–48.25%), 49.50% (36.75–62.50%), 36.75% (24.75–49.25%) |

Layers 1–3 on the 8,660 questions ([HARNESS.md](HARNESS.md)):

| | L1 payload | L2 cross-agent | L2 payload / others / own | L3 costly decisions | Regret, mean (median) | Top-1 | Spearman | Pairwise |
|---|---|---|---|---|---|---|---|---|
| U (oracle) | 0 | 0 | 0 / 0 / 0 | 0 | 0 (0) | 0 | 1 | 1 |
| Physics + mass estimation | 0.57 cm | 0.47 cm | 0.47 / 0.47 / 0.80 cm | 3.7% | +0.02 (+0.00003) | 0.08 | 0.96 | 0.95 |
| Physics (nominal 25 g) | 8.42 cm | 3.62 cm | 3.76 / 3.48 / 6.29 cm | 33.8% | +9.03 (+0.06) | 12.31 | 0.61 | 0.75 |
| Example MLP | 12.81 cm | 4.67 cm | 5.28 / 4.05 / 8.41 cm | 51.9% | +14.10 (+1.29) | 23.66 | 0.39 | 0.64 |

The intervals are 95% percentile intervals from 10,000 bootstrap resamples of the 50 layouts, each
bringing its 8 episodes and its questions (ANALYSIS.md, Section 4; NumPy `default_rng(0)`). For
physics + mass estimation, the nominal physics reference and the example MLP, L1 lies within
0.44–0.71, 6.02–10.86 and 11.91–13.70 cm, L2 within 0.37–0.56, 2.69–4.62 and 4.39–4.94 cm, and L3
within 2.6–4.8%, 24.8–42.8% and 44.9–58.0%.

- **U fails 95 episodes: 93 timeouts and 2 obstacle collisions.** U predicts exactly, so these are
  failures of the shared planner: on those layouts MPPI with 512 samples and a 1.5 s horizon does
  not reach the goal within 20 s. Its static episodes are identical at all four latencies
  ([HARNESS.md](HARNESS.md), Section 1), and 12 of the 50 layouts time out there. It never failed on
  its 5 `dev` layouts.
- **Physics + mass estimation matches U within the interval:** 311 against 305 successes, +1.5
  points (−3.0 to +6.0 over layouts). Their outcomes differ in 45 episodes: it succeeds in 25 that
  U fails and fails in 19 that U finishes. Unlike U, it plans from the delayed observation, so its
  static episodes differ between latencies.
- **The nominal physics reference** succeeds in 213 episodes. Its 187 failures are 76
  aircraft-separation failures, 66 timeouts, 41 obstacle collisions, 3 exits from the bounds and 1
  crash; 86 fall on episodes U fails too, 101 on episodes U finishes. As on `dev`, aircraft
  separation comes with heavy payloads: 41 of the 76 are on the 10 layouts whose payload weighs
  35.4–39.7 g, against the nominal 25 g. Its success falls with latency, from 33 to 25 of 50 without
  movers and from 30 to 22 with them, while its share of costly decisions rises only from 31.4% to
  32.6% and from 34.8% to 36.5%.
- **The example MLP** succeeds in 61 episodes, against 21 of 40 on `dev`, all of them episodes U
  finishes. Its failures are 168 obstacle collisions, 137 aircraft-separation failures and 34
  crashes.
- **CrazyMARL** succeeds in 142, 198 and 147 episodes with seeds 0, 1 and 2 (487 of 1,200, 40.6%),
  below the nominal physics reference and above the example MLP. Every failure but one is an
  obstacle collision, 2.1 s into the episode on average; the other is a timeout (seed 1). Its
  successes take 5.0 s on average, against U's 7.5–8.2 s, and 12–34 per seed fall on episodes U
  fails. As on `dev`, the outcome mostly depends on the layout alone: 41–43 of the 50 layouts end
  the same way in all 8 conditions. Latency costs it little: without movers, it goes from 20, 27
  and 20 successes at 0 ms to 18, 21 and 18 at 200 ms. Seed 0 led on `dev` (24 against 16 and 16
  of 40), seed 1 on `test`. Without a predictor, it has no layer scores (ANALYSIS.md, Section 2).
- **The layers order the rows as the closed loop does.** U scores exactly zero on all 8,660
  questions. Physics + mass estimation, the nominal physics reference and the example MLP come out
  in that order on every layer, on top-1 regret, Spearman correlation and pairwise accuracy. On L1
  and L3 their intervals do not overlap; on L2 the nominal physics reference's and the example's
  do. In 10.9% of the questions (7.9% on `dev`), U's own plan is more than 1 cost unit worse than
  its pool's best; L3 leaves them out for every row.
- **Compute.** The closed loops took 5.9 h for U through the recorder, 5.5 h for the nominal physics
  reference and 6.2 h for physics + mass estimation (0.50–0.52 s per decision), and 3.5 min for the
  example MLP; with the harness, 22.4 h. Python 3.10.21, torch 2.11.0+cu128, MuJoCo 3.3.0, NumPy
  2.2.6, SciPy 1.15.3, stable-worldmodel 0.1.1 and gcc 15.2.0 on the maintainer's laptop (Ryzen 9
  7845HX). Before the run, the same checkout reproduced one `dev` episode of every row (seed 0,
  static, 0 ms) and the example's `dev` scores in every field but the timing. The run paused once,
  after the nominal physics closed loop, and resumed after a reboot in the same environment. On
  2026-11-01, CrazyMARL's three seeds ran in parallel, one process each, in 1.5–1.7 min.
- **Files** (not in git): records in `data/uav_arena/results/` (`oracle_mppi_test_recorded.json`,
  `physics_mppi_test.json`, `adaptive_mppi_test.json`, `example_mlp_test.json`,
  `crazymarl_s{0,1,2}_test.json`), scores in `data/uav_arena/results/harness/*_test.{json,npz}`,
  questions and ground truth in `data/uav_arena/states/test` (0.47 GB).

### W-ar

The autoregressive state model's six formal models: history 4 and 60,000 steps, seeds 1, 2 and 3
on the 2,000 standard episodes and on the first 500 (selection and training in
[participants/war/README.md](participants/war/README.md)). They ran on 2026-11-02, one model at a
time, from a clean checkout of `f65e007` (`uav/war`). Since the tag, that commit adds participant
code, the Gymnasium and online-training files, tests and documentation; the environment, the
planner, the recorder and the harness are the tag's. Before the first `test` episode, the checkout
reproduced the 2,000-episode models' `dev` records exactly, timings aside, and the harness accepted
the predictor on `dev`. Success count out of 50 (one count per training seed) and mean time of the
successes:

| Condition | W-ar, 2,000 episodes (seeds 1, 2, 3) | W-ar, 500 episodes (seeds 1, 2, 3) |
|---|---|---|
| static, 0 ms | 32, 28, 31 (7.4 s) | 31, 30, 26 (6.5 s) |
| static, 50 ms | 28, 29, 27 (7.3 s) | 27, 27, 26 (6.6 s) |
| static, 100 ms | 25, 23, 25 (7.2 s) | 22, 26, 24 (6.6 s) |
| static, 200 ms | 16, 17, 16 (7.3 s) | 15, 17, 15 (6.8 s) |
| static+moving, 0 ms | 28, 29, 27 (8.1 s) | 29, 27, 29 (7.2 s) |
| static+moving, 50 ms | 23, 27, 30 (8.0 s) | 21, 23, 19 (6.8 s) |
| static+moving, 100 ms | 18, 20, 22 (8.5 s) | 22, 22, 20 (7.2 s) |
| static+moving, 200 ms | 11, 14, 15 (7.5 s) | 17, 16, 12 (7.0 s) |
| **Total** | **181, 187, 193 of 400** | **184, 188, 171 of 400** |
| Success rate (95% interval) | 45.25% (36.00–54.50%), 46.75% (37.50–56.00%), 48.25% (38.75–57.75%) | 46.00% (36.50–55.50%), 47.00% (36.75–57.00%), 42.75% (33.25–52.25%) |

Layers 1–3 on the same 8,660 questions, with 95% intervals for L1, L2 and L3:

| | L1 payload | L2 cross-agent | L2 payload / others / own | L3 costly decisions | Regret, mean (median) | Top-1 | Spearman | Pairwise |
|---|---|---|---|---|---|---|---|---|
| W-ar, 2,000 episodes, seed 1 | 8.46 cm (7.91–9.00) | 3.05 cm (2.82–3.27) | 3.60 / 2.49 / 6.26 cm | 41.2% (34.7–47.0%) | +8.53 (+0.15) | 12.10 | 0.58 | 0.72 |
| W-ar, 2,000 episodes, seed 2 | 8.65 cm (8.06–9.21) | 3.09 cm (2.86–3.31) | 3.66 / 2.51 / 6.22 cm | 42.1% (34.8–48.4%) | +9.43 (+0.16) | 12.25 | 0.58 | 0.72 |
| W-ar, 2,000 episodes, seed 3 | 8.53 cm (7.97–9.06) | 3.08 cm (2.85–3.30) | 3.65 / 2.51 / 6.20 cm | 40.9% (34.0–47.1%) | +8.44 (+0.15) | 11.47 | 0.59 | 0.72 |
| W-ar, 500 episodes, seed 1 | 9.35 cm (8.81–9.86) | 3.43 cm (3.24–3.62) | 3.97 / 2.89 / 6.63 cm | 43.6% (36.7–49.8%) | +10.38 (+0.23) | 15.52 | 0.52 | 0.69 |
| W-ar, 500 episodes, seed 2 | 9.62 cm (9.03–10.20) | 3.49 cm (3.29–3.68) | 4.04 / 2.93 / 6.67 cm | 44.3% (37.1–50.6%) | +11.42 (+0.27) | 17.28 | 0.51 | 0.69 |
| W-ar, 500 episodes, seed 3 | 9.40 cm (8.83–9.97) | 3.41 cm (3.21–3.60) | 3.95 / 2.86 / 6.53 cm | 43.3% (36.6–49.4%) | +9.74 (+0.21) | 14.65 | 0.52 | 0.69 |

- **Data size** (ANALYSIS.md, Section 5). Pooled over the three seeds, the 500-episode models are
  worse than the 2,000-episode models on every layer: L1 +0.91 cm (+0.74 to +1.08), L2 +0.37 cm
  (+0.33 to +0.42) and L3 +2.35 points (+1.33 to +3.39), as on the validation set (5.83–5.87
  against 4.44–4.49 cm). In the closed loop the two cannot be told apart: 543 against 561 successes
  of 1,200, −1.50 points (−5.08 to +1.92).
- **Against the nominal physics reference.** Pooled over seeds, W-ar succeeds in 46.75% (2,000
  episodes) and 45.25% (500) of the episodes, between the nominal physics reference (53.25%) and
  CrazyMARL (40.6% over its three seeds). Paired over layouts, the 2,000-episode models' J is 6.50
  points lower than the reference's (−14.50 to +1.33), their L1 about the same (+0.13 cm, −2.00 to
  +2.27), their L2 lower (−0.54 cm, −1.41 to +0.27) and their L3 higher (+7.60 points, +0.66 to
  +15.21). For the 500-episode models, J is 8.00 points lower (−14.92 to −1.00) and L3 9.95 points
  higher (+3.05 to +17.50). On this pair L3 orders the rows as the closed loop does and L2 does
  not; the analysis plan's test runs over the 30 primary models, not one pair.
- **Latency.** W-ar's successes fall from 91 to 49 of 150 without movers and from 84 to 40 with
  them (2,000 episodes; 87 to 47 and 85 to 45 at 500), against 33 to 25 and 30 to 22 of 50 for the
  nominal physics reference: at 0 ms the two are close, at 200 ms W-ar trails. On U's states its
  layers rise less: L1 from 7.66 to 9.45 cm and L3 from 40.3% to 42.6% without movers, and from
  7.83 to 9.83 cm and from 39.6% to 43.7% with them (2,000 episodes).
- **Failures.** At 2,000 episodes, 331 aircraft-separation failures, 207 obstacle collisions, 63
  crashes, 37 timeouts and 1 exit from the bounds. At 500 episodes obstacle collisions lead with
  307, then 269 aircraft-separation failures, 65 crashes, 14 timeouts and 2 exits. Per model, 5–11
  successes fall on episodes U fails and 119–140 failures on episodes U finishes. Successes take
  7.53–7.75 s on average at 2,000 episodes and 6.72–6.94 s at 500.
- **Compute.** 34–40 min per closed loop (0.075 s per planning call on the RTX 5070 Laptop GPU)
  and 13.6–13.8 min per harness run, 5.1 h for the six models. Training costs are in the W-ar
  README.
- **Files** (not in git): `data/uav_arena/results/formal_war_h4_60k_ep{2000,500}_s{1,2,3}_test.json`
  and `data/uav_arena/results/harness/formal_war_h4_60k_ep{2000,500}_s{1,2,3}_test.{json,npz}`.

### PETS

The PETS ensemble's six formal models: G7, five members of depth 3 and width 291, 60,000 steps per
member, seeds 1, 2 and 3 on the 2,000 standard episodes and on the first 500 (selection and training
by june in [participants/pets/PHASE_B.md](participants/pets/PHASE_B.md)). Each candidate is
propagated by one member drawn at random per planning call, with the member's mean at every step
(`sample_gaussian=false`): one particle instead of published PETS's 20, and mean propagation
([TASKS.md](TASKS.md#pets-ensemble-june)). They ran on 2026-11-02, one model at a time, from a clean
checkout of `87a502b` (PR #2), whose PETS files are merged here unchanged. Outside `participants/`,
`tests/` and the documentation, that commit is identical to W-ar's `f65e007`, so the environment,
the planner, the recorder and the harness are the tag's. june flew `dev` on Ubuntu 22.04, which
does not reproduce these episodes bit for bit, so the check before the first `test` episode was on
this machine: the six models flew `dev`, the 2,000-episode seed-1 model flew it again with identical
records, timings aside, and the harness accepted the predictor on `dev`. Success count out of 50
(one count per training seed) and mean time of the successes:

| Condition | PETS, 2,000 episodes (seeds 1, 2, 3) | PETS, 500 episodes (seeds 1, 2, 3) |
|---|---|---|
| static, 0 ms | 10, 15, 22 (7.5 s) | 6, 14, 14 (6.8 s) |
| static, 50 ms | 13, 20, 19 (7.1 s) | 12, 13, 12 (7.2 s) |
| static, 100 ms | 12, 11, 14 (7.9 s) | 9, 7, 10 (7.1 s) |
| static, 200 ms | 5, 7, 8 (7.8 s) | 6, 8, 6 (6.7 s) |
| static+moving, 0 ms | 9, 15, 17 (7.9 s) | 6, 11, 10 (7.5 s) |
| static+moving, 50 ms | 13, 15, 14 (8.2 s) | 9, 14, 10 (7.4 s) |
| static+moving, 100 ms | 7, 10, 8 (8.0 s) | 2, 5, 12 (8.3 s) |
| static+moving, 200 ms | 4, 6, 5 (7.8 s) | 2, 2, 6 (6.5 s) |
| **Total** | **73, 99, 107 of 400** | **52, 74, 80 of 400** |
| Success rate (95% interval) | 18.25% (11.50–25.75%), 24.75% (17.25–33.00%), 26.75% (19.00–35.25%) | 13.00% (8.00–19.00%), 18.50% (11.75–25.75%), 20.00% (12.75–28.00%) |

Layers 1–3 on the same 8,660 questions, with 95% intervals for L1, L2 and L3:

| | L1 payload | L2 cross-agent | L2 payload / others / own | L3 costly decisions | Regret, mean (median) | Top-1 | Spearman | Pairwise |
|---|---|---|---|---|---|---|---|---|
| PETS, 2,000 episodes, seed 1 | 12.93 cm (11.82–14.03) | 3.78 cm (3.51–4.06) | 4.60 / 2.96 / 7.02 cm | 46.7% (39.7–53.0%) | +12.32 (+2.22) | 19.06 | 0.43 | 0.66 |
| PETS, 2,000 episodes, seed 2 | 12.22 cm (11.22–13.22) | 3.67 cm (3.41–3.93) | 4.44 / 2.90 / 6.93 cm | 46.6% (39.7–52.8%) | +12.25 (+2.17) | 19.44 | 0.44 | 0.66 |
| PETS, 2,000 episodes, seed 3 | 12.44 cm (11.40–13.48) | 3.70 cm (3.43–3.97) | 4.47 / 2.94 / 6.97 cm | 46.6% (39.6–52.8%) | +12.07 (+2.24) | 19.39 | 0.44 | 0.66 |
| PETS, 500 episodes, seed 1 | 13.11 cm (11.97–14.25) | 3.88 cm (3.60–4.16) | 4.68 / 3.08 / 7.03 cm | 47.1% (40.2–53.2%) | +12.69 (+2.33) | 19.87 | 0.42 | 0.65 |
| PETS, 500 episodes, seed 2 | 13.64 cm (12.42–14.84) | 3.99 cm (3.69–4.27) | 4.79 / 3.19 / 7.16 cm | 48.4% (41.3–54.6%) | +13.19 (+2.61) | 20.18 | 0.41 | 0.65 |
| PETS, 500 episodes, seed 3 | 12.97 cm (11.88–14.05) | 3.89 cm (3.61–4.17) | 4.66 / 3.12 / 7.08 cm | 47.9% (40.8–54.1%) | +12.76 (+2.54) | 19.56 | 0.42 | 0.65 |

- **Data size** (ANALYSIS.md, Section 5). Pooled over the three seeds, the 500-episode models are
  worse than the 2,000-episode models in the closed loop, 206 against 279 successes of 1,200,
  −6.08 points (−10.00 to −2.25), and on every layer: L1 +0.71 cm (+0.59 to +0.82), L2 +0.20 cm
  (+0.18 to +0.22) and L3 +1.16 points (+0.71 to +1.57), as on the validation set (6.40–6.43
  against 6.21–6.27 cm).
- **Against the nominal physics reference and W-ar.** Pooled over seeds, PETS succeeds in 23.25%
  (2,000 episodes) and 17.17% (500) of the episodes, below the nominal physics reference (53.25%),
  W-ar (46.75% and 45.25%) and CrazyMARL (40.6% over its three seeds). Paired over layouts, the
  2,000-episode models' J is 30.00 points lower than the reference's (−38.25 to −22.08), their L1
  higher (+4.11 cm, +2.21 to +6.04), their L2 about the same (+0.10 cm, −0.71 to +0.88) and their
  L3 higher (+12.87 points, +6.30 to +20.06). Against W-ar at the same data size, J is 23.50
  points lower (−29.00 to −18.08) and every layer higher: L1 +3.98 cm (+3.33 to +4.62), L2
  +0.65 cm (+0.57 to +0.73) and L3 +5.26 points (+4.25 to +6.44); at 500 episodes, J is 28.08
  points lower (−34.25 to −21.91) and L3 4.07 points higher (+2.87 to +5.33). Against the
  reference, L3 orders the rows as the closed loop does and L2 does not, as for W-ar; against W-ar,
  all three layers do. The analysis plan's test runs over the 30 primary models, not single pairs.
- **Latency.** PETS's successes fall from 47 to 20 of 150 without movers and from 41 to 15 with
  them (2,000 episodes; 34 to 20 and 27 to 10 at 500). On U's states, L1 rises from 10.54 to
  15.07 cm and L3 from 43.5% to 48.7% without movers, and from 10.65 to 15.30 cm and from 44.5% to
  49.7% with them (2,000 episodes).
- **Failures.** At 2,000 episodes, 419 aircraft-separation failures, 349 obstacle collisions, 123
  crashes, 28 timeouts and 2 exits from the bounds; at 500 episodes, 457, 402, 101, 30 and 4. Per
  model, 0–2 successes fall on episodes U fails and 199–253 failures on episodes U finishes.
  Successes take 7.43–8.20 s on average at 2,000 episodes and 6.59–7.60 s at 500.
- **Compute.** 10.1–10.8 min per closed loop (19.2–21.9 ms per planning call on the RTX 5070 Laptop
  GPU) and 50.2–52.9 min per harness run, since a stochastic predictor makes 48 calls per question
  ([HARNESS.md](HARNESS.md)); 6.2 h for the six models. Training costs are in PHASE_B.md.
- **Files** (not in git): `data/uav_arena/results/pets_e{2000,500}_s{1,2,3}_mppi_test.json` and
  `data/uav_arena/results/harness/pets_e{2000,500}_s{1,2,3}_test.{json,npz}`. The checkpoints,
  handed over out of band, are
  `data/uav_arena/participants/pets/phase_b_formal_d3w291_s60k_e{2000,500}_seed{1,2,3}.pt`, with
  their SHA-256 in PHASE_B.md.

### JEPA

The entity-token JEPA's 18 formal models: `full`, `no_cross` and `flat` at G6 (history 4, 60,000
steps), with seeds 1, 2 and 3 on the 2,000 standard episodes and on the first 500. Yihao selected
the configuration in
[participants/jepa/PHASE_B_SELECTION.md](participants/jepa/PHASE_B_SELECTION.md), merged from
PR #3; the training is in [participants/jepa/FORMAL.md](participants/jepa/FORMAL.md).
- The maintainer trained them on 2026-11-03 from a clean checkout of `57e9b1a` (PR #3), 15 on
  AIHUB's RTX 4090 and 3 on the laptop. They ran on `test` on 2026-11-03 and 04 from the same
  checkout, one model at a time as each arrived.
- Since the tag, that commit adds only participant code, tests and documentation, so the
  environment, the planner, the recorder and the harness are the tag's.
- Before the first `test` episode, the checkout reproduced the 2,000-episode `full` seed-3 model's
  `dev` records exactly, timings aside, and the harness accepted the predictor on `dev`.

Success count out of 50 (one count per training seed) and mean time of the successes:

| Condition | `full`, 2,000 episodes (seeds 1, 2, 3) | `full`, 500 episodes | `no_cross`, 2,000 episodes | `no_cross`, 500 episodes | `flat`, 2,000 episodes | `flat`, 500 episodes |
|---|---|---|---|---|---|---|
| static, 0 ms | 27, 27, 25 (5.8 s) | 15, 23, 20 (6.3 s) | 30, 26, 31 (6.9 s) | 22, 14, 16 (6.5 s) | 10, 5, 12 (5.3 s) | 10, 6, 5 (5.7 s) |
| static, 50 ms | 33, 28, 24 (6.2 s) | 20, 22, 22 (6.5 s) | 30, 26, 28 (7.0 s) | 21, 10, 23 (7.1 s) | 12, 6, 4 (5.9 s) | 7, 10, 6 (5.3 s) |
| static, 100 ms | 26, 30, 27 (6.6 s) | 17, 21, 18 (6.6 s) | 27, 24, 27 (6.9 s) | 18, 13, 18 (6.7 s) | 10, 8, 9 (5.2 s) | 8, 5, 4 (5.7 s) |
| static, 200 ms | 22, 22, 21 (6.3 s) | 15, 17, 15 (6.4 s) | 28, 19, 24 (6.9 s) | 15, 14, 13 (6.8 s) | 10, 6, 6 (5.2 s) | 12, 12, 5 (5.6 s) |
| static+moving, 0 ms | 28, 29, 25 (6.4 s) | 16, 21, 21 (7.1 s) | 28, 23, 26 (6.9 s) | 20, 13, 16 (7.0 s) | 14, 7, 10 (5.4 s) | 4, 5, 3 (6.1 s) |
| static+moving, 50 ms | 26, 24, 25 (6.8 s) | 13, 16, 22 (6.5 s) | 25, 24, 26 (7.1 s) | 18, 14, 16 (7.3 s) | 10, 8, 6 (5.8 s) | 8, 8, 1 (6.3 s) |
| static+moving, 100 ms | 29, 27, 24 (6.6 s) | 16, 16, 16 (6.3 s) | 27, 23, 26 (7.5 s) | 6, 14, 15 (7.0 s) | 10, 8, 8 (6.3 s) | 5, 6, 3 (5.5 s) |
| static+moving, 200 ms | 20, 18, 18 (7.0 s) | 12, 13, 12 (7.1 s) | 19, 13, 13 (7.1 s) | 12, 8, 11 (7.1 s) | 7, 5, 3 (6.0 s) | 5, 5, 3 (5.2 s) |
| **Total** | **211, 205, 189 of 400** | **124, 149, 146 of 400** | **214, 178, 201 of 400** | **132, 100, 128 of 400** | **83, 53, 58 of 400** | **59, 57, 30 of 400** |
| Success rate (95% interval) | 52.75% (42.25–63.00%), 51.25% (40.50–62.00%), 47.25% (36.75–57.50%) | 31.00% (22.75–39.25%), 37.25% (27.50–47.25%), 36.50% (27.49–45.76%) | 53.50% (43.25–63.50%), 44.50% (34.50–54.50%), 50.25% (39.50–60.75%) | 33.00% (24.00–42.50%), 25.00% (17.50–32.75%), 32.00% (24.25–40.25%) | 20.75% (13.00–29.25%), 13.25% (8.75–18.25%), 14.50% (8.75–21.00%) | 14.75% (8.50–22.00%), 14.25% (7.75–21.75%), 7.50% (3.25–12.75%) |

Layers 1–3 on the same 8,660 questions, with 95% intervals for L1, L2 and L3:

| | L1 payload | L2 cross-agent | L2 payload / others / own | L3 costly decisions | Regret, mean (median) | Top-1 | Spearman | Pairwise |
|---|---|---|---|---|---|---|---|---|
| `full`, 2,000 episodes, seed 1 | 8.82 cm (8.24–9.40) | 3.51 cm (3.29–3.72) | 4.01 / 3.00 / 7.23 cm | 47.5% (40.4–53.8%) | +10.47 (+0.50) | 20.70 | 0.42 | 0.65 |
| `full`, 2,000 episodes, seed 2 | 9.13 cm (8.54–9.69) | 3.58 cm (3.37–3.78) | 4.08 / 3.07 / 7.45 cm | 48.4% (41.3–54.6%) | +11.11 (+0.56) | 21.11 | 0.42 | 0.65 |
| `full`, 2,000 episodes, seed 3 | 9.07 cm (8.48–9.66) | 3.59 cm (3.37–3.80) | 4.12 / 3.07 / 7.41 cm | 46.8% (40.0–52.9%) | +9.49 (+0.46) | 19.65 | 0.43 | 0.65 |
| `full`, 500 episodes, seed 1 | 11.26 cm (10.57–11.93) | 4.16 cm (3.96–4.36) | 4.66 / 3.65 / 8.14 cm | 49.2% (41.9–55.6%) | +12.60 (+0.86) | 22.83 | 0.39 | 0.64 |
| `full`, 500 episodes, seed 2 | 10.60 cm (10.02–11.20) | 3.99 cm (3.80–4.18) | 4.47 / 3.51 / 7.98 cm | 47.2% (40.3–53.3%) | +11.89 (+0.46) | 22.06 | 0.41 | 0.65 |
| `full`, 500 episodes, seed 3 | 10.67 cm (10.07–11.26) | 4.05 cm (3.86–4.24) | 4.54 / 3.57 / 7.99 cm | 47.1% (40.2–53.2%) | +11.29 (+0.58) | 21.12 | 0.41 | 0.65 |
| `no_cross`, 2,000 episodes, seed 1 | 8.74 cm (8.15–9.32) | 3.47 cm (3.25–3.68) | 3.98 / 2.96 / 6.94 cm | 45.6% (38.7–51.6%) | +9.90 (+0.30) | 19.54 | 0.44 | 0.66 |
| `no_cross`, 2,000 episodes, seed 2 | 8.95 cm (8.35–9.53) | 3.49 cm (3.28–3.69) | 4.00 / 2.97 / 7.04 cm | 47.6% (40.6–53.9%) | +10.84 (+0.55) | 19.61 | 0.42 | 0.65 |
| `no_cross`, 2,000 episodes, seed 3 | 8.70 cm (8.11–9.29) | 3.48 cm (3.26–3.69) | 4.00 / 2.96 / 6.96 cm | 45.1% (37.8–51.5%) | +10.40 (+0.26) | 19.04 | 0.42 | 0.65 |
| `no_cross`, 500 episodes, seed 1 | 10.59 cm (9.95–11.24) | 3.81 cm (3.61–4.02) | 4.46 / 3.16 / 7.73 cm | 46.4% (39.2–52.8%) | +12.47 (+0.39) | 21.74 | 0.41 | 0.65 |
| `no_cross`, 500 episodes, seed 2 | 10.83 cm (10.17–11.49) | 3.82 cm (3.63–4.02) | 4.48 / 3.16 / 7.75 cm | 46.7% (39.9–52.7%) | +11.55 (+0.47) | 20.62 | 0.43 | 0.65 |
| `no_cross`, 500 episodes, seed 3 | 10.60 cm (9.99–11.19) | 3.82 cm (3.62–4.03) | 4.47 / 3.17 / 7.72 cm | 46.9% (39.7–53.2%) | +12.28 (+0.43) | 22.21 | 0.42 | 0.65 |
| `flat`, 2,000 episodes, seed 1 | 11.18 cm (10.47–11.88) | 4.20 cm (3.98–4.43) | 4.68 / 3.72 / 8.78 cm | 56.2% (49.6–62.0%) | +16.67 (+2.39) | 28.96 | 0.31 | 0.61 |
| `flat`, 2,000 episodes, seed 2 | 11.21 cm (10.53–11.88) | 4.24 cm (4.02–4.46) | 4.71 / 3.78 / 8.75 cm | 56.9% (50.2–62.6%) | +15.78 (+2.24) | 27.75 | 0.34 | 0.62 |
| `flat`, 2,000 episodes, seed 3 | 11.21 cm (10.52–11.88) | 4.29 cm (4.06–4.51) | 4.77 / 3.82 / 8.75 cm | 56.2% (49.1–62.5%) | +15.56 (+2.10) | 28.87 | 0.33 | 0.61 |
| `flat`, 500 episodes, seed 1 | 12.37 cm (11.60–13.15) | 4.39 cm (4.18–4.61) | 4.90 / 3.89 / 8.34 cm | 51.0% (43.8–57.2%) | +14.95 (+1.15) | 27.40 | 0.34 | 0.62 |
| `flat`, 500 episodes, seed 2 | 12.47 cm (11.69–13.25) | 4.38 cm (4.17–4.60) | 4.91 / 3.86 / 8.35 cm | 50.7% (43.2–57.1%) | +15.12 (+1.03) | 27.06 | 0.35 | 0.62 |
| `flat`, 500 episodes, seed 3 | 12.93 cm (12.06–13.81) | 4.46 cm (4.25–4.68) | 4.99 / 3.94 / 8.40 cm | 52.5% (45.3–58.8%) | +15.33 (+1.43) | 26.23 | 0.34 | 0.62 |

- **Data size** (ANALYSIS.md, Section 5). Pooled over the three seeds, 500 episodes against 2,000:

  | Variant | J | L1 | L2 | L3 | Validation |
  |---|---|---|---|---|---|
  | `full` | 419 against 605 successes of 1,200, −15.50 points (−20.67 to −10.42) | +1.83 cm (+1.60 to +2.08) | +0.51 cm (+0.47 to +0.54) | +0.26 points (−1.25 to +1.84) | 6.68–6.93 against 4.85–5.04 cm |
  | `no_cross` | 360 against 593, −19.42 points (−24.59 to −14.33) | +1.88 cm (+1.64 to +2.11) | +0.34 cm (+0.31 to +0.37) | +0.55 points (−0.74 to +1.83) | 6.36–6.44 against 4.54–4.74 cm |
  | `flat` | 146 against 194, −4.00 points (−7.67 to 0.00) | +1.40 cm (+1.16 to +1.63) | +0.17 cm (+0.14 to +0.20) | −5.05 points (−6.71 to −3.47) | 7.39–7.48 against 6.47–6.54 cm |

  - L1 and L2 are worse with 500 episodes for every variant, as on the validation set.
  - In the closed loop, `full` and `no_cross` lose 15.50 and 19.42 points and `flat` 4.00.
  - L3 does not follow: it stays about the same for `full` and `no_cross` and falls by 5.05
    points for `flat`.
- **Against the nominal physics reference and W-ar.** Pooled over seeds at 2,000 episodes, `full`
  and `no_cross` succeed in 50.42% and 49.42% of the episodes. That is between W-ar (46.75%) and
  the nominal physics reference (53.25%); `flat` succeeds in 16.17%.
  - Paired over layouts, the closed loop cannot tell `full` from either: −2.83 points against the
    reference (−10.75 to +5.33) and +3.67 against W-ar (−3.17 to +10.42).
  - Every layer still puts `full` behind W-ar: L1 +0.46 cm (+0.29 to +0.64), L2 +0.49 cm (+0.46 to
    +0.52) and L3 +6.20 points (+4.84 to +7.57).
  - `no_cross` likewise: J +2.67 points (−3.92 to +8.83), with L1 +0.25 cm, L2 +0.41 cm and L3
    +4.73 points, each interval above 0.
  - At 500 episodes, W-ar leads `full` by 10.33 points (+6.92 to +13.92) and `no_cross` by 15.25
    (+11.08 to +19.58), and is again ahead on every layer.
  - The analysis plan's test runs over the 30 primary models, not single pairs (Pre-registered
    analysis, below).
- **Latency.** Pooled over seeds, successes from 0 to 200 ms, of 150:

  | Variant | 2,000 episodes, without movers | with movers | 500 episodes, without movers | with movers |
  |---|---|---|---|---|
  | `full` | 79 to 65 | 82 to 56 | 58 to 47 | 58 to 37 |
  | `no_cross` | 87 to 71 | 77 to 45 | 52 to 42 | 49 to 31 |
  | `flat` | 27 to 22 | 31 to 15 | 21 to 29 | 12 to 13 |

  - Without movers, `full` and `no_cross` lose less than W-ar (91 to 49).
  - On U's states L1 rises with latency, while L3 hardly changes. For `full` at 2,000 episodes,
    L1 goes from 8.14 to 9.60 cm without movers and from 8.44 to 9.91 cm with them; L3 stays at
    46.6% without movers and goes from 48.0% to 47.4% with them.
- **Failures.**

  | | `full` | `no_cross` | `flat` |
  |---|---|---|---|
  | 2,000 episodes | 344 obstacle collisions, 214 aircraft-separation failures, 27 timeouts, 10 crashes | 364, 185, 37, 21 | 280, 691, 21, 14 |
  | 500 episodes | 393, 318, 52, 16, and 2 exits from the bounds | 444, 283, 82, 29, and 2 exits | 542, 368, 25, 112, and 7 exits |

  - At 2,000 episodes obstacle collisions lead for `full` and `no_cross`, while `flat` fails
    mostly through aircraft separation. At 500 episodes obstacle collisions lead for all three.
  - Per model, 0–20 successes fall on episodes U fails and 108–275 failures on episodes U
    finishes.
  - Successes take 5.27–7.50 s on average; `flat`'s take 5.27–6.16 s.
- **Compute.**
  - Closed loop: 22.5–28.5 min per model for `full` and `no_cross` (52–59 ms per planning call on
    the RTX 5070 Laptop GPU); 6.0–10.1 min for `flat` (19–24 ms per call), whose episodes also end
    sooner.
  - Harness: 10.2–11.6 min per model for `full` and `no_cross`, 5.0–6.0 min for `flat`.
  - In all, 9.3 h for the 18 models with their `dev` runs and the check. Training costs are in
    FORMAL.md.
- **Files** (not in git):
  - `data/uav_arena/results/formal_{full,no_cross,flat}_h4_60k_ep{2000,500}_s{1,2,3}_test.json`;
  - `data/uav_arena/results/harness/formal_{full,no_cross,flat}_h4_60k_ep{2000,500}_s{1,2,3}_test.{json,npz}`;
  - the checkpoints, `data/uav_arena/participants/jepa/formal_{full,no_cross,flat}_h4_60k_ep{2000,500}_s{1,2,3}.pt`,
    with their SHA-256 in FORMAL.md.

### Pre-registered analysis

ANALYSIS.md, Sections 4 and 5, on the 30 primary models above, computed on 2026-11-04 by
`analysis.py` from their `test` records and harness scores.
- **Checks.** Before any statistic, the script checks that every model flew the same 400 episodes
  and was scored on the same 8,660 questions. It also checks that each model's recomputed J and
  layers equal the run's and the harness's own summaries, overall and per condition.
- **Provenance.** Every harness result comes from `arena-v0.3` or a descendant whose harness,
  environment and planner are the tag's, on the question manifest `3c352ce9…`. Each records its
  checkpoint's SHA-256, which matches the file here.
- **Intervals** come from the same 10,000 resamples of the 50 layouts as every interval above.

The rows at a glance. The learned rows are pooled over the three seeds: J over 1,200 episodes, the
layers the mean of the three models.

| Row | Data | Successes | J | L1 | L2 | L3 |
|---|---|---|---|---|---|---|
| U (oracle) | | 305/400 | 76.25% (65.00–86.75%) | 0 | 0 | 0 |
| Physics + mass estimation | | 311/400 | 77.75% (67.50–87.00%) | 0.57 cm (0.44–0.71) | 0.47 cm (0.37–0.56) | 3.7% (2.6–4.8%) |
| Physics (nominal 25 g) | | 213/400 | 53.25% (42.50–64.25%) | 8.42 cm (6.02–10.86) | 3.62 cm (2.69–4.62) | 33.8% (24.8–42.8%) |
| `full` | 2,000 episodes | 605/1,200 | 50.42% (40.33–60.25%) | 9.01 cm (8.43–9.58) | 3.56 cm (3.35–3.77) | 47.6% (40.6–53.7%) |
| `full` | 500 episodes | 419/1,200 | 34.92% (26.50–43.58%) | 10.84 cm (10.23–11.46) | 4.07 cm (3.87–4.26) | 47.8% (40.9–54.0%) |
| `no_cross` | 2,000 episodes | 593/1,200 | 49.42% (39.83–59.00%) | 8.80 cm (8.21–9.38) | 3.48 cm (3.26–3.69) | 46.1% (39.0–52.3%) |
| `no_cross` | 500 episodes | 360/1,200 | 30.00% (22.58–37.83%) | 10.67 cm (10.04–11.30) | 3.82 cm (3.62–4.02) | 46.7% (39.7–52.8%) |
| `flat` | 2,000 episodes | 194/1,200 | 16.17% (11.42–21.50%) | 11.20 cm (10.51–11.87) | 4.25 cm (4.02–4.47) | 56.4% (49.8–62.3%) |
| `flat` | 500 episodes | 146/1,200 | 12.17% (7.08–18.33%) | 12.59 cm (11.79–13.40) | 4.41 cm (4.20–4.63) | 51.4% (44.3–57.6%) |
| W-ar | 2,000 episodes | 561/1,200 | 46.75% (37.83–55.50%) | 8.55 cm (7.98–9.08) | 3.07 cm (2.84–3.29) | 41.4% (34.5–47.6%) |
| W-ar | 500 episodes | 543/1,200 | 45.25% (36.17–54.33%) | 9.46 cm (8.89–10.00) | 3.44 cm (3.25–3.63) | 43.7% (36.9–49.8%) |
| PETS | 2,000 episodes | 279/1,200 | 23.25% (16.42–30.92%) | 12.53 cm (11.49–13.58) | 3.72 cm (3.45–3.99) | 46.6% (39.7–52.8%) |
| PETS | 500 episodes | 206/1,200 | 17.17% (11.42–23.67%) | 13.24 cm (12.10–14.37) | 3.92 cm (3.63–4.20) | 47.8% (40.8–54.0%) |
| Example MLP | | 61/400 | 15.25% (9.25–22.25%) | 12.81 cm (11.91–13.70) | 4.67 cm (4.39–4.94) | 51.9% (44.9–58.0%) |

**Primary test: supported.** ρ(L2, J | L1) = −0.390 (−0.602 to −0.274) across the 30 models.
- The interval lies below 0, and J takes 30 distinct values, so the outcome is not degenerate.
- Controlling for L1, models with larger cross-agent error succeed less often: layer 2 explains
  closed-loop differences that layer 1 misses.
- The correlations it is built from are ρ(L1, J) = −0.880 (−0.911 to −0.838), ρ(L2, J) = −0.799
  (−0.862 to −0.762) and ρ(L2, L1) = +0.774 (+0.744 to +0.796).
- **Robustness:**
  - None of the 10,000 resamples gives a value of 0 or above.
  - Leaving out one layout at a time keeps the estimate between −0.450 and −0.352.
  - The resampled values center below the estimate (median −0.434), so the percentile interval
    leans negative. A bias-corrected and accelerated (BCa) interval, outside the plan, gives
    −0.498 to −0.168.

Secondary analyses (Section 5, no decision rule). The sensitivity set adds physics, physics + mass
estimation and the example MLP:

| Statistic | 30 primary models | Sensitivity set (33 models) |
|---|---|---|
| ρ(L2, J \| L1) | −0.390 (−0.602 to −0.274) | −0.404 (−0.586 to −0.147) |
| ρ(L3, J \| L1) | −0.231 (−0.524 to −0.092) | −0.291 (−0.570 to −0.138) |
| ρ(L1, J) | −0.880 (−0.911 to −0.838) | −0.896 (−0.921 to −0.846) |
| ρ(L2, J) | −0.799 (−0.862 to −0.762) | −0.816 (−0.878 to −0.711) |
| ρ(L3, J) | −0.615 (−0.764 to −0.518) | −0.700 (−0.809 to −0.616) |
| ρ(L2, J) − ρ(L1, J) | +0.081 (+0.008 to +0.124) | +0.080 (+0.012 to +0.159) |
| ρ(L2, J \| L1, own CEE) | −0.295 (−0.435 to −0.032) | −0.197 (−0.364 to +0.264) |
| ρ(L2 payload part, J \| L1) | −0.408 (−0.639 to −0.295) | −0.404 (−0.634 to −0.251) |
| ρ(L2 others part, J \| L1) | −0.317 (−0.566 to −0.188) | −0.328 (−0.529 to −0.084) |
| ρ(L2 payload part, J) | −0.867 (−0.921 to −0.830) | −0.883 (−0.929 to −0.839) |
| ρ(L2 others part, J) | −0.651 (−0.754 to −0.543) | −0.643 (−0.741 to −0.551) |

- **L1 alone tracks J more closely than L2 alone:** the difference of +0.081 lies above 0. Layer 2
  adds to layer 1; it does not replace it.
- **Layer 3** also explains differences that L1 misses.
- **The changed aircraft's own action sensitivity.** Controlling for it as well, L2's relation to
  J holds on the primary set and not on the sensitivity set, whose interval contains 0.
- **Each part of L2**, the payload and the other aircraft, relates to J on its own, controlling for
  L1.

Per condition, 50 episodes per model; the means are over the 30 models:

| Condition | Mean J | Mean L1 / L2 (cm) / L3 | Distinct J | ρ(L2, J \| L1) | ρ(L3, J \| L1) | ρ(L1, J) | ρ(L2, J) |
|---|---|---|---|---|---|---|---|
| static, 0 ms | 38.9% | 9.67 / 3.65 / 46.2% | 17 | −0.519 (−0.623 to −0.062) | −0.459 (−0.601 to −0.089) | −0.907 (−0.919 to −0.807) | −0.857 (−0.877 to −0.729) |
| static, 50 ms | 39.0% | 10.06 / 3.68 / 46.2% | 18 | −0.566 (−0.657 to −0.251) | −0.327 (−0.515 to −0.058) | −0.856 (−0.882 to −0.751) | −0.846 (−0.874 to −0.718) |
| static, 100 ms | 34.5% | 10.60 / 3.73 / 46.8% | 20 | −0.501 (−0.618 to −0.186) | −0.300 (−0.514 to −0.014) | −0.858 (−0.891 to −0.769) | −0.809 (−0.851 to −0.688) |
| static, 200 ms | 27.5% | 11.60 / 3.82 / 47.7% | 16 | −0.034 (−0.258 to +0.304) | −0.152 (−0.393 to +0.165) | −0.879 (−0.894 to −0.703) | −0.705 (−0.756 to −0.474) |
| static+moving, 0 ms | 36.4% | 9.92 / 3.72 / 47.3% | 21 | −0.285 (−0.470 to +0.128) | −0.193 (−0.455 to +0.161) | −0.900 (−0.925 to −0.818) | −0.818 (−0.859 to −0.701) |
| static+moving, 50 ms | 33.9% | 10.64 / 3.77 / 47.2% | 19 | −0.431 (−0.572 to +0.061) | −0.312 (−0.502 to +0.061) | −0.903 (−0.921 to −0.791) | −0.832 (−0.869 to −0.687) |
| static+moving, 100 ms | 29.8% | 11.14 / 3.92 / 50.2% | 19 | −0.232 (−0.399 to +0.016) | +0.075 (−0.314 to +0.276) | −0.845 (−0.889 to −0.751) | −0.734 (−0.793 to −0.615) |
| static+moving, 200 ms | 20.5% | 11.92 / 3.90 / 49.0% | 17 | −0.162 (−0.365 to +0.073) | −0.163 (−0.456 to +0.103) | −0.849 (−0.883 to −0.692) | −0.740 (−0.792 to −0.587) |

- **Static scenes at 0–100 ms:** the interval of ρ(L2, J | L1) lies below 0.
- **Static scenes at 200 ms, and moving obstacles at every latency:** the interval contains 0.
- **L1** stays closely tied to J in every condition.
- **Success falls with latency and with movers:** the mean J over the 30 models goes from 38.9% to
  27.5% without movers and from 36.4% to 20.5% with them.

Data size, 500 minus 2,000 episodes, mean over the three seeds:

| Row | J (points) | L1 (cm) | L2 (cm) | L3 (points) |
|---|---|---|---|---|
| `full` | −15.50 (−20.67 to −10.42) | +1.83 (+1.60 to +2.08) | +0.51 (+0.47 to +0.54) | +0.26 (−1.25 to +1.84) |
| `no_cross` | −19.42 (−24.59 to −14.33) | +1.88 (+1.64 to +2.11) | +0.34 (+0.31 to +0.37) | +0.55 (−0.74 to +1.83) |
| `flat` | −4.00 (−7.67 to 0.00) | +1.40 (+1.16 to +1.63) | +0.17 (+0.14 to +0.20) | −5.05 (−6.71 to −3.47) |
| W-ar | −1.50 (−5.08 to +1.92) | +0.91 (+0.74 to +1.08) | +0.37 (+0.33 to +0.42) | +2.35 (+1.33 to +3.39) |
| PETS | −6.08 (−10.00 to −2.25) | +0.71 (+0.59 to +0.82) | +0.20 (+0.18 to +0.22) | +1.16 (+0.71 to +1.57) |

- Less data raises L1 and L2 for every row.
- J falls for `full`, `no_cross` and PETS. For `flat` the interval reaches 0, and W-ar's change
  cannot be told from 0.
- L3 rises only for W-ar and PETS and falls for `flat`.

Structure: `full`, `no_cross` and `flat`, paired by seed and data size:

| Comparison | Data | J (points) | L2 (cm) |
|---|---|---|---|
| `full` − `no_cross` | 2,000 episodes | +1.00 (−2.50 to +4.42) | +0.08 (+0.07 to +0.09) |
| `full` − `no_cross` | 500 episodes | +4.92 (+0.92 to +8.83) | +0.25 (+0.24 to +0.26) |
| `full` − `no_cross` | both | +2.96 (+0.29 to +5.75) | +0.16 (+0.16 to +0.17) |
| `full` − `flat` | 2,000 episodes | +34.25 (+26.92 to +41.58) | −0.69 (−0.73 to −0.64) |
| `full` − `flat` | 500 episodes | +22.75 (+16.67 to +28.67) | −0.35 (−0.39 to −0.30) |
| `full` − `flat` | both | +28.50 (+22.54 to +34.38) | −0.52 (−0.56 to −0.48) |
| `no_cross` − `flat` | 2,000 episodes | +33.25 (+26.25 to +40.25) | −0.77 (−0.81 to −0.72) |
| `no_cross` − `flat` | 500 episodes | +17.83 (+13.17 to +22.50) | −0.60 (−0.63 to −0.56) |
| `no_cross` − `flat` | both | +25.54 (+20.17 to +30.88) | −0.68 (−0.72 to −0.64) |

- **Entity tokens against one token per frame:** `full` and `no_cross` beat `flat` by 28.50 and
  25.54 points of J over both data sizes, with 0.52 and 0.68 cm less L2.
- **`full` against `no_cross`:** J and L2 disagree. `full` succeeds 2.96 points more often, mostly
  at 500 episodes, while its L2 is 0.16 cm higher. This pair runs against the relation of the
  primary test.

**Files** (not in git): `data/uav_arena/results/primary_analysis.json`, with every number above
per model and per condition and each result's provenance, and
`data/uav_arena/results/primary_analysis.txt`, the printed report.

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

`openloop.py` on the contract's validation set (100 `explore-v2` episodes, seeds
299000–299099, every decision scored). Mean position error over the 15-decision (1.5 s) horizon,
payload / aircraft:

| Latency | Physics (nominal 25 g) | Physics + mass estimation |
|---|---|---|
| 0 ms | 3.67 / 5.55 cm | 0.23 / 0.33 cm |
| 50 ms | 3.77 / 5.58 cm | 0.24 / 0.33 cm |
| 100 ms | 4.09 / 5.99 cm | 0.26 / 0.35 cm |
| 200 ms | 4.50 / 6.40 cm | 0.28 / 0.37 cm |
| **Selection score** | **4.01 cm** | **0.25 cm** |

The selection score is the contract's model-selection metric: the payload error averaged over the
horizon and the four latencies. U has no row: it needs the simulator's true state, which recorded
data does not hold. With the true mass the physics reference scores below 10⁻⁶ m (tested). On the
`explore-v1` validation set the two rows scored 2.93 and 0.16 cm.

## Evaluation layers

Layers 1–3 score a predictor at decision states recorded from U's closed-loop runs (every 5th
decision; `dev` for development, `test` for results), against exact rollouts of the true simulator
from the recorded state. [HARNESS.md](HARNESS.md) defines them and `harness.py` computes them. Layer 4
is the closed loop of `run.py`.

1. **Open-loop prediction:** the error of the prediction for U's 15-decision plan, bridging the
   latency. The validation selection score above remains the metric for model selection.
2. **Cross-agent counterfactuals:** change one aircraft's increments, by the planner's own sampling
   noise or by a 10 cm offset, and compare predicted and true changes for the payload, the other two
   aircraft and the changed aircraft itself. The cross-agent CEE averages the payload and the other
   two aircraft.
3. **Decision level:** on the 512 candidates U's planner scored, the plan the planner would execute
   with the model's costs, and how much worse its true cost is than U's (executed-plan regret). The
   layer-3 number is the share of costly decisions, regrets above 1 cost unit, on the questions where
   U's own plan is a reliable reference. Top-1 and top-8 elite regret and the ranking metrics are
   reported alongside.
4. **Closed loop:** success, failure types, time to goal and clearance per condition.

On `dev` ([HARNESS.md](HARNESS.md), Section 7), U scores exactly zero on layers 1–3, and the layers
order physics + mass estimation, the nominal physics reference and the example MLP as their
closed-loop success does. Trained on `explore-v1`, the example MLP's layer-1 error on U's states
was 14.6 cm, against 1.46 cm on its validation set: the plans U flies lay outside that data. Trained
on `explore-v2`, it is 10.8 cm, and its `dev` closed loop rises from 7 to 21 successes of 40. On
`test` U again scores exactly zero, on all 8,660 questions, and the layers order the other rows as
the closed loop does (Results on test, above).

The headline analysis, pre-registered in [ANALYSIS.md](ANALYSIS.md), asks whether the cross-agent
layer explains closed-loop differences that layer 1 misses. Its test is the partial Spearman
correlation between layer 2 and the `test` success rate across 30 learned models (5 rows × 3 seeds ×
2 data budgets), controlling for layer 1, with a bootstrap interval over layouts.
- **Result:** −0.390 (−0.602 to −0.274), so the test is supported. Layer 1 alone still correlates
  more closely with success than layer 2 alone; layer 2 adds to it.
- **Per condition:** the relation holds in static scenes up to 100 ms. In static scenes at 200 ms
  and with moving obstacles, it is not established (Pre-registered analysis, above).

## Roadmap

1. ~~Environment, shared MPPI, U and physics references, setpoint segments, hidden payload mass~~
   (done).
2. CrazyMARL on arena tasks: ~~training environment, motor-level arena interface, seeds 0–2 on
   `dev` and on `test`~~ (done).
3. ~~Physics with online payload-mass estimation~~ (done).
4. ~~Participant contract, data collector with reference manifests, training windows, layer-1
   open-loop metric~~ (done, `arena-v0.2`; `explore-v2` data in `arena-v0.3`); ~~the
   autoregressive model, PETS and the entity-token JEPA variants~~ (done).
5. ~~Layer 1–3 harness on recorded U states~~ (done, validated on `dev`).
6. TD-MPC2 and DreamerV3: ~~a Gymnasium wrapper, adapters to the official code and their venvs~~
   (done); training, then their rows on `test`.
7. Runs on `test`: ~~the reference rows, the example participant, CrazyMARL, W-ar, PETS, the JEPA
   variants and the pre-registered layer-consistency analysis~~ (done); the TD-MPC2 and DreamerV3
   rows and the leaderboard tables.

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
  tests/test_uav_arena_example.py tests/test_uav_arena_replay.py tests/test_uav_arena_crazymarl.py \
  tests/test_uav_arena_harness.py tests/test_uav_arena_gym.py tests/test_uav_arena_online.py \
  tests/test_uav_arena_analysis.py
for p in oracle physics adaptive; do
  PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run --predictor $p --suite dev --seeds 0:5 \
    --workers 20 --out data/uav_arena/results/${p}_mppi_dev_s0-5_v3.json
done
```

The `dev` table above ran the three rows in parallel with 7 workers each. The outcome does not
depend on the worker count: two of its episodes, rerun alone with 3 workers, reproduced their
records exactly.

The `test` results:

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.record --suite test --seeds 0:50 --workers 20 \
  --out data/uav_arena/states/test --results data/uav_arena/results/oracle_mppi_test_recorded.json
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.harness truth --states data/uav_arena/states/test --workers 20
for p in physics adaptive oracle; do
  PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.harness score --predictor $p \
    --states data/uav_arena/states/test --workers 20 --out data/uav_arena/results/harness/${p}_test.json
done
for p in physics adaptive; do
  PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run --predictor $p --suite test --seeds 0:50 \
    --workers 20 --out data/uav_arena/results/${p}_mppi_test.json
done
```

The example participant runs the same `harness score` and `run.py` commands with
`--predictor uav_arena.participants.example.model:make` and its `--predictor-options`
([TASKS.md](TASKS.md)), writing `example_mlp_test.json` in both places; W-ar's six models run them
with `uav_arena.participants.war.model:make` and each checkpoint
([participants/war/README.md](participants/war/README.md#commands)), PETS's with
`uav_arena.participants.pets.model:make`
([participants/pets/PHASE_B.md](participants/pets/PHASE_B.md#commands)), JEPA's 18 with
`uav_arena.participants.jepa.model:make`
([participants/jepa/FORMAL.md](participants/jepa/FORMAL.md#commands)). CrazyMARL has no
predictor, so it skips the harness; each seed's final actor flies `test` directly:

```bash
for s in 0 1 2; do
  PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.crazymarl_policy --actor data/uav_arena/crazymarl/ippo_s$s.npz \
    --suite test --seeds 0:50 --out data/uav_arena/results/crazymarl_s${s}_test.json
done
```

The pre-registered analysis, once the 30 primary models and the reference rows have their `test`
and harness results:

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.analysis --results data/uav_arena/results
```

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.collect --seeds 200000:202000 \
  --out data/uav_arena/datasets/explore_v2
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.collect --seeds 299000:299100 \
  --out data/uav_arena/datasets/explore_v2_val
for d in explore_v2 explore_v2_val; do
  PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.collect --verify data/uav_arena/datasets/$d
done
for p in physics adaptive; do
  PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop --predictor $p \
    --data data/uav_arena/datasets/explore_v2_val --out data/uav_arena/results/${p}_openloop_val_v2.json
done
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.coverage --data data/uav_arena/datasets/explore_v2_val \
  --states data/uav_arena/states/dev
```

Both datasets take under a minute on 24 cores (0.7 GB and 35 MB), and `--verify` must report
every episode as a match with the reference manifests. The batched controller compiles once into
`data/uav_arena/native_build/` (override with `UAV_ARENA_BUILD_CACHE`).
