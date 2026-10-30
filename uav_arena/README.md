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
  the headline test, its decision rule and the secondary analyses.
- The evaluation harness for layers 1–3, [HARNESS.md](HARNESS.md), is built (`record.py`,
  `harness.py`), tested and validated on `dev`. It is maintained outside the frozen contract and
  scores every model on decision states recorded from U's closed-loop runs.
- Rows, owners and deliverables are on the task board, [TASKS.md](TASKS.md);
  `participants/example/` is a working example participant.
- CrazyMARL runs on arena tasks (`crazymarl_task.py`, `crazymarl_env.py`, `crazymarl_train.py`,
  `crazymarl_policy.py`). Seeds 0–2 are trained and evaluated on `dev`.
- Methods that train online, TD-MPC2 and DreamerV3, get a Gymnasium wrapper (`gym_env.py`,
  `gym_eval.py`; Gymnasium wrapper, below) and adapters to their pinned official code, with a venv
  each (TD-MPC2 and DreamerV3, below). Their training runs come next: 2M environment steps and
  seeds 42, 67 and 2027 for each method.
- The reference rows and the example participant ran on `test` on 2026-10-28 and 29, U's through
  the recorder, which also recorded the questions (Results on test, below). Next: the CrazyMARL
  rows on `test`, then the world-model rows (see Roadmap).
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
| CrazyMARL | external, model-free: official IPPO trainer, decentralized 250 Hz motor commands, retrained on arena tasks | own state fresh; payload, teammates and obstacles delayed | seeds 0–2 evaluated on `dev` |
| Entity-token JEPA (`full`, `no_cross`, `flat`), autoregressive state model, PETS ensemble | world models in the shared MPPI | observation history | planned |
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
`110c93c4…`). CrazyMARL has not run on `test` yet. Success count out of 50 and mean time of the
successes:

| Condition | U (oracle) | Physics (nominal 25 g) | Physics + mass estimation | Example MLP |
|---|---|---|---|---|
| static, 0 ms | 38/50 (7.9 s) | 33/50 (7.6 s) | 37/50 (7.9 s) | 9/50 (5.8 s) |
| static, 50 ms | 38/50 (7.9 s) | 27/50 (7.3 s) | 39/50 (7.8 s) | 12/50 (8.0 s) |
| static, 100 ms | 38/50 (7.9 s) | 27/50 (7.5 s) | 41/50 (8.4 s) | 7/50 (6.2 s) |
| static, 200 ms | 38/50 (7.9 s) | 25/50 (6.1 s) | 40/50 (8.0 s) | 10/50 (6.0 s) |
| static+moving, 0 ms | 38/50 (7.6 s) | 30/50 (8.5 s) | 39/50 (8.8 s) | 7/50 (6.2 s) |
| static+moving, 50 ms | 40/50 (8.2 s) | 26/50 (8.0 s) | 39/50 (8.8 s) | 6/50 (7.2 s) |
| static+moving, 100 ms | 36/50 (7.5 s) | 23/50 (7.1 s) | 39/50 (8.9 s) | 4/50 (6.6 s) |
| static+moving, 200 ms | 39/50 (8.0 s) | 22/50 (6.8 s) | 37/50 (7.7 s) | 6/50 (6.6 s) |
| **Total** | **305/400** | **213/400** | **311/400** | **61/400** |
| Success rate (95% interval) | 76.25% (65.00–86.75%) | 53.25% (42.50–64.25%) | 77.75% (67.50–87.00%) | 15.25% (9.25–22.25%) |

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
  after the nominal physics closed loop, and resumed after a reboot in the same environment.
- **Files** (not in git): records in `data/uav_arena/results/` (`oracle_mppi_test_recorded.json`,
  `physics_mppi_test.json`, `adaptive_mppi_test.json`, `example_mlp_test.json`), scores in
  `data/uav_arena/results/harness/*_test.{json,npz}`, questions and ground truth in
  `data/uav_arena/states/test` (0.47 GB).

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
2 data budgets), controlling for layer 1, with a bootstrap interval over layouts. Per-condition
results show how the relation changes with latency and moving obstacles.

## Roadmap

1. ~~Environment, shared MPPI, U and physics references, setpoint segments, hidden payload mass~~
   (done).
2. CrazyMARL on arena tasks: ~~training environment, motor-level arena interface, seeds 0–2 on
   `dev`~~ (done); evaluation on `test`.
3. ~~Physics with online payload-mass estimation~~ (done).
4. ~~Participant contract, data collector with reference manifests, training windows, layer-1
   open-loop metric~~ (done, `arena-v0.2`; `explore-v2` data in `arena-v0.3`);
   entity-token JEPA variants, the autoregressive model and PETS.
5. ~~Layer 1–3 harness on recorded U states~~ (done, validated on `dev`).
6. TD-MPC2 and DreamerV3: ~~a Gymnasium wrapper, adapters to the official code and their venvs~~
   (done); training, then their rows on `test`.
7. Runs on `test`: ~~the reference rows and the example participant~~ (done); the world-model rows,
   the leaderboard tables and the layer-consistency analysis.

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
  tests/test_uav_arena_harness.py tests/test_uav_arena_gym.py tests/test_uav_arena_online.py
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
([TASKS.md](TASKS.md)), writing `example_mlp_test.json` in both places.

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
