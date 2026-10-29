# Participant contract (arena-v0.3)

The rules for adding a world model, or any other predictor, to the UAV arena. Every row of the
benchmark follows them, the reference rows included. The task, the clocks and the reference rows
are described in [README.md](README.md).

## 1. What is frozen

Tag `arena-v0.3` freezes:
- the environment, observation and action (`env.py`, `lowlevel.py`, `geometry.py`);
- the scenario distribution, including the hidden payload mass and the seed suites (`scenarios.py`);
- the shared cost and constraints (`costs.py`);
- the planner and its settings (`planning.py`, `run.py`): MPPI with 512 samples, a 15-decision
  horizon, variance scale 0.04, temperature 0.05, one iteration, a warm start, and the solver
  reseeded every episode from (seed 0, layout seed);
- the data collector, its behavior policies and the reference manifests of the standard datasets
  (`collect.py`, `explore-v2`, `manifests/`), the training windows (`windows.py`) and the
  open-loop metric (`openloop.py`);
- the analysis plan, [ANALYSIS.md](ANALYSIS.md).

Since `arena-v0.2`, only `collect.py` has changed: it adds the behavior policy `explore-v2`, whose
datasets replace the `explore-v1` ones as the standard (Section 5). The closed loop never imports
`collect.py`, so the reference rows' results carry over. The tag was created on 2026-10-28, before
the first `test` run. The evaluation harness for layers 1–3 ([HARNESS.md](HARNESS.md)) is maintained by the
maintainer and is not frozen; ANALYSIS.md fixes what each quantity it reports means.

Participants never edit these files: `.github/CODEOWNERS` routes any change to the maintainer. A
change to any of the code files needs a new tag and reruns of the U, physics and adaptive rows. A
change to ANALYSIS.md after any `test` result has been seen needs a new tag and is labelled as made
after seeing results.

## 2. The predictor interface

A participant is one object; the shared planner calls it.

```python
class MyWorldModel:
    def reset(self): ...              # optional: a new episode starts; clear all episode state
    def observe(self, obs, layout): ...   # optional: every decision, before planning
    def __call__(self, info, setpoints):  # required
        # setpoints: torch (..., H, 3, 3), world-frame segment end points for the next H decisions,
        #            the first one starting at obs["decision_time"]
        # returns {"uav": (..., H, 3, 3), "payload": (..., H, 3), "payload_vel": (..., H, 3)}:
        #            the team state at the end of each of those decisions, on the device and in the
        #            dtype of `setpoints`
        ...
    def close(self): ...              # optional: release workers or GPU memory
```

- `obs` is `ArenaEnv.observation()`, recorded `obs["latency"]` before the decision. The predictor
  has to bridge the delay from `obs["time"]` to `obs["decision_time"]` itself, using
  `obs["recent_setpoints"]` and `obs["recent_setpoint_times"]` (the last four setpoints sent, with
  their start times; each moves linearly from the previous one over 100 ms).
- `layout` is `Scenario.public()`: start, goal, pillars, movers and condition, with
  `payload_mass=None`.
- `info` holds the current observation as batched tensors (`planning.info_from_observation`).
  MPPI calls the predictor with leading shape (1, 512). The evaluation harness also calls it with
  (1, 1) and (1, 555), so a predictor must accept any leading shape.
- To use the augmented-Lagrangian solver (`--solver lagrangian`, reported separately), the output
  must be differentiable with respect to `setpoints`.
- The same object serves every evaluation layer: open-loop prediction, cross-agent
  counterfactuals, plan ranking and closed loop ([HARNESS.md](HARNESS.md)).
- **Stochastic predictors**, whose output is random for a fixed input, set `stochastic = True`,
  draw all their randomness from torch's global generator, and reseed that generator in `reset()`
  from a fixed seed. Each episode then replays alone exactly, whatever ran before it; `run.py`
  reseeds only the solver's own generator. The harness fixes the seed before its calls, so a
  counterfactual and its base see the same draws. A predictor without `stochastic = True` must
  return identical outputs for identical calls.

Run it in closed loop, or on recorded episodes, without touching frozen files:

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run --predictor my_package.my_model:make \
  --predictor-options '{"checkpoint": "data/my_model/ckpt.pt"}' --suite dev --seeds 0:5 \
  --out data/uav_arena/results/my_model_mppi_dev_s0-5.json
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop --predictor my_package.my_model:make \
  --predictor-options '{"checkpoint": "data/my_model/ckpt.pt"}' --data data/uav_arena/datasets/explore_v2_val \
  --out data/uav_arena/results/my_model_openloop_val.json
```

## 3. Information rules

At decision time a predictor may use:
- `obs`, `layout` and `info`;
- anything it computed from earlier observations of the same episode.

At decision time a predictor may not use:
- the payload mass, by any route (seed lookups, data files, the environment object);
- the true undelayed state, the onboard motor and controller states, or simulator internals;
- future obstacle paths. Obstacles are scored by the shared cost with constant-velocity
  forecasts, so a predictor only predicts the team.
- anything carried over from another episode. `reset()` starts every episode from the trained
  model, including the state of its random generator (Section 2).

In training:
- Use only training seeds (Section 4), through `collect.py` or your own simulation of those
  seeds' scenarios.
- The payload mass stored in data files is for diagnostics. It is neither an input nor a training
  target.
- An ablation that breaks a rule is reported as privileged, separately from the rows.

## 4. Seeds and splits

| Use | Seeds |
|---|---|
| `dev` suite: closed-loop development and debugging (reference rows on 0–4) | 0–19 |
| `test` suite: final closed-loop results, once per trained model | 1000–1049 |
| CrazyMARL training layouts (reserved) | 100000–102047 |
| World-model training data, and the layouts online methods train on (Gymnasium wrapper) | 200000–298999 |
| World-model validation data (model selection) | 299000–299999 |

## 5. Data

- **Collector:** `python -m uav_arena.collect --seeds 200000:202000 --out data/uav_arena/datasets/explore_v2`.
  Behavior policy `explore-v2`: formation flight between random waypoints, at random speeds and
  formation sizes, flown with the shared planner's action statistics.
  - Every decision, each aircraft's increment is the step to its place in the formation plus
    Gaussian noise at MPPI's sampling scale, 0.04 m per axis. The offsets this builds up decay over
    the planner's 15-decision horizon, so within a horizon they wander like a candidate's.
  - Formation speeds reach the planner's action limit, and setpoint targets keep only the cost's
    margins: 0.30 m above the payload and 0.18 m between aircraft.
  - Episodes are open-field versions of the seeds' scenarios, with the hidden 10–40 g payload, and
    last up to 20 s. An arena event ends most of them early. Of the 2,000 standard training
    episodes, 204 fly the full 20 s; 1,686 end in an aircraft-separation event, 104 in an exit from
    the workspace and 6 in a crash. The data thus holds the close approaches that the planner's
    candidates explore.
- **Why `explore-v2`.** U's planner commands larger, more individual and more abrupt increments than
  `explore-v1` flew. `coverage.py` compares a dataset with U's flights (HARNESS.md, Section 7). On
  U's `dev` flights, `explore-v1` left 47%, 70% and 64% of U's per-aircraft increments, differences
  from the team's mean increment and changes of increment beyond its 99th percentile; `explore-v2`
  leaves 8%, 15% and 16%, and covers U's states. Trained on `explore-v2` instead, the example
  participant's layer-1 error on U's states falls from 14.6 to 10.8 cm, and its `dev` closed loop
  rises from 7 to 21 successes of 40.
- **What a file holds:** the true state every 10 ms (aircraft position, velocity, rotation columns,
  body rates; payload position, velocity) and every setpoint flown, with its time. Any latency is
  applied offline, as `openloop.observation` does.
- **Standard budget:** 2,000 training episodes (seeds 200000–201999), 170,304 decisions (about
  4.7 hours of flight) and 0.7 GB. The data-scaling axis uses the first 250, 500, 1,000 and 2,000.
  The analysis uses two budgets, the standard 2,000 and the first 500 (seeds 200000–200499); see
  Section 6.
- **Validation set:** seeds 299000–299099 (100 episodes), collected with
  `--seeds 299000:299100 --out data/uav_arena/datasets/explore_v2_val`.
- **Storage and verification:** `data/` is never committed. Regenerate the datasets from the tag
  (about a minute on 24 cores) or copy them, then run
  `python -m uav_arena.collect --verify <directory>`. It compares every episode's content hash
  (array names, dtypes, shapes and values) with the reference manifests of its policy in
  `uav_arena/manifests/`; each manifest records the commit that generated it. A file's own SHA-256
  also depends on the zlib build that compressed it; the content hash does not. Regeneration has
  been checked on Linux x86-64 (Ubuntu 26.04, gcc 15.2). It does not reproduce the references on
  Ubuntu 22.04; there, and on any platform where verification fails, copy the data. An episode
  whose contents differ from the reference is another dataset.
- **Older policy:** `explore-v1`, the standard data up to `arena-v0.2`, stays in `collect.py` with
  its manifests (`--policy explore-v1`). Results on it are labelled with the policy and are not
  comparable with `explore-v2` results.
- **Training windows (optional):** `windows.py` turns episodes into training examples aligned
  exactly as `openloop.py` scores predictions. It never loads the payload mass, and it pads the
  episode start the same way at training and evaluation time (call `padded_recent` in `observe`).
- **Another behavior policy** (for example MPPI-collected data) is a new policy version, added to
  `collect.py` by the maintainer and named in every result that uses it.

## 6. Budgets and model selection

- Every model trains on the same data budgets (Section 5) and gets at most 8 hyperparameter
  configurations, selected at the standard budget.
- Selection uses the validation set only. The score is `openloop.py`'s selection score: the
  payload position error averaged over the 15-decision horizon and the latencies 0, 50, 100 and
  200 ms. Closed-loop `dev` runs are for debugging and never choose a configuration.
- The chosen configuration trains with 3 new seeds, not the selection run (selection biases that
  run's validation score upward), and each runs once on `test`, all 8 conditions, with the frozen
  planner.
- The same configuration, with the same number of training steps and no new selection, also
  trains 3 seeds on the first 500 standard episodes (seeds 200000–200499), and each of these runs
  once on `test` too. Each row thus brings 6 models to the analysis ([ANALYSIS.md](ANALYSIS.md),
  Section 2).
- Report the parameter count, training GPU-hours and the time of one planning call (512 × 15).

## 7. Results and pull requests

- The task board, [TASKS.md](TASKS.md), lists the rows, their owners, the two phases and what each
  row delivers; `participants/example/` is a working template for the whole path.
- `run.py` and `openloop.py` write JSON under `data/uav_arena/results/`. These files are not
  committed.
- Layers 1–3 are computed with the evaluation harness ([HARNESS.md](HARNESS.md)) on U's recorded
  `test` questions, and the headline analysis follows [ANALYSIS.md](ANALYSIS.md).
- Work on your own branch from `uav-arena`, whose frozen files are those of `arena-v0.3`. Put the
  model under your own module (for example `uav_arena/participants/<name>/`) with its tests, and
  open a pull request into `uav-arena`. The description holds the summary tables and the commands
  that produced them.
- The maintainer adds the row to the README.
