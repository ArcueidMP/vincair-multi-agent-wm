# Analysis plan (draft for arena-v0.3)

The pre-registered analysis behind the benchmark's headline result. Measurements come from the
evaluation harness ([HARNESS.md](HARNESS.md)) and from closed-loop runs with `run.py`. The task,
rows and contract are in [README.md](README.md) and [PARTICIPANTS.md](PARTICIPANTS.md).

**Status: draft of 2026-10-27.**
- This file is frozen with tag `arena-v0.3` before any `test` run, the reference rows included.
- A change made after any `test` result has been seen needs a new tag, and the paper labels it as
  made after seeing results.
- `dev` runs never choose analysis settings.

## 1. Question

Does the cross-agent counterfactual layer (layer 2) explain closed-loop differences between learned
world models that open-loop prediction (layer 1) misses? How does that change with latency and with
moving obstacles?

## 2. Models

- **Primary set: 30 models.**
  - Rows: the learned world models in the predictor interface, namely entity-token JEPA `full`,
    `no_cross` and `flat`, W-ar and PETS.
  - Each row uses its selected configuration (PARTICIPANTS.md, Section 6) and trains 3 new seeds on
    the standard 2,000 episodes, plus 3 seeds on the first 500 episodes (seeds 200000–200499).
- **Sensitivity set.** The primary set plus the physics reference, physics + mass estimation and the
  example MLP. Physics + mass estimation is near-perfect on every layer and could create a strong
  correlation on its own, so it stays out of the primary set.
- **Not included.**
  - U, because it defines the truth.
  - CrazyMARL, because it has no predictor.
  - The kinematic test fixture.
  - TD-MPC2 and DreamerV3, which have no state probes yet.
- **No exclusions on results.** Every model that completes the protocol is included, whatever its
  results. Failed episodes count as failures, and nothing is rerun to replace a result.

## 3. Measurements per model, on `test`

| Symbol | Quantity | Source |
|---|---|---|
| J | Closed-loop success rate over the 400 `test` episodes (50 layouts × 8 conditions), with the shared MPPI | `run.py` |
| L1 | Payload position error for U's 15-decision plan, averaged over the 15 steps and all questions | harness, layer 1 |
| L2 | Cross-agent CEE: ½ payload + ½ other aircraft, over A changes, averaged over changes and questions | harness, layer 2 |
| L3 | Executed-plan regret against U's executed plan, averaged over questions | harness, layer 3 |

- **Questions.** Every 5th decision of U's 400 `test` episodes (HARNESS.md, Section 1).
- **Stochastic predictors.** L2 uses common random numbers. L1 and L3 are averaged over 5 repeats.
- **Also reported for every model, with no decision rule:**
  - closed loop: mean time to goal over successes, and failure types;
  - layer 1: the validation selection score;
  - layer 2: each part of L2, the changed aircraft's own CEE, B changes, and relative CEE with its
    unresolved share;
  - layer 3: top-1 and top-8 elite regret, whole-pool Spearman and pairwise accuracy;
  - cost: the parameter count, training GPU-hours and the time of one planning call.

## 4. Primary test

- **Statistic.** The partial Spearman correlation between L2 and J across the 30 primary models,
  controlling for L1:

  ρ(L2, J | L1) = (ρ(L2, J) − ρ(L2, L1) ρ(L1, J)) / √((1 − ρ(L2, L1)²) (1 − ρ(L1, J)²)),

  from the Spearman correlations among L1, L2 and J. Ties take average ranks.
- **Interval.** A 95% percentile interval from 10,000 bootstrap resamples of the 50 `test` layouts.
  - A resampled layout brings its 8 episodes and its questions.
  - Every model's J, L1 and L2 are recomputed on each resample, so models and layers stay paired.
  - The set of models is fixed: the inference is about these 30 models.
- **Decision rule.**
  - **The interval lies entirely below 0: supported.** Layer 2 explains closed-loop differences that
    layer 1 misses; higher cross-agent error goes with lower success.
  - **The interval contains 0: not established.** This is reported as insufficient evidence, not as
    evidence that there is no effect.
  - **The interval lies entirely above 0:** reported as an effect in the unexpected direction.
- **Degenerate outcome.** If J takes fewer than 5 distinct values among the 30 models, for example
  because nearly every model fails, the test is reported as uninformative.
- **Power.** In a simulation with Gaussian data, 30 models detect a true partial correlation of
  0.43–0.48 in 67–79% of runs. Spearman ranks and the dependence between seeds of one row lower this.
  The paper states it.

## 5. Secondary analyses

All are reported with intervals and none has a decision rule.
- **Layer 3** analysed the same way: ρ(L3, J | L1).
- **Plain correlations.** ρ(L1, J), ρ(L2, J) and ρ(L3, J), and the difference ρ(L2, J) − ρ(L1, J).
- **Per condition** (2 obstacle settings × 4 latencies). J and L1–L3 per condition, and the primary
  statistic per condition. This answers how the relation changes with latency and moving obstacles.
- **Cross-agent beyond action sensitivity.** The correlation of L2 with J, controlling for both L1 and
  the changed aircraft's own CEE.
- **Each part of L2 alone.** The payload, and the other aircraft.
- **Data size.** Each row's 500-episode models against its 2,000-episode models.
- **Structure.** `full`, `no_cross` and `flat` compared on J and L2, paired by seed and data size.
- **Sensitivity set.** The primary test repeated on it.

## 6. Reporting

- Every model, seed and data size is reported. No seed is chosen after the fact.
- Every number has a 95% bootstrap interval over layouts.
- Each result records the arena tag, the harness commit, the question-set manifest and each
  checkpoint's hash.

## 7. Not decided here

- Who runs the harness for each model.
- Whether TD-MPC2 and DreamerV3 join through state probes.
