# Analysis plan (draft for arena-v0.3)

The pre-registered analysis behind the benchmark's headline result. Measurements come from the
evaluation harness ([HARNESS.md](HARNESS.md)) and from closed-loop runs with `run.py`. The task,
rows and contract are in [README.md](README.md) and [PARTICIPANTS.md](PARTICIPANTS.md).

**Status: draft of 2026-10-27.**
- This file is frozen with tag `arena-v0.3` before any `test` run, the reference rows included.
- A change made after any `test` result has been seen needs a new tag, and the paper labels it as
  made after seeing results.
- `dev` runs never choose analysis settings, with one exception: a measurement problem found while
  validating the harness may be fixed before any `test` run. Each such fix is listed in Section 8
  with its date and evidence.

## 1. Question

Does the cross-agent counterfactual layer (layer 2) explain closed-loop differences between learned
world models that open-loop prediction (layer 1) misses? How does that change with latency and with
moving obstacles?

## 2. Models

- **Primary set: 30 models.**
  - Rows: the learned world models in the predictor interface, namely entity-token JEPA `full`,
    `no_cross` and `flat`, W-ar and PETS.
  - Each row uses its selected configuration (PARTICIPANTS.md, Section 6) and trains 3 new seeds on
    the standard 2,000 `explore-v2` episodes, plus 3 seeds on the first 500 of them (seeds
    200000–200499).
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
| L3 | Share of costly decisions: questions where the executed plan's regret against U's executed plan exceeds 1 cost unit, among questions with a reliable reference | harness, layer 3 |

- **Questions.** Every 5th decision of U's 400 `test` episodes (HARNESS.md, Section 1).
- **L3 in detail.** A question's regret is the true cost of the plan the frozen MPPI executes on the
  model's costs of U's candidates, minus the true cost of U's executed plan (HARNESS.md, Section 5).
  - A decision is costly when its regret exceeds 1 cost unit. That is about a plan that cuts 1.7 cm
    into a safety margin, or makes about 0.6 m less progress over the 1.5 s horizon.
  - A question has a reliable reference when U's own executed plan costs at most 1 cost unit more
    than the best of its candidates. The other questions, where U's plan averages candidates that
    go different ways, are left out of L3 for every model alike.
- **Stochastic predictors.** L2 uses common random numbers. L1 and L3 are averaged over 5 repeats;
  for L3, the share is taken over questions and repeats.
- **Also reported for every model, with no decision rule:**
  - closed loop: mean time to goal over successes, and failure types;
  - layer 1: the validation selection score;
  - layer 2: each part of L2, the changed aircraft's own CEE, B changes, and relative CEE with its
    unresolved share;
  - layer 3: the mean and the median regret and the share of costly decisions over all questions,
    the share of questions without a reliable reference, top-1 and top-8 elite regret, whole-pool
    Spearman and pairwise accuracy;
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

## 8. Changes before any `test` run

Fixes under the exception in the status above:
- **2026-10-27, L3.** L3 was the mean regret over all questions. It is now the share of costly
  decisions among questions with a reliable reference (Section 3).
  - Evidence: the harness's validation on `dev` (HARNESS.md, Section 7). In 8% of the questions,
    U's own executed plan cost more than 1 cost unit above its best candidate, because MPPI's
    weighted mean averaged candidates that went different ways. On such questions even the
    near-perfect physics + mass estimation scored regrets of −216 and −100. Its 5 largest regrets
    made up 97% of its summed regret.
  - The decision used only U, the physics references and the example MLP on `dev`. No relation
    between L3 and J was examined.
- **2026-10-28, training data.** The standard datasets are `explore-v2` instead of `explore-v1`,
  with the same seeds; the validation set switches too (PARTICIPANTS.md, Section 5).
  - Evidence: on U's `dev` flights, `explore-v1` left 47–70% of U's per-aircraft increments,
    differences between aircraft and changes of increment beyond its 99th percentile (`coverage.py`).
    The example MLP trained on it had a layer-1 error of 14.6 cm on U's states, against 1.46 cm on
    its validation set, and 7 `dev` successes of 40. Trained on `explore-v2`: 10.8 cm and 21 of 40.
  - A first version of `explore-v2` redrew each aircraft's offset at every decision. It covered the
    per-step statistics, but consecutive increments became anti-correlated (lag-1 autocorrelation
    −0.47, against U's −0.04). The example MLP trained on it did worse than on `explore-v1` (17.5 cm,
    2 of 40), so it was discarded before any model was trained for the analysis.
  - PETS in june's Phase A configuration also improved on every layer (layer 1 on U's states from
    24.3 to 20.5 cm). With its default Gaussian sampling it still failed all 40 `dev` episodes;
    with sampling off, the same checkpoint succeeded in 6 (HARNESS.md, Section 7).
  - The choice used the example MLP and PETS on `dev` and U's `dev` flights. No model has been
    trained for the analysis, and no `test` run has been made.
- **2026-10-28, the PETS row.** PETS propagates each candidate with the mean of its randomly drawn
  ensemble member, without Gaussian sampling (TASKS.md, PETS card). The member draw stays random,
  so PETS remains a stochastic predictor.
  - Reasons: the validation selection score (`--stride 10`), 9.42 cm without sampling against
    18.6 cm with it on `explore-v2` and 3.04 against 5.32 cm on `explore-v1`; and the shared
    planner scores one trajectory per candidate, so nothing averages the samples as published PETS
    averages its cost over 20 particles, and for that single trajectory a sample is a noisier
    estimate than the mean.
  - Disclosure: the `dev` diagnostic had been seen before the decision. With the same `explore-v2`
    checkpoint, sampling gave 0 of 40 `dev` successes and no sampling 6, with layer 1 at 20.5
    against 14.0 cm, layer 2 at 4.44 against 3.84 cm and costly decisions at 46.0% against 38.8%
    (HARNESS.md, Section 7). The `dev` result is not among the reasons.
  - No model has been trained for the analysis, and no `test` run has been made.
