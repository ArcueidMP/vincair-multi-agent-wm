# Evaluation harness: layers 1–3

How the arena scores a predictor on evaluation layers 1–3 at recorded decision states, against the
true simulator. Layer 4, the closed loop, stays with `run.py`. The analysis that uses these numbers
is specified in [ANALYSIS.md](ANALYSIS.md).

**Status: implemented on 2026-10-27 in `record.py` and `harness.py`, tested in
`tests/test_uav_arena_harness.py` and validated on `dev` seeds 0–4 (Section 7).**
- The maintainer maintains the harness. It is not part of the frozen contract.
- The definitions of the quantities it reports are fixed by [ANALYSIS.md](ANALYSIS.md), which is
  frozen with `arena-v0.3`. A harness change may fix a bug or make it faster. Changing what a
  quantity means needs a new tag of ANALYSIS.md.
- Every result records the harness commit that produced it.

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.record --suite dev --seeds 0:5 \
  --out data/uav_arena/states/dev --results data/uav_arena/results/oracle_mppi_dev_s0-5_recorded.json
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.harness truth --states data/uav_arena/states/dev
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.harness score --predictor physics \
  --states data/uav_arena/states/dev --out data/uav_arena/results/harness/physics_dev.json
```

`score` takes the same `--predictor` and `--predictor-options` as `run.py`, so a participant's
`module:factory` runs unchanged.

## 1. Questions: decision states flown by U

- **Source.** The oracle U (`TrueSimPredictor` with the shared MPPI) flies `dev` for development and
  `test` for final results, in all 8 conditions. Every model is scored on the same questions.
- **Questions.** Every 5th decision of every recorded episode (k = 0, 5, 10, ...), about 5,400 on
  `test`. A stateful predictor observes every decision of the episode up to a question.
- **Recorder** (`record.py`). It flies U through `run.run_episode` and the frozen planner, unchanged:
  wrappers around the planner's MPPI solver and its cost only read what each solve uses and returns.
  Its records are those `run.py` writes, so U's run on `test` is both the reference row and the
  source of the questions. A file per episode extends `collect.py`'s format: the true 10 ms state log
  and every setpoint sent, from which `openloop.observation` rebuilds the observation of any decision
  bit for bit. It adds:
  - U's 15-decision plan at every decision, as the planner clipped it;
  - at questions, the physics snapshot U planned from: the plant's integration state, the onboard
    controller memory and the setpoints in force, which is what `simulate` needs for exact rollouts;
  - at questions, the state of the solver's random generator before the solve, so the 512 candidates
    U scored can be regenerated exactly (Section 5);
  - at questions, U's cost of each of those candidates. U's predictions are exact rollouts from the
    true state, so these are the candidates' true costs.
- **Frozen files stay untouched.** Tests check that recorded episodes reproduce `run.py`'s records,
  and that the rebuilt observations, the regenerated candidates and rollouts from the stored
  snapshots equal those of the live run bit for bit.
- **Order.** The recorder has to exist before the reference rows run on `test`.
- **Static conditions.** U plans from the true state, and with no movers its cost does not depend on
  the latency, so its static episodes are identical at all four latencies. Their questions share
  snapshots, plans and candidates and differ only in what the predictor observes. The layer-2
  changes are seeded without the latency, so they share those too.
- **Storage.** One file per episode under `data/uav_arena/states/<suite>/`, named by seed and
  condition, with a manifest of content hashes as in `collect.py`. The ground truth goes to
  `truth/` beside them and records the content hash of the questions it was computed for.

## 2. Calling a predictor

The harness uses only the participant interface ([PARTICIPANTS.md](PARTICIPANTS.md), Section 2):
- For each episode it calls `reset()`, then `observe(obs, layout)` at every decision, in order, with
  the observation the planner saw and the layout without the payload mass.
- At a question it calls `predictor(info, setpoints)` once, with
  `info = info_from_observation(obs, anchor)` expanded over the batch as the solver expands it, and
  all of the question's sequences stacked on the leading axis. These are U's plan, the 42 layer-2
  changes and the 512-candidate pool, so the leading shape is (1, 555).
- Predictors never see snapshots, true states or the payload mass. The information rules of
  PARTICIPANTS.md, Section 3, apply unchanged. As in `ArenaPlanner.act`, only a privileged predictor
  without `observe`, the oracle U, receives the snapshot through `set_state`; results record whether
  a predictor was privileged.
- **Stochastic predictors** are those whose output is random for a fixed input. They declare
  `stochastic = True` and draw their randomness only from torch's global generator.
  - Layer 2 uses common random numbers. The harness calls U's plan and each change as single sequences
    and sets the same seed before every call, so a change and its base see the same draws.
  - Layers 1 and 3 call the predictor the way the planner does, 5 times with different seeds, and
    average the metrics.
  - A predictor that does not declare itself stochastic is called twice at each episode's first
    question; if any output differs by more than 10⁻⁵ m, the run stops with an error.
- Outputs must have the shapes of the interface and be finite; otherwise the run stops with an error.

## 3. Layer 1: factual prediction on U's states

- **Sequence.** U's 15-decision plan at the question.
- **Truth.** `simulate` of that plan from the snapshot. This is an open-loop rollout: U itself replanned
  after one decision, so the recorded flight is not the truth here.
- **Reported.** Payload and aircraft position error, per step and averaged over the 15 steps, per
  condition. The payload error averaged over the 15 steps is the layer-1 number of ANALYSIS.md. The
  validation selection score of `openloop.py` is reported alongside.

## 4. Layer 2: cross-agent counterfactuals

- **Base.** U's 15-decision plan at the question.
- **Changes.** Each aircraft in turn gets changed increments, while the other two keep theirs.
  - **A, primary, 8 per aircraft.** Gaussian noise added to that aircraft's increments, with standard
    deviation 0.04 m per decision and axis. This is the shared MPPI's own sampling noise: swm 0.1.1
    multiplies the noise by `var_scale`, which is 0.04. The noise is drawn from a generator seeded by
    the layout, the obstacle setting and the decision.
  - **B, supplement, 6 per aircraft.** A 0.10 m offset along +x, −x, +y, −y, +z or −z, ramped in over
    the first 3 decisions and then held.
  - Increments are clipped to ±0.12 m, as the planner clips them.
- **Effects.** The predicted effect is Δŝ = f(changed) − f(base). The true effect is
  Δs = `simulate`(changed) − `simulate`(base), both from the snapshot.
- **Error per change.** ‖Δŝ_k − Δs_k‖ averaged over the 15 steps k, for three parts:
  - the payload;
  - the other aircraft, as the mean over the two unchanged ones;
  - the changed aircraft itself, a control for models that ignore the action pathway.
- **Cross-agent CEE.** ½ payload + ½ other aircraft, over A changes. This is the layer-2 number of
  ANALYSIS.md.
- **Relative CEE.** The error divided by the mean true effect. It is computed only where the mean true
  effect is at least 0.5 cm, and the share of changes below that threshold is reported.
- **Reported.** Per change kind (A, B), part, condition and step.
- **Scale.** On U's `dev` states (544 questions), the median true effect of an A change is 14.0 cm on
  the changed aircraft, 6.2 cm on the payload and 2.7 cm on each other aircraft; of a B change, 7.6,
  3.8 and 1.5 cm. Fewer than 0.2% of A changes fall below the 0.5 cm threshold. An earlier probe on
  open-field states found 13, 5.8 and 2.2 cm for A changes.

## 5. Layer 3: decision level

- **Pool.** The 512 candidates U's MPPI scored at that decision. Candidate 0 is U's warm start, the
  previous plan shifted by one decision and padded with zeros. The other 511 are the warm start plus
  noise × 0.04, regenerated from the recorded generator state. Every model scores the same pool.
- **Costs.**
  - Predicted cost Ĉ: the shared cost (`costs.soft_cost`) on the model's predictions.
  - True cost C: U's own cost of each candidate, recorded with the question: the same cost on exact
    rollouts from the true state, with the same obstacle information and the same constant-velocity
    forecasts.
- **A model's executed plan.** The MPPI update on its own costs, as the frozen planner computes it:
  softmax weights exp(−(Ĉ − min Ĉ)/0.05) over all 512 candidates, the weighted mean of the candidates,
  clipped to ±0.12 m. Its true cost needs one more rollout from the snapshot. For U this reproduces
  its recorded plan bit for bit.
- **Executed-plan regret.** C(model's executed plan) − C(U's executed plan), per question. It is
  zero for U and often slightly negative: MPPI's weighted mean is not the best plan its candidates
  allow, so a small change in the costs can improve it.
- **Costly decisions** (primary, the layer-3 number of ANALYSIS.md). The share of questions whose
  regret exceeds 1 cost unit, among the questions with a reliable reference: those where U's plan
  costs at most 1 cost unit more than the best candidate in its pool.
  - The other questions are left out for every model. There U's plan averages candidates that go
    different ways, its 15-step cost lies far above theirs, and the regret swings widely for any
    model (Section 7).
  - One cost unit is about a plan that cuts 1.7 cm into a safety margin, or makes about 0.6 m less
    progress over the horizon.
  - A stochastic predictor's share counts each of its 5 repeats.
- **Elite regret, top-1 and top-8.** The mean true cost of the model's k best candidates, minus that
  of the pool's k truly best. It is zero for U.
- **Also reported.** The mean and the median regret, the share of costly decisions over all
  questions and the share of unreliable references. The Spearman correlation between Ĉ and C over
  the pool, and the pairwise accuracy P[sign(Ĉ_a − Ĉ_b) = sign(C_a − C_b)], which is the group
  meeting's ranking accuracy.
- **Why the executed plan, not a ranking.** On 12 `dev` states, MPPI's weights had a median effective
  sample size of 2.9 (range 1–12). At a temperature of 0.05, most candidates sat 5–38 cost units above
  the best, so a ranking of the whole pool is dominated by obviously bad candidates. Over all 544 `dev`
  questions, U's executed plan beat the pool's best single candidate in 65% of them, and was more
  than 1 cost unit worse in 8%.

## 6. Outputs and checks

- **Outputs.** `score` writes a JSON summary, overall and per condition: the layer numbers of
  ANALYSIS.md (L1, L2, L3) and every reported diagnostic. Beside it, an `.npz` holds the metrics of
  every question with its layout seed and condition, which the bootstrap over layouts resamples:
  among them its regret, whether the decision was costly and whether the reference is reliable. The
  JSON records the harness commit (`git describe`), the SHA-256 of the question-set manifest, the
  checkpoint's SHA-256 and whether the predictor was stochastic or privileged.
- **Tests** (`tests/test_uav_arena_harness.py`, on short episodes with a small planner).
  - Recorded episodes reproduce `run.py`'s records exactly.
  - Rebuilt observations, regenerated pools and rollouts from stored snapshots equal those of the live
    run, bit for bit.
  - U scored as a predictor gets exactly zero on every layer, with Spearman and pairwise accuracy 1.
    The physics reference given the true mass stays below 10⁻⁵ m on layers 1 and 2 and 10⁻⁴ on
    layer 3.
  - Predictors receive observations and the public layout only, and never the snapshot, whatever
    they claim.
  - The call shapes: (1, 1 + 42 + pool) once per question, and for a stochastic predictor 43 calls of
    (1, 1) plus 5 of the full batch.
  - A predictor whose noise does not depend on the action gets the same layer-2 errors as its
    deterministic version; undeclared randomness is rejected.
  - Each change alters exactly one aircraft, and the changes repeat for the same question.

## 7. Validation on `dev` and compute

U flew `dev` seeds 0–4 through the recorder on 2026-10-27: 40 episodes, 2,680 decisions and 544
questions.
- **Recorder.** All 40 records are identical to `run.py`'s reference row at `arena-v0.2`
  (`oracle_mppi_dev_s0-5_v3.json`) in every field but the timing. The static episodes are identical
  at all four latencies (15 of 15 pairs).
- **Scores** of the reference rows and the example participant, over all 544 questions:

| | L1 payload | L2 cross-agent | L2 payload / others / own | L3 costly decisions | Regret, mean (median) | Top-1 | Spearman | Pairwise | `dev` closed loop |
|---|---|---|---|---|---|---|---|---|---|
| U (oracle) | 0 | 0 | 0 / 0 / 0 | 0 | 0 (0) | 0 | 1 | 1 | 40/40 |
| Physics + mass estimation | 0.35 cm | 0.30 cm | 0.28 / 0.32 / 0.59 cm | 1.8% | −0.57 (−0.0002) | 0.04 | 0.97 | 0.96 | 40/40 |
| Physics (nominal 25 g) | 4.31 cm | 2.07 cm | 2.18 / 1.96 / 3.87 cm | 19.6% | +1.17 (+0.02) | 3.94 | 0.71 | 0.79 | 28/40 |
| Example MLP | 14.55 cm | 4.54 cm | 4.98 / 4.10 / 7.76 cm | 44.3% | +9.78 (+0.38) | 17.51 | 0.37 | 0.64 | 7/40 |

- **U scores exactly zero** on every question, with Spearman correlation and pairwise accuracy
  exactly 1: the costs, plans and rollouts the harness recomputes equal the recorded ones.
- **The layers order these rows as the closed loop does.** So does layer 1, so `dev` cannot separate
  layer 2 from layer 1. That is the job of the 30 learned models on `test`.
- **The example MLP leaves its training distribution on U's states.** Its validation selection score
  is 1.46 cm, half the nominal physics reference's 2.93 cm. On U's states its layer-1 error is
  14.6 cm, 3.4 times the physics reference's 4.3 cm. `openloop.py` along U's flown setpoints gives the
  same picture: 10.8 cm on the static 0 ms episodes, against 1.1 cm on `explore-v1` validation
  episodes. The plans U flies lie outside what `explore-v1` covers.
- **Layer 3's mean is dominated by a few questions.** In 43 of the 544 questions (7.9%), U's own
  executed plan is more than 1 cost unit worse than the best candidate in its pool. At the worst one,
  the two candidates carrying 99% of the weight cost 3.6 and 3.7, and their weighted mean costs 219:
  the averaged plan brings two aircraft within 5.9 cm of each other over its 15 open-loop steps. Only
  its first increment is flown, and U succeeds. At two such questions, physics + mass estimation
  scores regrets of −216 and −100, which puts its mean below zero; without them its mean is +0.05.
  - That is why layer 3 counts costly decisions on the questions with a reliable reference
    (Section 5; decided on 2026-10-27 and recorded in ANALYSIS.md, Section 8). On the 501 such
    questions the share is 0, 1.8%, 19.6% and 44.3% for the four rows.
  - Other summaries tried: the share over all questions (0, 2.4%, 19.3%, 45.2%), the mean over the
    reliable questions (0, 0.05, 2.02, 10.80) and the median (0, −0.0002, 0.017, 0.375). The median
    puts physics + mass estimation a hair ahead of U: small negative regrets are common (57% of its
    questions).
  - In a crude bootstrap over the 5 `dev` layouts, the share over all questions varied least
    relative to its value: 0.24 for the physics reference, against 0.77 for the mean.
  - The physics reference's share of costly decisions rises from 11–16% in the static conditions to
    20–29% with movers.
- **Compute** on the maintainer's laptop (Ryzen 9 7845HX, 24 threads):

| Part | Workers | `dev`, measured | `test` (10 times the layouts), estimated |
|---|---|---|---|
| Recording U | 20 | 24 min (0.53 s per decision) | about 4 h, as U's reference run |
| Ground truth | 20 | 36 s | about 6 min |
| Example MLP on CPU | 20, for rollouts | 15 s (2.5 ms per call) | about 3 min |
| U and the two physics references, in parallel | 7 each | 18–20 min each (1.6 s per call of 555 rollouts) | about 3.5 h |
| Storage | | 30 MB: questions 16 MB, ground truth 15 MB | about 0.3 GB |

A stochastic predictor makes 48 calls per question instead of one: 43 single sequences and 5 full
batches.

## 8. Contract changes for `arena-v0.3`

These changes are on the branch since 2026-10-27. They take effect with the tag, which is created
once the recorder and the harness are validated on `dev`, before the first `test` run.
- **PARTICIPANTS.md, Section 1.** The tag also freezes ANALYSIS.md. The code files are unchanged, so
  the reference rows' results carry over. The harness is not frozen.
- **PARTICIPANTS.md, Section 2.**
  - Predictors accept any leading shape; the harness calls them with (1, 1) and (1, 555).
  - Stochastic predictors declare `stochastic = True` and use only torch's global generator.
  - They reseed it in `reset()` from a fixed seed, so each episode replays alone exactly. `run.py`
    reseeds only the solver's own generator.
- **PARTICIPANTS.md, Section 3.** `reset()` also resets the random generator.
- **PARTICIPANTS.md, Sections 5–6.** After selection, the chosen configuration also trains 3 seeds on
  the first 500 standard episodes (seeds 200000–200499), and each runs once on `test`.
- **PARTICIPANTS.md, Section 7.** Layers 1–3 come from this harness, ANALYSIS.md is the analysis
  plan, and branches start from the files of `arena-v0.3`.
- **TASKS.md.** The extra 500-episode runs for each row, tests for the new call shapes, the PETS
  card's randomness rule, the maintainer's row and the Phase B compute estimate.
- **CODEOWNERS.** `ANALYSIS.md`, `record.py` and `harness.py` are added.
- **README.md.** The status names `arena-v0.3`, and the evaluation-layers section points to
  HARNESS.md and ANALYSIS.md.

## 9. Open

- Who runs layers 1–3 for each model: the maintainer centrally, or each participant.
- Whether TD-MPC2 and DreamerV3 join layers 1–3 through state probes.
