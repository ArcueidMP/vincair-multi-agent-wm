# Evaluation harness: layers 1–3 (draft)

How the arena scores a predictor on evaluation layers 1–3 at recorded decision states, against the
true simulator. Layer 4, the closed loop, stays with `run.py`. The analysis that uses these numbers
is specified in [ANALYSIS.md](ANALYSIS.md).

**Status: draft of 2026-10-27; nothing here is implemented yet.**
- The maintainer maintains the harness. It is not part of the frozen contract.
- The definitions of the quantities it reports are fixed by [ANALYSIS.md](ANALYSIS.md), which is
  frozen with `arena-v0.3`. A harness change may fix a bug or make it faster. Changing what a
  quantity means needs a new tag of ANALYSIS.md.
- Every result records the harness commit that produced it.

## 1. Questions: decision states flown by U

- **Source.** The oracle U (`TrueSimPredictor` with the shared MPPI) flies `dev` for development and
  `test` for final results, in all 8 conditions. Every model is scored on the same questions.
- **Questions.** Every 5th decision of every recorded episode (k = 0, 5, 10, ...), about 5,400 on
  `test`. Observations are recorded at every decision, because a stateful predictor has to observe
  the whole episode up to a question.
- **Recorder** (`record.py`, new). It runs the planner loop of `run.py` unchanged and writes the same
  per-episode records, so U's run on `test` is both the reference row and the source of the
  questions. At every decision it also stores:
  - the observation, the public layout and the condition;
  - U's 15-decision plan as the planner clipped it, and the warm start it was solved from;
  - the state of the solver's random generator before the solve, so the 512 candidates U scored can
    be regenerated exactly;
  - at questions, the physics snapshot `ArenaEnv.physics_state()`: the plant, the onboard controller
    memory and the setpoints, which is what `simulate` needs for exact rollouts.
- **Frozen files stay untouched.** The recorder wraps the solver instead of editing `planning.py` or
  `run.py`. A test checks that recorded episodes reproduce `run.py`'s records exactly.
- **Order.** The recorder has to exist before the reference rows run on `test`.
- **Static conditions.** U plans from the true state, so its static episodes are identical at all four
  latencies (checked on `dev`, seeds 0–4). Their questions share a snapshot and differ only in what the
  predictor observes. Ground truth is computed once per distinct snapshot.
- **Storage.** One file per episode under `data/uav_arena/states/<suite>/`, with content hashes and a
  manifest as in `collect.py`. About 0.5 GB for `test`, ground truth included.

## 2. Calling a predictor

The harness uses only the participant interface ([PARTICIPANTS.md](PARTICIPANTS.md), Section 2):
- For each episode it calls `reset()`, then `observe(obs, layout)` at every decision, in order.
- At a question it calls `predictor(info, setpoints)` once, with
  `info = info_from_observation(obs, anchor)` and all of the question's sequences stacked on the
  leading axis. These are U's plan, the 42 layer-2 changes and the 512-candidate pool, so the leading
  shape is (1, 555).
- Predictors never see snapshots, true states or the payload mass. The information rules of
  PARTICIPANTS.md, Section 3, apply unchanged.
- **Stochastic predictors** are those whose output is random for a fixed input. They declare
  `stochastic = True` and draw their randomness only from torch's global generator.
  - Layer 2 uses common random numbers. The harness calls U's plan and each change as single sequences
    and sets the same seed before every call, so a change and its base see the same draws.
  - Layers 1 and 3 call the predictor the way the planner does, 5 times with different seeds, and
    average the metrics.
  - A predictor that does not declare itself stochastic but returns different outputs on a repeated
    call is rejected.

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
    multiplies the noise by `var_scale`, which is 0.04. Each change has a fixed seed per question,
    aircraft and change.
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
- **Scale.** A probe on open-field states (2026-10-27) found these median true effects under A changes:
  13 cm on the changed aircraft, 5.8 cm on the payload and 2.2 cm on each other aircraft.

## 5. Layer 3: decision level

- **Pool.** The 512 candidates U's MPPI scored at that decision. Candidate 0 is U's warm start, the
  previous plan padded with zeros. The other 511 are the warm start plus noise × 0.04, regenerated from
  the recorded generator state. Every model scores the same pool.
- **Costs.**
  - Predicted cost Ĉ: the shared cost (`costs.soft_cost`) on the model's predictions.
  - True cost C: the same cost on trajectories from `simulate`, started from the snapshot, with the same
    obstacle information and the same constant-velocity forecasts.
- **A model's executed plan.** The MPPI update on its own costs, as the frozen planner computes it:
  softmax weights exp(−(Ĉ − min Ĉ)/0.05) over all 512 candidates, the weighted mean of the candidates,
  clipped to ±0.12 m. Its true cost needs one more rollout. For U this reproduces its recorded plan.
- **Executed-plan regret** (primary). C(model's executed plan) − C(U's executed plan). It is zero for
  U, can be slightly negative, and is the layer-3 number of ANALYSIS.md.
- **Elite regret, top-1 and top-8.** The mean true cost of the model's k best candidates, minus the
  pool's true best.
- **Also reported.** The Spearman correlation between Ĉ and C over the pool, and the pairwise accuracy
  P[sign(Ĉ_a − Ĉ_b) = sign(C_a − C_b)], which is the group meeting's ranking accuracy.
- **Why regret is primary.** On 12 `dev` states, MPPI's weights had a median effective sample size of
  2.9 (range 1–12). At a temperature of 0.05, most candidates sat 5–38 cost units above the best, so a
  ranking of the whole pool is dominated by obviously bad candidates. The executed plan also beat the
  pool's best single candidate in half of the states.

## 6. Outputs and checks

- **Outputs.** Per predictor and suite: per-question metrics and per-condition summaries, under
  `data/uav_arena/results/harness/`. Each file records the harness commit, the tag of ANALYSIS.md, the
  hash of the question-set manifest and the checkpoint hash.
- **Tests.**
  - Recorded episodes reproduce `run.py`'s records exactly.
  - U scored as a predictor (true state, exact rollouts) gets zero on every layer. So does the physics
    reference given the true mass.
  - The regenerated pool equals the pool U scored, bitwise.
  - Predictors never receive snapshots or the payload mass.
  - A predictor whose noise does not depend on the action gets the same layer-2 errors as its
    deterministic version.
  - Predictors answer leading shapes (1, 1) and (1, 555).

## 7. Compute

Estimated on 2026-10-27 for the maintainer's laptop.

| Part | Resource | Time |
|---|---|---|
| Recording U on `dev` seeds 0–4 | CPU, 20 workers | about 20 min (a rerun) |
| Recording U on `test` | CPU, 20 workers | 2.5–4 h, as part of the reference run |
| Ground truth on `test`, layers 2 and 3 | CPU, 20 workers | about 0.5 h |
| One deterministic learned model | GPU | about 2–3 min |
| One stochastic model (PETS) | GPU | about 4–6 min |
| The physics references and the example MLP | CPU | about 2–3 h together |

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

- **CODEOWNERS.** `ANALYSIS.md` is added.
- **README.md.** The status names `arena-v0.3`, and the evaluation-layers section points to
  HARNESS.md and ANALYSIS.md.

## 9. Open

- Who runs layers 1–3 for each model: the maintainer centrally, or each participant.
- Whether TD-MPC2 and DreamerV3 join layers 1–3 through state probes.
