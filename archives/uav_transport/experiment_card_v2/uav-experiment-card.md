# UAV experiment card: cross-agent dynamics for cooperative transport

**Version 2.0 — 2026-10-26. Status: development protocol; formal evaluation is not yet frozen.**

This revision replaces the two-week v1.1 draft. Progress is governed by validation
gates and evidence quality. There is no two-week completion deadline. The official
multi-UAV payload controller **L** and the autoregressive state predictor **W-ar**
are mandatory. A self-built controller cannot replace L in the formal comparison.
All primary arms use **500 Hz flight feedback**. Historical 50 Hz evidence remains
an immutable, separately labeled comparison.

The intended contribution is actionable: determine which cross-agent action
effects a world model must preserve for cooperative aerial planning, and test
whether measuring those effects improves prediction of closed-loop performance
and selection of models for new transport tasks. These are hypotheses to test;
this card records no new experimental result.

## 1. Questions and claims

**H1a — additional information from cross-aircraft effects.** Among models capable
of representing coupling, prediction error for the effect of one aircraft's
action on the *other aircraft* improves prediction of closed-loop performance
beyond task-matched factual error, absolute counterfactual prediction error,
own-aircraft effect error, and payload effect error. Section 8 specifies the
analysis and its validation. A positive result supports a cross-agent diagnostic
claim even after accounting for task relevance and cancellation of shared error.

**H1b — decision diagnostic reference.** Elite regret is evaluated as a secondary
planning-facing diagnostic. Its association with performance is measured, not
assumed. H1b alone does not satisfy H1a or establish a cross-agent-specific result.

**H2a — value of payload-to-aircraft feedback in the predictor.** The star-mediated
`no_cross` model improves the registered closed-loop score over `indep`, which
retains a joint payload predictor but removes incoming inter-entity information
from aircraft prediction tokens.

**H2b — sufficiency of star-mediated message passing.** `full` and `no_cross` have
equivalent registered scores within a prespecified 10% relative margin. This
tests a structural inductive bias in the stated plant. The payload latent can
relay general information; its name does not impose a physical causal bottleneck.

**Model-selection validation.** Before test evaluation, select models using each
development diagnostic and compare their transport performance on the same held-out
tasks. This provides an operational test of diagnostic usefulness in addition to
regression and correlation. Selection rules and ties are frozen in advance.

The H1 primary population contains `full`, `no_cross`, `flat`, and W-ar. `indep`
remains a mandatory structural ablation. With correctly isolated aircraft tokens,
its predicted cross-aircraft response is zero; normalized error is one when the
true response is nonzero. This is an implementation check, not a discovery.

## 2. Platform, information, and clocks

Use `uav_transport.menagerie.MenagerieTransportEnv`: three official Crazyflie 2
airframes, each 0.027 kg with the pinned Menagerie inertia and geometry; a
centrally attached spherical payload; three 0.6 m tension-only cables; a 30 ms
motor time constant; and a 0.16 N per-rotor limit. The cooperative scene and task
are constructed in this repository. Menagerie supplies the airframe assets.
The present plant includes MuJoCo fluid drag, with no wind or downwash model.

| Quantity | Primary protocol |
|---|---|
| Native flight feedback, every learned arm and U | 500 Hz, fresh physical state every 2 ms |
| Official L feedback and its payload control call | 500 Hz, firmware time in milliseconds |
| MPPI action and replanning period | 10 Hz, one action block every 100 ms |
| Learned-model transition | 100 ms |
| Nominal integration timestep | 2 ms |
| Integration refinements | 1 ms and 0.5 ms, feedback still every 2 ms |
| Native calls per 100 ms action block | 50 per aircraft at every integration resolution |

A 20 ms software logging/container interval may be reused, but it must not reduce
feedback to 50 Hz. Actual rotor commands are recorded at their 2 ms update times.
Timing is established by timestamps and update counts, not a configuration label.
The existing [frequency audit](lee-controller-frequency-audit.md) supplies the
tested multirate scheduling and replay pattern. Its shared payload allocator is
not part of the new learned/U action interface.

For learned arms and U, each aircraft receives a world-frame position offset
relative to its own position at the start of an action block. Each coordinate
is bounded by +/-0.3 m. Convert it once to an absolute setpoint and hold that
setpoint for 100 ms; do not re-anchor it to the moving aircraft every 2 ms.
Target velocity is zero. The same upstream `NativeLee` feedback and bounded
rotor mixer execute every learned/U action.

Nominal cable-load feedforward uses 25 g and is identical in learned/U arms.
For a positive upward cable component, the proposed rule is
`T_i = m_nominal * g / (3 * q_i,z)` and `a_ff,i = T_i * q_i / m_uav`, with
`q_i` directed from the payload to aircraft i. Its bounded behavior at shallow
angles, slack cables, and actuator saturation must be specified, validated,
and frozen in the adapter; no true payload mass enters this rule. Feedforward
geometry uses fresh state at 500 Hz in every learned/U arm; the absolute position
setpoint remains held for its 100 ms action block.

Keep true plant parameters separate from parameters visible to a controller or
predictor. W-ar, JEPA variants, and nominal L have the same known airframe,
cable, actuator, state-observation and nominal-load information. True payload
mass is available only to the explicitly privileged U-true and L-true arms.
U additionally restores complete motor and controller state for exact rollouts.
The information inventory must disclose that privilege; a finite history is
not assumed to identify all hidden motor, integral, or mass states exactly.

### Tasks and data splits

- Point-to-point payload transport from hover: horizontal displacement
  0.5--1.5 m with random direction, vertical displacement +/-0.3 m, duration 10 s.
  Payload workspace is x,y in [-2,2] m and z in [0.9,1.5] m.
- Payload mass is uniformly sampled from 10--40 g. Initial histories are produced
  by physical simulation using declared nominal-information controllers.
- Start with 20 development tasks and 50 test tasks. All methods share tasks,
  full initial physical states, exogenous random streams, and time budgets.
  Training, validation, development, and test trajectory seeds are disjoint.
- Task generation and stratification across mass, direction and distance are
  written before pilot inspection. Test tasks are held out until formal freeze.
  Counts can increase through the pre-test power gate in Section 8.
- A task revision requires a new development-protocol version and a recorded
  physical or coverage rationale applied to every method. Do not shrink the
  formal distribution solely until one preferred model passes a threshold.

## 3. Mandatory reference methods

| Arm | Required implementation and role |
|---|---|
| W-full / W-nocross / W-indep / W-flat | Learned entity/flat models; structural comparisons |
| **W-ar** | Mandatory direct state-increment model with full entity attention and MPPI |
| **U-true** | Exact plant and controller rollouts, true mass, same MPPI interface and candidate budget |
| **L-nominal** | Mandatory official multi-UAV payload controller, nominal 25 g load |
| **L-true** | Same official controller with true mass, explicitly privileged reference |

U-true is a true-dynamics planning reference. Finite sampling and a finite horizon
do not make it a mathematical upper bound. If another arm performs better, retain
the result and audit matched candidates, timing, objective and search behavior.

The official L implementation is `controller_lee_payload.c`, from firmware
revision `b08d9f3cbcc581d8e7f988cabad22c26d35f4a58`, called through its Python
bindings. Use one controller state per aircraft and the common bounded rotor
mixing/actuator path. It receives a minimum-jerk payload reference with velocity
and acceleration feedforward. Log its actual QP executions and force allocation.

### Required L adapter validation

The upstream defaults describe a different airframe/load: 35.6 g aircraft,
10 g payload, different inertia, zero payload-position integral limit, and
0.1 m avoidance radius. A **calibrated default-gain** run must first set this
platform's known mass, inertia, cable lengths, attachment IDs and nominal payload.
Then report both calibrated default gains and development-tuned gains. Include
integral limits, avoidance margin and trajectory duration in the tuning space.

The upstream test uses an older call signature and does not exercise the required
three-aircraft QP. Retain it as provenance and add a version-correct integration
test with three explicit controller instances. The acceptance suite must verify:

1. Correct API signatures, 3-UAV QP branch, force/torque units and gyro units;
   point-payload mode selected using the firmware's payload-quaternion sentinel;
   self at `team_state[0]`; complete ID matches for attachments and cable inputs.
2. Millisecond ticks 0,2,4,... at 500 Hz; fine integration steps do not introduce
   extra feedback calls or reuse uninitialized outputs from skipped ticks.
3. Matched state/reference outputs against a pinned official col-trans example,
   with numerical tolerances registered before comparison and differences retained.
4. Complete controller/solver state handling, independent instance behavior,
   reset repeatability, and checks for QP workspace contamination across tasks.
5. Hover and 1 m transfers on this platform with true mass, then nominal mass
   across the development mass range; finite output, actuator limits, and actual
   tracking/safety outcomes. Document justified tuning and remaining failures.

Official L remains mandatory if integration takes longer than expected. Keep
debugging and record the formal comparison as incomplete until the adapter gate
passes. **No fallback to a self-built load allocator is allowed.** Existing
`ExternalLeeController` results remain historical or supplemental engineering
references and cannot occupy L's row.

### Mass-information check

On development tasks, compare exact-rollout U with true and nominal 25 g mass,
keeping search settings, initial measured state and controller memory matched.
Construct nominal-model states explicitly; the existing snapshot guard correctly
rejects arbitrary mass changes and must not be silently bypassed.
Declare a small mass-information effect only if the prespecified confidence
interval for the paired relative score difference lies wholly inside +/-5%.
An interval containing zero is not an equivalence result. Report this check
without searching for a mass range that makes a preferred method win.

## 4. Models, data, and training fairness

Reuse the EntityJEPA backbone and SIGReg implementation while implementing the
UAV feature and action mappings explicitly. Aircraft tokens contain position,
velocity, the first two rotation-matrix columns, and body angular velocity
(15 features); the payload token contains position and velocity (6 features).
Actions enter their corresponding aircraft tokens. Use training-only feature
normalization and a common observation/history contract for every learned arm.

Use **six state frames and five intervening action blocks**, spanning an actual
0.5 s history at 10 Hz. The training rollout is five transitions (0.5 s), and
validation includes the full 15-transition (1.5 s) planning horizon. Store exact
state/action alignment and test it independently.

| Variant | Information flow | Interpretation |
|---|---|---|
| full | All entity tokens communicate | Full coupling-capable model |
| no_cross | Aircraft read self and payload; payload reads every aircraft | Star-mediated message passing |
| indep | Aircraft read only self; payload reads every aircraft | Removes feedback into aircraft prediction, retains joint payload prediction |
| flat | Entities concatenated per frame | Unstructured joint representation |
| W-ar | Full entity communication, autoregressive physical-state increments | Mandatory state-space learning reference |

Apply masks to every encoder/predictor layer and every rollout step. Test the
absence of forbidden action-to-output paths, including preprocessing and readout.
Only full/no_cross/indep are mask-only changes. Flat changes tokenization: match
its total latent budget to four 64-dimensional entity latents and report parameter
count, FLOPs, rollout time, and any remaining capacity differences.

Initial architecture: width 128, two encoder layers and four predictor layers.
JEPA uses latent prediction, SIGReg and physical readout losses. W-ar predicts
normalized state increments using multistep state loss, without latent prediction
or SIGReg; it uses the same history, full-attention backbone and training data.
Match capacity as closely as possible and disclose the resulting counts.

Collect 500 and 2,000 independent 10 s training trajectories plus 200 validation
trajectories. The 500-trajectory set is a frozen subset of the larger pool, common
to all variants. Split entire trajectories, never overlapping windows. Use smooth
formation motion with independent per-aircraft perturbations, record coverage
of individual and joint actions, and record rejected/failed collection episodes
with reasons. Pilot coverage changes apply to all methods before the data freeze.

The initial minimum is **30 fits: 5 variants x 3 seeds x 2 data budgets**. W-ar's
six fits are part of this minimum. The power gate may require more seeds, applied
symmetrically across variants and data budgets. Additional checkpoints from one
fit do not count as independent trained models.

Give each family the same declared training-tuning allowance, training-update
budget and data access. Select checkpoints using a common validation criterion
in decoded physical units at registered horizons, rather than comparing latent
loss magnitudes across objectives. Publish learning curves and the actual tuning
and training compute. Resolve implementation or optimization defects on development
data; retain every formally registered fit, including poor performers.

## 5. Planning and tuning fairness

The starting MPPI setting is 512 candidate sequences, 15 steps (1.5 s), one
iteration per 100 ms decision, and shifted-plan warm starts. Cost includes payload
goal distance squared, aircraft separation below 0.2 m, action size and action
change. Fix numerical cost weights and failure handling before formal evaluation.
Candidate failures use a frozen cost exceeding the largest admissible candidate
cost under those weights; the rule is identical for every predictor and truth
rollout. Keep this planning cost distinct from the mission score in Section 6.
Record actual model queries, simulated physics steps, memory, wall time, and
per-decision latency distributions. Report real-time capability only if measured.

Publish **both** panels:

- **Shared-planner panel:** identical cost, horizon, candidate count, candidate
  random streams and search hyperparameters across W variants and U. Choose the
  common search settings on development tasks with equal representation of all
  five learned variants and U, rather than JEPA-only selection. This panel supplies
  the primary structural and diagnostic comparisons.
- **Equal-tuning-budget panel:** each learned family and U receives the same
  registered search-tuning allowance; L receives an explicit controller/reference
  tuning allowance. Start with 12 configurations per method on the same development
  tasks. Search noise, temperature and reference/controller parameters may vary;
  scoring, physical limits, horizon and candidate budgets remain common. Do not
  tune away a failure through changed task definitions or cost weights.

Keep parameter spaces, trials, seeds, failed trials, selection criteria and ties
in the development ledger. If pilot evidence shows the tuning allowance is
inadequate, expand the applicable allowance symmetrically before formal freeze.
For W variants, choose settings across a balanced set of pilot seeds/data budgets,
then reuse them for the registered fits. Score tuning trials by failure count
first and mean registered mission score second.

Profile U before scheduling the complete study: persistent simulator instances,
batched candidate dispatch and cached truth rollouts can reduce overhead. Validate
the fast path against the instrumented path, including substep failure detection
and all planning costs; removing bookkeeping must not remove constraint checks.
A budget change is a development
revision applied to every applicable arm; latency is never a reason to omit L,
W-ar, or an inconvenient result. Keep equal-candidate and wall-time comparisons
separate, with the latter labeled separately if performed.

## 6. Mission scoring and numerical validity

Every episode has the same 10 s horizon and target. Tracking RMS is
`J_track = sqrt(mean_t ||payload_position(t) - goal||^2)`, accumulated in physical
time. Report final-2-second error, payload speed/swing, minimum aircraft separation,
peak cable tension, motor saturation time, energy/control effort, and planning time.

Hard failure includes an aircraft/payload height below 0.1 m, aircraft separation
below 0.1 m, unintended body collision, leaving the declared payload workspace,
or simulator nonfiniteness/invalid numerical state. Check at physics-step
resolution. Freeze contact tolerances and boundary conventions in the scorer.

The primary mission score **J** equals `J_track` for a nonfailed episode and
**6 m** for a failed episode. This fixed penalty exceeds the maximum goal error
inside the declared workspace (less than 5.69 m). Preserve pre-failure tracking
RMS, failure type and time, and publish failure rate beside J. Never replace
failure with initial distance, discard a failed run, or report successful-only
tracking as the sole mission result. Statistical normalization uses `Y = J / d0`,
where `d0` is the fixed initial goal distance of that task.

Task success additionally requires final-2-second mean goal error at most 0.1 m
and payload RMS speed at most 0.1 m/s, with no hard failure. These are provisional
development design values; any justified change is versioned before formal freeze.
Flight failures and compute/deadline misses are distinguished. Offline computation
does not consume simulated flight time unless a separately registered latency
experiment explicitly models it.

At development, compare fresh closed-loop runs at 2, 1 and 0.5 ms integration,
with all arms retaining 500 Hz feedback. Also replay the exact recorded 2 ms rotor
commands at finer integration resolution. Register tolerances and the rule for
choosing the formal integration step before model-outcome inspection. If 2 ms
does not satisfy that rule, use the validated finer step for every formal arm and
recollect/retrain the affected data/models before freeze.

Require bitwise nominal snapshot/replay where supported, report continuous
trajectory/metric differences and threshold-label changes, and retain every
exception. Preselect a stratified task/model subset for formal numerical replication
before results are known; add all numerical failures for diagnosis without
replacing that subset. A numerical defect cannot be handled by relaxing the
tolerance after seeing the winner.

## 7. Matched diagnostic comparisons

Build the common decision-state set D from the shared-planner U-true trajectory:
times 1,3,5,7 s in every task, initially 200 test states. Store complete plant,
motor, controller, clock, planner/RNG state and the preceding 0.5 s observation/
action history. Construct a separate development D for diagnostic tuning and
model selection. If U fails before a scheduled state, preserve the missing-state
record and use the frozen missingness analysis; do not silently substitute an
easier task or retain only successful U tasks. Register any additional state-
source robustness panel before test access.

### Action bank and paired interventions

At each state, cache U's complete candidate bank and true rollouts. Select a
common 64-sequence diagnostic subset using a frozen seed *independent of true
and predicted cost*. Retain the full-bank costs and membership. Using only the
64 best true sequences would change which prediction errors are being measured.

For factual error, replay the recorded subsequent 15 action blocks from exactly
that state, with feedback operating normally and the high-level actions held
fixed. For interventions, form A0 by projecting the first five blocks of this
reference into [-0.2,0.2] m per coordinate. Keep later blocks unchanged. Add
0.1 m to one aircraft/axis for those five blocks, using both signs and all three
axes: 18 legal interventions per state, initially 3,600 test pairs. Every pair
shares the same full snapshot and uses A0 as its unperturbed member. Record actual
applied offsets; this construction avoids clipping away the intervention.

Resolve all trials using the declared action-to-absolute-reference convention,
real low-level feedback and actual simulated time. Counterfactual evaluation
does not rerun a high-level planner during the 1.5 s sequence. Save both absolute
trajectories and their difference, so every competing diagnostic uses matched
states, actions, horizons, target bodies, and true physical outcomes.

### Required diagnostics

| Diagnostic | Definition and purpose |
|---|---|
| E-open-all | Position RMS over all four bodies on the factual sequence |
| **E-open-payload** | Payload-only factual error, matched to the mission target |
| **E-CF-payload / E-CF-other** | Absolute prediction error on the same unperturbed/intervened trajectories used for effect error |
| **E-bank-payload / E-bank-other** | Prediction error on the common 64-sequence bank |
| **CEE-self** | Effect error for the aircraft whose action was changed; controls for generic action sensitivity and differencing |
| **CEE-payload** | Effect error on the common payload |
| **CEE-other** | Effect error on the other two aircraft; primary cross-aircraft diagnostic |
| Elite regret | Secondary true-outcome decision diagnostic with a common reference pool |
| Rank agreement | Secondary ordering agreement on the common bank |

For target set S, let Delta be the true intervention-minus-A0 position trajectory
and Delta_hat the corresponding model difference. Compute
`CEE_S = sum ||Delta_hat_S - Delta_S|| / sum ||Delta_S||`, aggregating registered
times, bodies and interventions before division. Use the full 1.5 s trajectory
on a shared 100 ms grid. Report factual errors on that same grid plus the
0.5/1.0/1.5 s endpoints. Error units and body/time weights are fixed in the scorer.

Publish the unnormalized effect error and true effect magnitude as well. Fix a
physical effect-resolution threshold from development timestep/refinement tests
before model comparison. Below that threshold, label the effect unresolved and
report its absolute error and prevalence; do not create arbitrary ratios or
drop the case without reporting it. Freeze aggregation, missingness and sensitivity
rules for zero/unresolved effects and nonfinite model predictions. A composite
CEE may be reported with predeclared weights, but CEE-other and CEE-payload remain
separate and cannot be swapped after observing results.

For elite regret, evaluate each model's final MPPI plan and eight predicted-best
candidates in the true simulator. Form one common pool from the full U bank,
U's final plan, and **all evaluated models' final plans and eight candidates**.
The reference is the minimum true cost in that shared union; each model's own
final plan is included, making its regret nonnegative. Freeze pool construction
and the cost-dispersion normalization. Candidates that fail receive the same
registered failure-cost rule in every method. Cache and account for all truth queries.

## 8. Statistical analysis and the sample-size gate

Use the **model-run x task** table, retaining all four decision times as repeated
measurements of their task. At the initial minimum, H1 has 12 coupling-capable
models per data budget and 600 model-task observations per budget. These are
repeated observations of 12 trained models and 50 shared tasks, not 600 independent
models. Architecture, data budget, seed pairing, shared training data, and task
dependencies must remain visible in the analysis.

### Primary H1a analysis

Compare two prespecified predictive regressions for Y. Both include data budget,
architecture and observable task-difficulty covariates, plus crossed intercepts
for training run and task. The base diagnostics are E-open-all, E-open-payload,
E-CF-payload/other, E-bank-payload/other, CEE-self and CEE-payload. The augmented
model adds CEE-other. Transforms, regularization, fitting method and treatment of
failures/unresolved effects are chosen on development data and locked in the
analysis implementation before test access.

Evaluate prediction with blocked validation: hold out a task fold and a training-
seed block together, fit without either block, and predict their intersection.
The seed block keeps corresponding variants and data budgets together. Do not
estimate a held-out run/task intercept from its held-out outcomes. Fix all folds
before opening test results and average predictive improvements within each data
budget before averaging budgets. Report task-only and run-only holdouts separately.

H1a requires both a positive conditional association between CEE-other and worse
Y and a positive reduction in held-out prediction MAE relative to the full base
diagnostic model. Use the larger of the two calibrated one-sided p-values as the
conjunction-test p-value. Publish effect sizes, uncertainty, per-architecture
results and failure coverage. A model-level Spearman comparison remains secondary.

Use a crossed resampling procedure over task IDs and matched training-seed blocks,
recomputing relevant fits/contrasts. Keep the paired methods, data budgets and
within-task diagnostic states intact. Starting with 10,000 resamples does not
resolve a shortage of independent training runs. Calibrate the entire procedure,
including blocked fitting and missingness, before relying on its intervals.

### H2, equivalence, and multiplicity

Use the 2,000-trajectory models for primary H2 and the smaller budget as secondary.
H2a is the paired improvement of no_cross over indep on Y, with task and training-
seed uncertainty retained. H2b is equivalence of
`R = mean(Y_full) / mean(Y_no_cross) - 1` within [-0.10,+0.10], using paired
resampling and two one-sided equivalence tests. State the score-specific result
and report failure differences alongside it.

Apply Holm correction to the three primary hypothesis p-values: H1a, H2a and H2b.
For H2b, use the larger of its two one-sided p-values. Report adjusted decisions
and confidence intervals consistent with the testing procedure; an unadjusted
90% interval alone is not sufficient after multiplicity correction. H1b, ranking,
and model-selection contrasts are predeclared secondary analyses with their own
clearly labeled intervals and multiplicity policy.

Failure to reject a difference is **inconclusive**. Evidence of a small effect
requires the whole corresponding interval inside the registered equivalence
margin. Do not infer that Lee absorbed all coupling from an inconclusive H2a.

### Power and model-selection gates

Before formal freeze, register a smallest effect of scientific interest, then
simulate the complete proposed analysis under null and alternative settings.
Include conservative grids of diagnostic correlations, task/seed variance,
architecture clustering, failure frequency, and unresolved effects. Archive the
generator, seeds, null rejection rate, power and Monte Carlo uncertainty. Target
nominal type-I error control and at least 80% power at the registered meaningful
effect under the selected conservative design scenario. Do not derive power from
two marginal correlations alone.

If the initial 30-fit/50-task design is inadequate, add independent fit seeds and/or
tasks symmetrically **before** formal evaluation, then lock the resulting sample
size. More checkpoints or bootstrap replicates do not replace independent fits.
Sample size is never increased in response to a formal-test p-value or winner.
Pilot runs used to set the design remain labeled development evidence.

For each data budget, freeze the coupling-capable model selected by each development
diagnostic: E-open-all, E-open-payload, the matched absolute-error controls,
CEE-self, CEE-payload, CEE-other and elite regret. Use the same candidate model pool,
development tasks and truth-query accounting; break exact ties by a fixed artifact
ID order. Evaluate these already-selected checkpoints on the common test tasks.
Report test performance differences and uncertainty even when selectors agree.
Do not select a diagnostic, model or checkpoint using test J.

## 9. Freeze and change-control rules

This document is a development specification, not a claim that the study has been
registered or completed. Freeze the formal study only after Gates A--D below pass.
The frozen record must be a Git commit plus a checksummed registration manifest
that identifies all code and all ignored/local artifacts needed to reproduce it.
The manifest is finalized **before test tasks are executed or test scores read**.

The freeze record must contain:

- Scope, primary/secondary hypotheses, minimum meaningful effects, equivalence
  margins, tests, multiplicity handling, resampling, power evidence, folds,
  model-selection rules and claim-to-result mapping.
- Actual source revisions, source/binary hashes, environment/driver versions,
  simulator/compiler settings and exact 500 Hz clock implementation.
- Official L adapter validation, calibrated/default/tuned parameters and known
  versus privileged information; complete W-ar recipe and validation.
- Task/data generators and seeds, disjoint split IDs, collection coverage and
  rejection records, final sample sizes, final timestep and numerical tolerances.
- Model architectures, feature/history mappings, training/tuning budgets, all
  checkpoint hashes, selection logs, planner settings for both comparison panels,
  candidate/RNG rules, scoring code and all failure/undefined-metric policies.
- D extraction times and sources, intervention amplitudes and bounds, diagnostic
  banks and common regret pools, effect-resolution thresholds and numerical
  replication subsets.
- The full run matrix, artifact paths, completion/retry policy and all pending
  work. A mandatory method marked unimplemented cannot pass this gate.

Development changes require a dated rationale and a versioned ledger. Compare
physical validity, coverage, numerical stability and baseline adequacy; retain
unfavorable pilot outcomes. Test seeds/tasks remain untouched during this process.
Decide the final design from the development ledger rather than a desired test
ordering. Historical 50 Hz results and v1.1 source remain unchanged archives.

After freeze, execute every registered arm, seed and task. Interrupted infrastructure
jobs may resume/retry with the same configuration and random state, retaining all
attempts. Training divergence, physical failure and bad scores are outcomes, not
reasons to replace a seed or remove a row. Report completeness and failures.

An implementation bug requires a documented amendment, preserved original artifacts,
and reruns of **all affected arms/cases** under the corrected implementation.
Never fix only the losing method or favorable subset. If inspection of test outcomes
motivates a scientific design change, label the old set as exposed/exploratory and
register a fresh held-out test set for new confirmatory claims. Do not overwrite
the first freeze, delete failed results, or redefine success thresholds afterward.

## 10. Quality gates and deliverables

| Gate | Work | Evidence required to advance |
|---|---|---|
| A: physics and interfaces | Pin assets/firmware; implement identical 500 Hz feedback, observations, action semantics and limits | Full-state replay, correct clocks/units, validated 2/1/0.5 ms behavior, physical feasibility across the development range |
| B: credible reference methods | Integrate official L and mandatory W-ar; validate U and its fast rollout path | L's genuine three-UAV QP and official-example parity; calibrated/default/tuned runs; W-ar training/rollout checks; exact-versus-fast U parity |
| C: informative pilot | Collect balanced pilot data; train every family; run shared and individually tuned planning panels | Coverage, learning curves, nontrivial resolved cross-aircraft effects, complete development outcomes, measured compute and latency |
| D: analysis and formal freeze | Calibrate power/error rates; finalize counts; finish training/tuning and development selectors | Passed analysis calibration, frozen checkpoint IDs, complete registration manifest and commit, no test access |
| E: full evaluation | Run all mandatory arms, all registered seeds/tasks and diagnostic pairs | Complete row accounting, scored failures, shared-bank evidence, numerical replication and retained exceptions |
| F: interpretation and release | Execute frozen analysis and publish evidence | English methods/results, effect sizes and intervals, all baseline provenance, logs/checksums and accurate claim-to-result mapping |

A missed gate leads to diagnosis and a versioned development revision. It does
not trigger a calendar-based fallback, deletion of a baseline, or selection of
easier formal tasks. Scope and resources are adjusted transparently before freeze.

## 11. Result interpretation

| Observed result | Supported interpretation |
|---|---|
| H1a supported | Cross-aircraft effect fidelity adds decision-relevant information beyond matched prediction and own/payload-effect controls |
| Development CEE-based selection also improves held-out missions | The diagnostic provides actionable model-selection value for cooperative transport |
| H1b supported while H1a is inconclusive | Planning-facing regret is informative; report the cross-aircraft diagnostic's effect and uncertainty separately |
| H2a supported | Modeling the removed feedback channel improves the registered mission score |
| H2b supported | Star-mediated and fully connected prediction have equivalent scores within the registered margin in this plant/task family |
| A primary comparison is inconclusive | The available evidence does not resolve that hypothesis; retain estimates and intervals |
| Registered equivalence test supported | Evidence bounds the effect within its stated practical margin |
| A learned arm exceeds U or L | Retain and audit the result, then report the measured advantage with matched protocol and search/interface details |
| A mandatory baseline fails integration or validation | The formal comparison is unfinished; continue the baseline work and label pilot evidence accordingly |

## 12. Code, evidence, and subsequent studies

Implement new work in `uav_wm/` on `codex/novelty`. Reuse the Menagerie plant,
complete snapshot/replay contracts, upstream native controller and multirate clock
pattern. Keep historical `uav_transport/` experiment definitions and results intact;
validate shared infrastructure changes against their existing tests. Compile the
full official payload-controller bindings in an isolated ignored environment,
retaining the pinned upstream source and licenses.

Raw datasets, models, registrations and execution logs belong under `data/uav_wm/`.
Publish compact English protocol/results and checksummed evidence under
`experiments/uav_wm/v2/`. Chinese working summaries and historical Chinese drafts
remain in ignored `docs-zh-local/`. The canonical English card is this file on
`codex/novelty`; local reading copies must identify the same protocol version.

Subsequent registered studies can test adaptive physical-model baselines,
external learned multi-UAV policies, constraint changes, wind/downwash mismatch,
slack-cable recovery, real-flight datasets and hardware. Their scope follows the
claims being investigated, rather than an arbitrary calendar cutoff. The present
study establishes cross-agent prediction, diagnostic usefulness and cooperative
planning against the mandatory references above.

## Sources and existing evidence

- [Native 500 Hz Lee audit](lee-controller-frequency-audit.md): completed prior
  frequency, snapshot and timestep evidence; its mission results are not results
  for this new point-to-point protocol.
- [Public airframes and existing external baselines](uav-external-baselines.md).
- [Wahba and Honig, efficient cable-force allocation, RA-L 2024](https://arxiv.org/abs/2304.02359)
  and [official col-trans implementation](https://github.com/IMRCLab/col-trans).
- [Pinned payload-controller source](https://github.com/IMRCLab/crazyflie-firmware/blob/b08d9f3cbcc581d8e7f988cabad22c26d35f4a58/src/modules/src/controller_lee_payload.c),
  [bindings](https://github.com/IMRCLab/crazyflie-firmware/blob/b08d9f3cbcc581d8e7f988cabad22c26d35f4a58/bindings/setup.py),
  and [historical test](https://github.com/IMRCLab/crazyflie-firmware/blob/b08d9f3cbcc581d8e7f988cabad22c26d35f4a58/test_python/test_controller_lee_payload.py).
- [Objective Mismatch in Model-based Reinforcement Learning](https://arxiv.org/abs/2002.04523).
- [AD-WM](https://arxiv.org/abs/2609.30264): action discrimination and elite-regret diagnostics.
- [SkyJEPA](https://arxiv.org/abs/2606.23444): single-quadrotor JEPA and sampling control.
- [LeWorldModel](https://arxiv.org/abs/2603.19312) and
  [LeJEPA/SIGReg](https://arxiv.org/abs/2511.08544): modeling foundations.

The two unverified NeurIPS 2026 references in v1.1 are retained in the local
historical draft. They are not used as verified premises for this protocol.
