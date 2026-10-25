# Contact-Mode Coverage Bottlenecks Joint Multi-Agent World-Model Planning

[Branch overview](../README.md) · [Companion aerial-transport study](uav-transport-world-model-audit.md)

**Executable contact transitions are a structural determinant of joint
planning capability. The bottleneck persists when both agents are planned
and when outcomes are predicted by the true simulator. Expanding contact-mode
coverage restores complete-task performance at the same candidate budget.**

Across 1,152 complete executions, adding physical contact transitions raises
the learned joint planner from **0/108 to 68–81/108** successful tasks, and the
simulator-based joint planner from **0/36 to 32–34/36**. Every prespecified
allocation preserves all 273 primary/helper programs per decision, with the
same force limits and task-time budgets. The advantage survives numerical
refinement and uses local models frozen before task evaluation.

This extends the contact-interface finding into centralized multi-agent
planning: both the primary action and the helper's behavior are decision
variables. The experiment establishes physical interaction coverage as an
independent design dimension of world-model planning, alongside prediction
quality and coordination.

## Complete-task results at 160 substeps

All 1,152 prespecified executions finished with weights, normalization,
candidate allocations, costs and termination rules frozen before evaluation.

| Predictor and helper | Direct | Expanded, allocation 0 | Allocation 1 | Allocation 2 | Shared first-stage success |
|---|---:|---:|---:|---:|---:|
| Learned, fixed helper | 2/108 | 75/108 | 71/108 | 66/108 | 77/108 |
| Learned, jointly planned helper | **0/108** | **71/108** | **68/108** | **81/108** | 87/108 |
| True-option simulator, fixed helper | 1/36 | 35/36 | 35/36 | 35/36 | 35/36 |
| True-option simulator, jointly planned helper | **0/36** | **32/36** | **34/36** | **34/36** | 36/36 |

Three results support the central claim:

1. **Joint control retains the bottleneck.** Giving the planner seven helper
   choices for every primary program still yields zero direct-interface
   completions. Supplying contact transitions recovers 68–81 learned-model
   tasks and 32–34 simulator-planned tasks across the three allocations.
2. **The intervention works with true dynamics.** The simulator arm establishes
   that the available physical interactions constrain capability even with
   accurate option prediction.
3. **Coverage matters at a fixed search budget.** Direct and expanded joint
   interfaces preserve the exact same primary-force/duration/helper multiset
   and 273 candidate slots. Every allocation retains a large advantage.

Joint planning reaches 87/108 first-stage successes with the learned model and
36/36 with simulator prediction. Contact expansion then enables the final
corrections needed to complete the composed task. The fixed-helper rows give
the corresponding 39-candidate reference; all allocation outcomes are shown.

The helper choice varies during execution: 100/108 learned direct
episodes and 100/101/106 of 108 expanded episodes use multiple helper options;
the corresponding simulator counts are 36/36 and 35/36, 36/36, 31/36.
The recorded choices verify that the planner changes the helper's behavior
over the task as part of its joint decisions.

All 864 direct/expanded pairs retain matching first-stage outcomes and
pre-final clocks. Across complete 24D physical states, the maximum prefix
discrepancy is 2.794e-7; joint-command discrepancy is at most 7.451e-8, below
the 1e-6 audit tolerance. There are 5,580 real option decisions across the
matrix. Raw histories, selected options and controller memory are retained.
**All 864 paired comparisons have zero regressions at both 160 and 320
substeps:** expansion preserves every direct-interface success while
recovering additional complete tasks.

Compact evidence: [per-episode outcomes](../experiments/planned_helper/v1/episodes.csv),
[summary](../experiments/planned_helper/v1/summary.json),
[paired comparisons](../experiments/planned_helper/v1/paired_results.json), and
[query/resource accounting](../experiments/planned_helper/v1/compute.json).

## Executable interface

`joint_interface.py` extends the original atomic interface with a fifth field,
`helper_id`. Seven helper options are available: the historical support law,
release, body-relative contact tracking, corner pressure, corner withdrawal,
and two tangential anchor shifts. Anchor changes update reference memory only.
Both agents remain finite-mass collidable bodies, with individual Euclidean
force limits of one. Both commands run concurrently throughout a primary
option, including its navigation and contact-acquisition steps. The planner
chooses the pair using the task endpoint cost. Low-level feedback and local
prediction are goal-independent; goals enter the planner's cost.

Helper option zero reproduces the historical support law. A support label
specifies the initial/prelude/recovery condition and the behavior of option
zero. In a planned-helper arm, even a scene initialized with passive support
may subsequently receive an active helper choice. Retreat and rebracing use
common scripted task phases. The seven low-level feedback programs are
handwritten, and their high-level selection belongs to the joint planner.

## Frozen comparison

There are 39 primary force/duration programs. Fixed-helper arms offer 39
options; planned-helper arms offer all 39 × 7 = 273 primary/helper pairs.
Within each helper regime, direct and expanded interfaces have exactly the
same number of slots and the same force/duration/helper parameter multiset.
Expansion assigns a route to each nonzero pair using
`(duration_index + force_index + helper_id + allocation) % 3`. All three
allocations are declared and reported. Zero pushes remain direct; when a
route is unavailable, the complete direct bank on the current face is used.
The first stage always uses direct routes in both members of a paired test.

The primary comparison is **joint direct273 versus joint expanded273**.
The fixed/planned-helper factor spans 39 and 273 candidates respectively.
Candidate evaluations, simulator integration steps, selector time, actual
travel and both agents' work are measured separately.

The finite matrix covers 36 physical conditions: three initial support
conditions, two orientations, three velocity perturbations, and two goal
pairs. Each learned arm uses three frozen fit seeds; the simulator control is
run once per condition. Eight arms across these predictors give 1,152 complete
executions. Fit seeds and allocations provide paired repeats over these 36
physical conditions. Each decision evaluates one complete option of lookahead.

## Local learning and evidence

Independent goal-free local collection uses 32 training, eight validation and
eight test contexts per support, all seven helper IDs, physically acquired
contact faces, and physically executed continuation states. Half of the prepared sources
receive a prescribed helper/primary action before collection to cover
helper-induced continuations. There are 295,680 atomic-option labels. Test
partitions hold out primary actions or contexts while retaining the seven
helper IDs. Training consists exclusively of local atomic-option outcomes.
The frozen models then compose those predictions into the four-stage task.

One two-hidden-layer width-128 MLP per fit seed is shared across support
conditions. Its 46 inputs encode current physical state, the remembered
anchor, primary route/force/duration, helper choice and the original support
law. Five outputs predict bar displacement, option duration and completion.
All arms use the same frozen weights. Training uses 2,000 updates, batch size
512, training-only normalization, and the final checkpoint. Validation only
sets the duration residual margin. Body-frame canonicalization remains a
geometric prior; VMAS's componentwise friction is not exactly invariant under
rotation.

The completion output is clipped regression. Every collected atomic option
completed, so the completion labels have zero negative examples; the main
prediction evidence comes from continuous physical outcomes.
Across seeds, strict action-test angular RMSE is 0.0479–0.0502 rad and
context-test angular RMSE is 0.0487–0.0519 rad. These longer/unseen local
programs provide a stricter distribution than the task's selected options.
Each deployed model has 23,173 parameters; three fits measure seed sensitivity.

The training partition has 262,080 examples, validation 20,160, and each strict
test 6,720. Independent NumPy inference on 4,096 predetermined rows per seed
differs from Torch by at most 1.431e-6; training-only normalization reproduces
exactly. Replaying the first and last complete collection batches per support
reproduces 9,984 sampled local outcomes, input features and all 24D endpoints
bit for bit.

The study's original regression run passed **247 tests**; current combined-branch
validation is listed in the [overview](../README.md#environments-and-checks). The new tests cover physical helper
actuation, force bounds, exact fixed-helper compatibility, controller-memory
forks, candidate-budget preservation, goal-free inputs, simulator-free learned
selection, context splits, and portable-weight reconstruction.

## Numerical validation

Every one of the 1,152 original trajectories and its scoring labels reproduces
bit for bit at 160 substeps. Fixed-action replay at 320 substeps retains the
interface advantage in every arm:

| Predictor and helper | Direct at 320 | Expanded allocation 0 | Allocation 1 | Allocation 2 |
|---|---:|---:|---:|---:|
| Learned, fixed helper | 2/108 | 75/108 | 71/108 | 66/108 |
| Learned, jointly planned helper | 0/108 | 69/108 | 68/108 | 80/108 |
| True-option simulator, fixed helper | 1/36 | 34/36 | 34/36 | 34/36 |
| True-option simulator, jointly planned helper | 0/36 | 31/36 | 33/36 | 33/36 |

**1,149/1,152** trajectories pass the original maximum bar-position/angular
error thresholds of 0.001 scene units and 0.003 rad. Maximum discrepancies
are 0.00055642 and 0.00342917 rad. Three learned joint-direct trajectories
exceed the angular threshold; all three remain unsuccessful at both
resolutions. Their original numerical-validation failure flags are retained.

Nine ordered-success labels change to failure at the original dwell/stop
boundaries. These include allocation repeats on shared physical conditions.
All are counted in the table above, using identical recorded actions and the
original stopping times at the finer resolution.

See [verification](../experiments/planned_helper/v1/verification.json) and the
per-episode table for the complete nominal/fine scores and refinement flags.

An additional **post-hoc** 320/640 check replays both complete batches that
contain the three threshold failures (36 trajectories). It preserves all
original failures and stopping times. The maximum new discrepancies are
0.00049282 position units and 0.00317216 rad: 35/36 pass, while a different
joint-direct trajectory exceeds the angular threshold. No ordered-success
label changes between 320 and 640 in these batches. This additional check
preserves the task-level conclusions while identifying residual numerical
sensitivity in long direct-control trajectories. The complete
[finer diagnostic](../experiments/planned_helper/v1/finer_diagnostics.json)
accompanies the primary audit.

All nominal executions remain finite and clear of arena boundaries, with no
failed-option events. Both agents' force norms remain within 1+1e-6. Work and
distance accounting use simulator scene units.

## Reproduction

From the repository root with the CPU audit dependencies installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.joint_experiment \
  --output data/novelty/planned_helper_new --register-only
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.joint_models \
  --output data/novelty/planned_helper_new/local --workers 3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.joint_experiment \
  --output data/novelty/planned_helper_new --workers 6
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.joint_audit \
  --output data/novelty/planned_helper_new --workers 6
# Optional post-hoc diagnostic of every group failing the original 160/320 check:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.joint_refinement \
  --output data/novelty/planned_helper_new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.joint_report \
  --run data/novelty/planned_helper_new --output experiments/planned_helper/new
```

Use fresh output directories. Evaluation can resume completed, hash-verified
groups; it rejects changed models or registered source files. Compact results,
per-episode tables and portable frozen weights belong in
`experiments/planned_helper/v1/`. Large data and full recorded trajectories
remain under the ignored `data/novelty/planned_helper_v1/` with SHA256 indexes.

For evaluation without recollecting or retraining, register a fresh output
directory and restore the published model pack there:

```bash
.venv/bin/python -m vmas_pusht.joint_report \
  --restore-model experiments/planned_helper/v1/frozen_models.json \
  --output data/novelty/planned_helper_new/local/models.pt
```

The JSON preserves every float32 parameter and normalization value. Repacked
Torch files can have different container hashes; the restored tensors must
match exactly. Full local-data replay audits additionally require the raw
collection files. Numerical checks replay recorded actions at 160 and 320
substeps from identical complete states, keeping the original stopping times
and five-state dwell windows.

The optional 640-substep diagnostic preserves every original failure and
threshold. It replays each entire flagged batch with the same recorded
commands and original stop times.
