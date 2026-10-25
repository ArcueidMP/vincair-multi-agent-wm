# Completed Contact Interface and Frozen Local-WM Composition Rerun

2026-10-25, `codex/novelty`.

**The completed action interface enables learned models to solve full composed tasks. With the same frozen velocity-augmented model, allowing real side changes raises ordered success from 6/108 to 64/108 paired executions.** These are 36 physical conditions evaluated under three model initializations, not 108 independent random samples. The first-stage outcomes agree in every pair, and trajectories before the final stage differ by at most 1.49e−8 across all physical coordinates.

This round starts each complete task from its initial scene. It is distinct from the earlier [108/108 true-simulator recovery result](recovery-attribution-mpc-audit.md), which replaced continuations from stored handoff states. The new learned planners do not query future simulator states.

Follow-up: the [predictor-by-interface factorial audit](predictor-interface-factorial-audit.md) adds the missing
larger-model and true-option direct controls, repeats the existing cells, and tests three prespecified
39-candidate expanded allocations without retraining. It reports the resulting interface bottleneck and
its numerical and physical-cost limits separately from the broader physical-abstraction hypothesis.

## 1. Executable contact options

[action_interface.py](../vmas_pusht/action_interface.py) exposes a finite, task-independent interface:

- Continue pushing at the current contact face.
- Physically withdraw and move to the upper-right contact, then push downward.
- Physically travel around the left exterior to the lower-left contact, then push upward.

A command specifies route, direction, force magnitude, and pushing duration. The force bank is zero drive plus directions −20/0/20 degrees at magnitudes 0.15/0.3/0.6/0.9. Pushing durations are 5/10/20 steps, giving 39 direct options and 117 expanded options. A route can be initiated only from the original contact mode; this experiment permits one side change, not arbitrary mode cycling.

Every navigation, acquisition, pushing, and failure-handling step is executed through ordinary physics. The interface retains current mode, route phase, elapsed time, contact confirmation, and held-force memory. A failed route does not become an acquired contact: subsequent nonzero continuation is rejected, while a timed zero-drive fallback remains available. Cancellation and phase-boundary resets modify controller memory only.

The original helper law, anchor semantics, finite masses, physics parameters, and force bound are retained. The interface itself never reads the task goal. Actual goal dwell and deadlines may interrupt an option; interrupted executions are recorded separately from complete-option outcomes.

## 2. Local data and frozen models

[option_models.py](../vmas_pusht/option_models.py) collected **66,240 goal-free atomic-option trajectories**, with 22,080 per support condition:

| Split, per support | Examples |
|---|---:|
| Training | 18,720 |
| Validation | 1,440 |
| Strict action test | 960 |
| Strict context test | 960 |

Each support uses 48 training, 12 validation, and 24 test contexts. Contexts vary initial velocity, angular velocity, contact-position error, anchor inset, and local motion history. Current upper/lower faces are reached by actual bounded routes, not by teleporting agents to a convenient contact.

All 252 face preparations per support succeeded. Each acquired face/context supplies its initial query plus one genuine continuation query after pushing, stopping, or finite PD holding. The history input contains two real past five-step geometry/action windows. Training therefore covers the reverse-contact continuation states omitted by the earlier fixed-contact model.

Training contains all declared bank13 force programs at durations 5/10/20. Validation uses directions ±10 degrees and strengths 0.45/0.75. Strict tests use directions ±35 degrees, strengths 0.45/0.75, durations 15/25, and held-out 30-degree contexts. No complete task trajectory, task goal, old handoff snapshot, or recovery-MPC task label enters collection or fitting.

The five outputs are bar displacement in x/y, wrapped angular displacement, elapsed physical seconds, and a clipped completion regression score. Inputs add current contact mode and requested route to the earlier local-state representation.

| Model | Input dimensions | Parameters per support | Additional information |
|---|---:|---:|---|
| Coarse body-frame model | 24 | 2,021 | Geometry, action, duration, anchor, and mode/route indicators |
| Velocity-augmented matched model (`full_matched`) | 38 | 1,949 | Raw velocities, angular velocities, and physical parameters before canonicalization |
| History matched model | 58 | 1,919 | Two real past intervals; 34 extra stored values |
| Larger velocity-augmented model | 38 | 22,149 | Same input information, larger network |

There is one network per support condition. A deployment serving all three support configurations would need three networks of the chosen architecture. Three seeds and four architectures are experimental comparisons, not free components of one small deployed model. The fixed helper, navigation controller, sensing, and memory also count toward system resources.

All 36 models use the same applicable training rows, 1,000 Adam updates, batch size 256, and learning rate 0.003. Normalization uses training data only; elapsed-time residual margins use validation only. Weights and normalization were frozen before task evaluation.

Two limits matter:

- Every sampled production option completed. Completion accuracy is therefore trivial; the completion score has no demonstrated ability to distinguish failure risk. Runtime failure behavior is enforced by the interface and tested synthetically.
- Body-frame canonicalization imposes the earlier geometric-invariance prior. VMAS's componentwise friction is not exactly rotation-equivariant. The velocity-augmented model is not a fully faithful simulator representation merely because its name contains `full`.

Strict-test angular RMSE, averaged over the three seeds and 1,920 test examples per support:

| Support | Coarse | Matched velocity | History | Larger velocity |
|---|---:|---:|---:|---:|
| passive | 0.04652 | 0.04891 | 0.05018 | 0.04489 |
| world_anchor | 0.06126 | 0.04877 | 0.04800 | 0.05149 |
| preloaded_anchor | 0.05987 | 0.04969 | 0.05285 | 0.05201 |

These test options include unseen directions and longer pushes; their errors cannot be directly equated with the shorter options selected in task execution. They nevertheless do not establish a uniformly sufficient coarse representation.

Data and training report: [option_interface_local_models.json](../data/novelty/option_interface_local_models.json). Frozen checkpoint SHA256: `447460169e669319b7874de6d3f1c91dfbeae95479af198a0e23dc37e8a135a4`.

## 3. Complete task evaluation

The runner executes rotation, retreat, rebracing, and final rotation from the initial scene. It retains the real 10-step prelude, stage budgets 60/20/80/60, unit force bound, angular tolerance 0.015 rad, translation limit 0.05, and five-step goal dwell. Goals are defined relative to the post-prelude pose.

There are 36 physical conditions: three support configurations, two initial orientations, three velocity perturbations, and goal pairs `.04→.08` or `.06→.12 rad`. Five learned policies are evaluated under three model seeds; two fixed-feedback policies and one true-option control are evaluated once. This gives **648 complete task executions**.

The first rotation uses direct options for every policy. The final rotation enables side changes except in `wm_full_direct`. That ablation uses exactly the same retrained model as `wm_full`, with the same cost and data; it removes side-change choices only. Its direct durations are still 5/10/20, so it is a matched ablation of this interface rather than an exact rerun of every detail of the historical bank13 planner.

The planner evaluates **one complete atomic option** and replans after it finishes. It does not perform the earlier full-remaining-horizon simulator search. Direct-option duration is exactly known from the interface. Routed options use predicted duration plus a validation residual margin, and the real stage deadline is always enforced. Raw model timing outputs are retained for error analysis rather than replaced by these known-duration constraints.

Goals enter the endpoint cost, never the model input. The `true_option` control evaluates the identical candidate options and endpoint criterion with the simulator. Fixed upper/lower feedback controls choose a predetermined reverse route after overshoot and quantize the original angular-feedback law to bank13.

### Ordered-goal success

Each seed column has 36 physical conditions. The dashes denote controls that were not redundantly repeated across model seeds.

| Planner | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Matched velocity model, direct only | 1/36 | 1/36 | 4/36 |
| Matched velocity model, expanded | **24/36** | **15/36** | **25/36** |
| Coarse model, expanded | 21/36 | 19/36 | 14/36 |
| History model, expanded | 22/36 | 19/36 | 5/36 |
| Larger velocity model, expanded | 23/36 | 25/36 | 16/36 |
| Fixed upper-route feedback | 10/36 | — | — |
| Fixed lower-route feedback | 15/36 | — | — |
| True-option simulator control | 35/36 | — | — |

The direct/expanded matched-model comparison is the clearest controlled result:

- Both pass the first goal in the same 74/108 executions.
- All phase endpoints before final rotation agree. The maximum discrepancy in the complete physical prefix is 1.49e−8.
- Expanded options turn **58 paired failures into successes, with zero paired regressions**; six executions succeed under both interfaces.
- Complete success rises from **6/108 to 64/108**, or from 6/74 to 64/74 conditional on the shared successful first stage.

The true-option control reaches the final goal in all 36 cases but fails the first goal in one, yielding 35/36 complete ordered successes. A later success does not erase a failed prefix. Contact-interface completion is reported separately from external ordered-goal success; passive support is not automatically failed for lacking an active dual-contact gate.

The models still vary substantially across seeds. The larger model also totals 64/108, so this run does not show that increasing capacity alone resolves the remaining failures. A fixed lower-route controller achieves 15/36 without learned parameters and remains a material baseline.

For the expanded matched model, successes by support are passive 23/36, world_anchor 21/36, and preloaded_anchor 20/36, pooled over the three model seeds. Success with passive support prevents attributing the improvement uniquely to cooperation making prediction simpler.

Main results: [interface_complete_v1.json](../data/novelty/interface_complete_v1.json). Paired prefix checks: [interface_complete_v1_analysis.json](../data/novelty/interface_complete_v1_analysis.json).

## 4. Numerical verification and boundary-sensitive scores

Every actual batch of 18 is replayed independently, retaining its original complete post-prelude snapshot, row order, and joint actions. At 160 substeps, all 24 physical coordinates and all scoring labels reproduce exactly.

All **648/648** active trajectories pass the original refinement thresholds of 0.001 position units and 0.003 rad when replayed at 320 substeps. The maximum discrepancies are **0.00061517 / 0.00265336 rad**. All trajectories remain finite and avoid walls.

However, passing a trajectory-error threshold does not guarantee identical discrete dwell outcomes. At the original stopping times, complete ordered successes change **294→290**, with four losses and no gains:

- True-option control: 35→34.
- Coarse model, seed 2: 14→13.
- History model, seed 1: 19→17.
- Every other planner/seed ordered count is unchanged. In particular, expanded matched-model counts **24/15/25** and direct-only counts **1/1/4** all persist.

One example crosses the strict first-goal tolerance in the first sample of its five-state dwell window: angular error changes from 0.01498461 to 0.01505780 rad. The endpoint remains valid, but only four consecutive states qualify at the original stopping time. That case is recorded as a loss; no padding steps or relaxed thresholds repair its score.

The audit retains all 13 changed first/final/terminal/ordered labels individually, including cases where an unchanged aggregate hides one loss and one gain. This is fixed-action numerical refinement, not a new closed-loop planner evaluation at 320 substeps.

One nominal trajectory slightly exceeds the previously declared weld-angle limit: maximum 0.08003736 rad versus 0.08; the finer maximum is 0.07997966. Maximum weld-position residual is about 0.006293. The object remains a finite-stiffness bar/stem assembly, and these local microtasks score bar pose rather than validating a rigid-body or hardware abstraction.

Numerical report: [interface_complete_v1_numerics.json](../data/novelty/interface_complete_v1_numerics.json). Boundary details: [interface_complete_v1_numerics_boundaries.json](../data/novelty/interface_complete_v1_numerics_boundaries.json).

## 5. Common-state prediction audit

A separate read-only audit evaluates every architecture and seed on identical states and selected options from the true-option control. Its labels are complete-option simulator counterfactual outcomes, including when the real task terminated the option early after satisfying its goal. Such counterfactual labels are kept distinct from interrupted real endpoints. No diagnostic data return to training or policy selection.

The audit contains 124 selected-option queries: 66 in the first rotation and 58 in the final rotation. There are 57 interrupted real executions with complete simulator counterfactual labels, and no fallback queries lacking such labels.

Both original batches replay bitwise before feature extraction. Mean angular RMSE over the three seeds on these identical queries is:

| Model | All 124 queries | First-stage 66 | Final-stage 58 |
|---|---:|---:|---:|
| Coarse | 0.01553 | 0.01855 | 0.01112 |
| Matched velocity | 0.01414 | 0.01638 | 0.01096 |
| History | 0.02329 | 0.02123 | 0.02420 |
| Larger velocity | 0.01244 | 0.01300 | 0.01172 |

The larger network improves average prediction error on this common set without improving aggregate task success. Endpoint MSE and closed-loop action selection therefore need separate evaluation.
History seed 2 has a first-stage angular bias of −0.01073 rad, versus −0.00018/−0.00232 for seeds 0/1. It classifies only 27/55 truly inside oracle endpoints as inside, versus 35/55 for the other seeds. This is compatible with its poor first-stage control, but cannot establish the cause of its own rankings or visited states from oracle-selected options alone.

Validation timing margins also do not provide guaranteed task-state coverage: observed coverage on these queries is approximately 85%/81%/67%/85% for coarse/matched/history/large models. Completion labels are again all positive and do not validate failure detection.

Common-state audit: [interface_complete_v1_model_audit.json](../data/novelty/interface_complete_v1_model_audit.json).

## 6. What this establishes

The interface is executable and partially learnable: local models trained without complete-task trajectories can exploit it to complete the fixed rotation–retreat–rebrace–rotation task at both goal pairs. The matched-weight ablation directly demonstrates the value of adding real contact-side changes, independently of retraining or increased parameter count between those paired policies.

The stronger claim remains open. The helper and navigation controllers are hand-written and use extra sensing and memory; the coarse model is not uniformly sufficient; seed sensitivity is substantial; and only a restricted task grammar with one side change has been tested. This experiment does not establish learned physical abstraction, general primitive recombination, or lower total system complexity.

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.option_models \
  --output data/novelty/option_interface_local --workers 3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.interface_experiment \
  --models data/novelty/option_interface_local_models.pt \
  --output data/novelty/interface_complete_v1 --model-seeds 0 1 2 --workers 8
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.interface_numerics \
  --manifest data/novelty/interface_complete_v1.json --fine-substeps 320 \
  --workers 8 --out data/novelty/interface_complete_v1_numerics
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.interface_model_audit \
  --source data/novelty/interface_complete_v1.json \
  --out data/novelty/interface_complete_v1_model_audit
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

Use fresh output prefixes to preserve existing artifacts. All model/data/result files remain under the gitignored `data/novelty/` directory. Earlier checkpoints and experimental artifacts are preserved.
The full-task launcher also checks the two earlier `stable_local_original_models.pt` and `stable_local_selected_models.pt` packs as immutable references; these must exist from the previous audit, but are not used to choose new task actions.

**216 tests pass.** The [provenance audit](../data/novelty/interface_complete_v1_provenance_check.json) verifies
274 references across 174 distinct source, data, model, and result files with no mismatches. The earlier model packs remain unchanged.
