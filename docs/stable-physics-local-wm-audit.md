# Stable Physics, Local WM Retraining, and Frozen Composition Evaluation

2026-10-23, `codex/novelty`.

**This round established a new physical configuration that retains rotational control authority and passes numerical checks on a finite grid, while preserving the original configuration as a control.**
Local WMs were retrained separately under the original and new configurations with equal budgets, then frozen for composition evaluation.
The new configuration and its corresponding retrained model increased first-stage goal successes, but complete composition still achieved **0/720**.
The main failure was traced to rebracing changing the object's pose and overshooting the final goal; it cannot all be attributed to WM representation or prediction failure.

The subsequent [recovery attribution and closed-loop simulator recovery audit](recovery-attribution-mpc-audit.md) is now complete: true-simulator MPC with the original bank13 still could not recover
108 cases that passed the first-goal gate. Adding physical side changes and reverse contact achieved 108/108, and all successes were preserved in fixed-action replay at 320 substeps.
The 0/720 in this report is the result for the original frozen action interface; it does not imply that these states are physically unrecoverable.

## 1. Physics Was Selected Before Training and Task Evaluation

Added [physics_profiles.py](../vmas_pusht/physics_profiles.py). The original `scenario.make` defaults were unchanged, and the original results, weights, and data were preserved.

| Parameter | Original configuration `original` | New configuration `dry_light` |
|---|---:|---:|
| Total T mass | 2 | 2 |
| Agent force multiplier | 1 | 1 |
| friction_ratio | 1.5 | 0.25 |
| rot_friction_ratio | 1.2 | 0.10 |
| Environment timestep dt | 0.1 | 0.1 |
| Collisions, finite-stiffness joints, drag | Original implementation | Original implementation |

The Euclidean control-norm limit remains 1. No infinite-strength fixture, object locking, or additional force was used.
Selection used only local responses, numerical discrepancies, and joint residuals, without using composition success or WM test performance.
Candidate physical coefficients were frozen first; numerical resolution was then selected against the accuracy thresholds. The final choice was **160 substeps for training/execution and 320 substeps for verification**.
The frozen selection is recorded in [stable_physics_selected.json](../data/novelty/stable_physics_selected.json).

The thresholds were a maximum position discrepancy of ≤0.001 scene units and a maximum angular discrepancy of ≤0.003 rad over all common timepoints.
The mean/maximum rotational effect relative to a matched zero-primary-action rollout also had to reach 0.04/0.12 rad, with joint position/angular residuals ≤0.01/0.08 and bounded actions.
Both fixed feedback controllers and replay of exactly the same joint actions were compared; the table reports the worse discrepancy of the two:

| Complete local test set | Substeps compared | Maximum position discrepancy | Maximum angular discrepancy | Result |
|---|---|---:|---:|---|
| Development: 57 trajectories | 40→160 | 0.001711 | 0.006060 | Fail |
| Development: 57 trajectories | 80→160 | 0.000542 | 0.002196 | Pass |
| Development: 57 trajectories | 160→320 | 0.000262 | 0.001374 | Pass |
| Independent validation: 135 trajectories | 80→320 | 0.000824 | 0.003910 | Fail |
| Independent validation: 135 trajectories | **160→320** | **0.000275** | **0.001647** | **Pass** |

At 320 substeps, the mean/maximum rotational effect of primary action relative to matched zero-action controls under preloaded support was approximately 0.198/0.658 rad on the development set
and 0.148/0.214 rad on independent validation. Under the original configuration, the same development set at 160 substeps produced only approximately 0.0029/0.0087 rad.
The new configuration's nontrivial motion did not disappear under refinement.

The maximum joint residuals on independent validation were approximately 0.00470/0.05784, within the local thresholds. Actual dual-contact fractions were approximately 81% for preloaded support and 56% for the ordinary anchor.
Some local actions lose contact, so the entire action family is not unconditionally characterized as an ideal hinge.
This is evidence from a finite local grid, not a convergence proof over all states or a real-hardware validation.

## 2. Equal-Budget Retraining on Local Data

[stable_local_wm.py](../vmas_pusht/stable_local_wm.py) collected local data separately under the original and new configurations at 160 substeps.
Each physical configuration contains 1,638 primitive-family trajectories. Causal prefixes shared by the 5/10/20-step programs were merged before simulation to avoid duplicate samples and batch-roundoff discrepancies.
Each support condition ultimately has 1,092 unique training windows, 336 validation windows, and 964 strict test windows.
The training windows come from 273 primitive families; correlated windows must not be treated as independent random replications.

- Each label is the response to a constant primary action over 5 steps; longer primitives provide local continuation states.
- All modes share a 10-step real zero-primary-action observation prelude. Helper actions and their costs are recorded; history is not fabricated.
- Training uses directions −20/0/20°, strengths 0.15/0.3/0.6/0.9, and durations 5/10/20. Validation and unseen-action directions, strengths, and durations are assigned separate splits.
- Training geometry includes small contact-position variations; the initial orientation of 30° and combined velocity perturbations are held out for testing.
- No task goals, complete compositions, retreat, rebracing, or task-counterfactual labels enter training.
- Each support condition uses a fixed 1,000 updates and three initialization seeds. Architectures and update counts are not selected using test results.

Each physical configuration has 36 trained models, all loadable with the default safe loader. Weights and normalization statistics were frozen before task execution:

| Model | Parameters | Information added relative to the coarse model |
|---|---:|---|
| Coarse geometry / body-frame coarse model | 1,763 | No velocity; current geometry, action, duration, resolution, and anchor |
| Full state | 1,731 | Velocities, angular velocities, and physical parameters |
| History model | 1,741 | Two real past geometry-and-joint-action intervals; 34 additional stored values |

The helper remains a hand-written PD controller with no learned parameters. Its anchor and phase memory, velocity/contact sensing, geometric knowledge, and physical actuation costs—and those of the recovery coordinator—cannot be treated as free resources.
The raw world-coordinate coarse model also exhibits orientation-OOD extrapolation problems, so the main interpretation uses the prespecified body-frame, full-state, and history baselines.

Strict test results use `test_action/test_context/test_nuisance` and exclude validation.
The inherited `all_heldout` aggregate in the original report includes validation and must not be directly described as a strict OOD test.

Angular prediction RMS under the new configuration, over 964 strict test windows per support condition, averaged across three initializations:

| Support | Body-frame coarse | Full state | History |
|---|---:|---:|---:|
| passive | 0.02361 | 0.02980 | 0.02037 |
| world_anchor | 0.03793 | 0.01502 | 0.01780 |
| preloaded_anchor | 0.10244 | 0.01300 | 0.02430 |

Fixed support did not make this velocity-free coarse representation generally sufficient. Full state and history retain substantial advantages under active support.
Finite data and optimization also affect these results, so the network's entire fitting error cannot be interpreted as an information-theoretic lower bound.

Models and data: [original configuration](../data/novelty/stable_local_original_models.json),
[new configuration](../data/novelty/stable_local_selected_models.json),
and [strict summary excluding validation](../data/novelty/stable_angular_selected_primary_strict_test.json).

## 3. Frozen Composition and the Two-by-Two Control

All executions use the same bank13: zero action plus three directions × four strength levels, with each action held constant for 5 steps.
The goal sequence retains `.04→.08 rad` and additionally tests `.06→.12 rad`; angular tolerance is 0.015 and the bar-translation limit is 0.05.
Prelude time, force, work, and movement are recorded separately and included in the costs; goals are defined relative to the physical state after the prelude.
Phase budgets remain 60/20/80/60, and recovery control and criteria were frozen before task outcomes were observed.

The primary scoring distinguishes two outcomes:

- **External ordered-goal success**: the first and final goals meet their consecutive-five-step gates, with terminal pose, finiteness, and boundary constraints satisfied.
- **Complete contact-interface execution**: additionally requires actual retreat and sustained recovery.

This avoids automatically declaring passive support unable to achieve the manipulation goals merely because it cannot satisfy the active dual-contact requirement.

The original/new physics × original/new equal-budget retrained body-frame coarse models were all executed at the same 160 substeps. First-stage successes are shown below, with 36 design conditions per cell:

| Execution physics / training physics | Original-configuration model | New-configuration model |
|---|---:|---:|
| Original configuration | 10/36 | 0/36 |
| New configuration | 0/36 | 25/36 |

This indicates that the new physical configuration restores useful control authority and requires a matching local dynamics model.
Under the new configuration, ordinary angular feedback achieved 36/36 first-stage successes, full state 23/36, and history 24/36.
After 160→320 refinement, the coarse model changed from 25→24; the counts for the other three remained unchanged. These are first-stage successes and must not be presented as complete composition successes.

The complete fixed matrix contains **720 composition executions, with external ordered-goal success of 0/720**.
The primary model uses the prespecified seed0 for execution; local prediction results separately report three initializations. No model was selected using task outcomes.
A further 4,032 unseen-action counterfactual branches were recorded. All frozen models were compared afterward on the same states and ground truth, avoiding a comparison restricted to each model's own visited states.

## 4. Failure Occurs at Recovery Handoff, Not Only in WM Prediction

At 160 substeps under the new configuration, of the 108 cases that passed the first stage, **108/108 had already exceeded the final angular tolerance after rebracing**.
Of these, 75 used active support. A representative case follows; all angles are relative to the post-prelude initial state:

| Stage | Bar angle |
|---|---:|
| End of first stage; goal 0.04 | 0.038512 |
| End of retreat | 0.038423 |
| End of rebracing and primary-agent repositioning | 0.121292 |
| Final state; goal 0.08 | 0.128607 |

The rebracing stage added **0.082869 rad**. Contact and low-relative-speed gates can be satisfied without constraining the change in object pose.
The recovery coordinator moves both agents, so the entire disturbance cannot be attributed to the helper acting alone.

From the same complete physical states after handoff, the true simulator tried all 13 candidates over 5 future steps.
Among the 75 active-support cases, no action was found that reduced the combined goal cost or achieved the final pose.
This diagnostic saved 2,808 oracle branches. It shows only that the current short-horizon action library cannot provide an immediate repair; it is not proof of unreachability under all longer-horizon policies.
The current library only pushes from the prescribed contact direction. Increasing model capacity cannot be assumed to supply reverse rotation or repositioning capabilities automatically.

Local training covers only constant-push windows, not stopping after a strong push, arbitrary policy changes, retreat, or rebracing.
Final failure therefore also involves training coverage and primitive-transition contracts. It cannot all be attributed to missing information in the coarse representation, nor can this single implementation rule out all composable abstractions.

Diagnostic: [handoff states, same-state model comparison, and the true short-horizon oracle](../data/novelty/stable_composition_diagnostics.json).

## 5. Long-Task Physical Replay Also Passes the Numerical Thresholds

A few coarse-model cases had larger endpoint discrepancies in complete closed-loop execution. Their actual joint actions were therefore fixed for an additional replay,
preserving every batch member, row order, and full snapshot, and comparing all timepoints over 203 steps.
There were two groups of 108 cases each: replay at 160 substeps was bitwise identical to the original trajectories in all 24 physical-state coordinates.

| Goal sequence | Maximum position discrepancy, identical actions at 160→320 | Maximum angular discrepancy |
|---|---:|---:|
| .04→.08 | 0.000696 | 0.002307 |
| .06→.12 | 0.000611 | 0.002030 |

Both are below `.001/.003`. The checks also pass when excluding zero-primary-push cases, when stratifying by phase/controller, and for post-termination padding segments separately.
Thus the approximately `.033 rad` endpoint discrepancies in a few closed-loop cases cannot directly be labeled a failure of physical integration to converge under identical actions.
They more plausibly involve amplification of small differences by feedback, readiness states, and event decisions; which factor is a sufficient cause has not been independently identified.

One physical-fidelity limitation remains: the maximum bar/stem relative-angle residual during long tasks was **0.08249 rad**, slightly above the local threshold of 0.08.
The position residual was approximately 0.00631. The scene remains a finite-stiffness welded approximation, and the microtask score still uses bar pose; it cannot unconditionally be interpreted as an ideal rigid T.

Data: [fixed-action numerical audit of complete tasks](../data/novelty/stable_task_numerics.json).

## 6. Separate Angular-Velocity Diagnostic, Without Replacing the Primary Experiment

An exploratory check was conducted after the primary models were frozen: rotation has inertia, and geometric constraints need not eliminate angular velocity.
Only current bar angular velocity was added to the coarse body-frame input. The resulting model has 1,708 parameters and uses the same training rows and update count; it was not used for action selection in the primary experiment.

Under the new configuration, strict-test angular RMS for passive/world/preload changed from `0.02361/0.03793/0.10244`
to `0.02068/0.02356/0.04185 rad`. Second-order motion information explains part of the error, but full state remains a strong baseline.
There is also no evidence that the helper makes this compact representation generally outperform passive support.

The comparable repeated subset in the exact-alias analysis contains only 105/189 rows, all with initial omega=0; this coverage must be reported.
On the same subset and under the same primary actions, world/preload reduce RMS sensitivity to the remaining translational perturbations by approximately 13.7%/16.9%,
but the remaining angular RMS increases from passive support's 0.001328 to 0.003089/0.002335 rad.
Reducing geometric degrees of freedom and reducing task-relevant predictive uncertainty require separate verification.

This is a post hoc mechanism diagnostic. It neither retrained nor replaced the primary models, and results from the revised representation are not presented as a preregistered success.
Data: [angular-velocity diagnostic](../data/novelty/stable_angular_selected_models.json)
and [alias-coverage audit](../data/novelty/stable_angular_selected_alias.json).

## 7. Conclusions Currently Supported

The completed progress is a new configuration that retains the original-parameter control, permits rotation, and passes local and actual-action replay numerical checks,
followed by evidence of improved first-stage goals after equal-budget local training.
What remains unproven is that fixed support makes the coarse representation sufficient, reduces total system resource requirements, or produces a complete composable interface from this round's local training alone.

The most direct next step is to give rebracing an **exit condition that bounds pose disturbance**, or to include its actual state transition and reverse operations in the local primitive library.
First establish that the goal is reachable within that library, then test whether the frozen small model can predict and compose its operations.
Any complexity carried by the helper, coordinator, or additional sensing must be included in the accounting.
This round neither proves the strong claim nor establishes that the idea is impossible under other reasonable interface designs.

## 8. Reproduction and Implementation

```bash
# The original configuration remains available; the new selection is recorded in stable_physics_selected.json
.venv/bin/python -m vmas_pusht.stable_physics --profiles dry_light original \
  --split development --substeps 40 80 160 --out data/novelty/stable_physics_development
.venv/bin/python -m vmas_pusht.stable_physics --profiles dry_light \
  --split validation --substeps 80 160 320 --out data/novelty/stable_physics_validation320
.venv/bin/python -m vmas_pusht.stable_physics --profiles dry_light \
  --split development --substeps 160 320 --out data/novelty/stable_physics_development320

# Exactly the same local-training budget under both physical configurations
.venv/bin/python -m vmas_pusht.stable_local_wm --profile original --substeps 160 \
  --output data/novelty/stable_local_original
.venv/bin/python -m vmas_pusht.stable_local_wm --profile dry_light --substeps 160 \
  --output data/novelty/stable_local_selected

# Frozen full controls under the new configuration; original-physics and cross-matrix arguments are in the corresponding JSON files
.venv/bin/python -m vmas_pusht.stable_composition --profile dry_light \
  --new-models data/novelty/stable_local_selected_models.pt --training-profile dry_light \
  --substeps 160 320 --planners new_coarse_body_small new_full_matched new_history_matched \
  analytic_feedback zero_push legacy_wm --counterfactual \
  --output data/novelty/stable_composition_selected_matched
.venv/bin/python -m vmas_pusht.stable_diagnostics
.venv/bin/python -m vmas_pusht.stable_task_numerics
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

New code: [physics profiles](../vmas_pusht/physics_profiles.py), [physics validation](../vmas_pusht/stable_physics.py),
[local training](../vmas_pusht/stable_local_wm.py), [frozen composition](../vmas_pusht/stable_composition.py),
[handoff diagnostics](../vmas_pusht/stable_diagnostics.py), and [task numerical replay](../vmas_pusht/stable_task_numerics.py).

Primary execution data: [matched original configuration](../data/novelty/stable_composition_original_matched.json),
[matched new configuration](../data/novelty/stable_composition_selected_matched.json),
[original physics / new model](../data/novelty/stable_composition_original_cross.json),
and [new physics / original model](../data/novelty/stable_composition_selected_cross.json).

The reproduction commands write to the listed output paths. To preserve this round's frozen artifacts, copy them to a separate experiment directory before running the commands.
The [final hash verification](../data/novelty/stable_provenance_audit.json) checked 132 references and 40 distinct files.
The `stable_physics.py` source used for two early physics experiment records lacks the subsequently added `finalize_selection` function present in the current version.
Source snapshots matching the original records' SHA256 hashes exactly have been saved. Simulation, criteria, local training, and task-execution code were not changed by that addition, and historical records were not overwritten.

**143 tests passed**. Source, physical-configuration, data, and model hashes were preserved. At the time of this audit, all changes were on the local `codex/novelty` branch and had not been pushed.
