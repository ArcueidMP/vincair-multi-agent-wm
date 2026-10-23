# Contact Recovery, Termination Conditions, and Substep Convergence Audit

2026-10-23, `codex/novelty`. This round kept the environment parameters, force limits, trained model weights, and normalization unchanged. It corrected contact recovery and termination criteria, then tested numerical refinement and controlled comparisons on the composed task.

This page preserves the audit of the original physics parameters. A separate configuration with numerically stable, controllable rotation was subsequently introduced, followed by retraining under equal budgets and evaluation with frozen models. See [Stable Physics and Local World-Model Audit](stable-physics-local-wm-audit.md). That new configuration did not replace the original setting documented here.

**Conclusion: active contact recovery can now be completed reliably. Under the current parameters, however, the rotational response did not pass the convergence checks at the tested resolutions, and refinement did not preserve the control authority required by the original goals. The strong physical-abstraction claim remains unsupported.**
The previous interpretation of **0/216** as a physical failure of composition must also be withdrawn: that result was directly affected by an incorrect contact criterion.

## 1. Correct the Measurements While Preserving the Original Trajectories

VMAS 1.5.2 includes `LINE_MIN_DIST = 4/600 = 0.0066667` in its sphere–box collision distance. A positive geometric surface gap can therefore coexist with an actual normal reaction force. The old `surface_gaps <= 0.005` criterion measures geometric proximity only; it cannot determine whether contact is occurring in the engine. This issue affects the previous contact statistics and composition completion scores, but not the recorded object trajectories, prediction errors, or rotation-goal errors.

The new `helper_contact_forces` function uses the same closest-point, inner-point, and constraint-force calculations as the engine, and tests compare its outputs directly against the engine's collision calculations. In this round, dual contact means that each of the two normal contact forces reaches 0.001 simulator units. The checks also enforce a suitable contact region and penetration depth.

The original 216 trajectories were remeasured only: **the controller was not rerun, and the original data were not rewritten**.

| Metric | Previous report | Corrected using actual collision reaction forces |
|---|---:|---:|
| Contact at the end of recovery, step 90 | 3/216 | 216/216, all with dual contact |
| Composition success with all other original completion criteria retained | 0/216 | 116/216 |
| Additionally requiring actual dual contact throughout the last 5 recovery steps and the first 5 steps after handoff | Not measured | 93/216 |

Separated by resolution, the original 10-substep runs achieved 74/108 successes, or 51/108 with sustained contact across handoff. The original 20-substep runs achieved 42/108, also 42/108 with sustained handoff contact. These scoring changes cannot be credited to the new controller.

The original physical primitives were also remeasured. At orientation 0° and substeps=10, the ordinary fixed-anchor condition had previously reported any-contact/dual-contact proximity fractions of 15.20%/7.83%. Actual contact reaction forces instead give **83.04%/78.35%**. The finite-preload condition has an actual dual-contact fraction of **99.65%**. The interpretation that the ordinary anchor had almost no contact is therefore also withdrawn. The earlier preload supplement remains a valid physical control, but the low-contact-rate diagnosis that motivated it was incorrect.

Data: [Remeasurement of the Original Composition Runs](../data/novelty/rebrace_calibration_old_contact.json), [Remeasurement of Contact in the Original Primitives](../data/novelty/abstraction_physics_contact_audit.json).

## 2. The Corrected Recovery Primitive

The new [rebrace.py](../vmas_pusht/rebrace.py) reads only geometry, velocities, and contact state. It does not read task goals or world-model predictions. The primary actor and helper remain the original finite-mass bodies throughout; there is no teleportation, body locking, or infinitely rigid fixture.

1. Approach from outside the concave corner, rather than treating an agent that has not reached its contact position as already in contact.
2. Acquire dual contact using finite PD control and a 0.01 inward shift of the reference point, with control norm bounded by 1.
3. After freezing the anchor, confirm contact for 5 consecutive steps using **the helper control law that will actually execute after handoff**.
4. Verify both actual normal forces, the correct contact region, low relative velocity, primary-actor positioning, unsaturated force, and acceptable bar–stem residuals.
5. Clear the confirmation when the current conditions cease to hold. A timeout is a failure; a past contact event is not latched as success.

The gains remain `kp=30,kd=4`. After handoff, passive mode always applies zero helper actuator force; it does not secretly receive active support. The new active recovery consistently uses finite preload. For the old ordinary-anchor model, this is an explicitly recorded shift in the anchor-input distribution, not model retraining.

The controller was developed only on the subset with friction 1.5, substeps=20, orientation 0°, and zero initial perturbation, then frozen. All 6/6 active cases became ready at step 22. Handoffs were tested separately with zero primary force and primary force 0.3; all 12/12 maintained actual dual contact for the following 20 steps.

The independent recovery check used complete states generated by the same admissible motion and retreat history, comparing three friction settings, two orientations, two perturbations, and two handoff forces:

| substeps | Active recovery and sustained dual contact for 20 steps after handoff | Passive mode |
|---|---:|---:|
| 20 | 48/48 | 0/24 |
| 40 | 48/48 | 0/24 |
| 80 | 48/48 | 0/24 |
| 160 | 48/48 | 0/24 |

Recovery generally took 21–22 steps, and readiness times differed by at most 1 step between 80 and 160 substeps. This establishes reliable recovery and handoff over a finite window; it does not establish that support can be maintained indefinitely under arbitrary subsequent primary actions.

The added resources include phase state, persistence counters, anchor memory, and engine contact and weld-residual information. The number of learned parameters remains zero. These additional control and sensing costs must count toward total system complexity; reporting only the WM parameter count would be insufficient.

Data: [Development-Set Check](../data/novelty/rebrace_calibration.json), [Recovery Across Numerical Resolutions](../data/novelty/abstraction_convergence_recovery.json).

## 3. Numerical Convergence and Nontrivial Control Authority

The environment timestep dt=0.1, physics parameters, complete initial state, and controller gains were fixed; only the internal substeps changed. The main grid comprised three friction settings × two orientations × five initial-velocity conditions × thirteen primary-action programs × three helper modes, for 1,170 conditions. The primary action lasted 20 environment steps, followed by zero primary action through step 60. In a separate comparison, the **exact same joint-action sequence** from the finest resolution was replayed at the other resolutions, separating changes in controller responses from differences in integration itself.

The prespecified tolerances were a maximum position difference of 0.001 scene units and a maximum angle difference of 0.003 rad at all common timestamps, with goal-classification agreement also checked. Comparisons were not restricted to endpoints, and agreement caused by both systems remaining motionless or both failing was not treated as evidence of a usable control interface.

Maximum differences over the complete 60-step trajectory, from 80 to 160 substeps:

| friction_ratio | Fixed feedback: position / angle | Identical joint actions: position / angle |
|---|---|---|
| 0.75 | 0.001302 / 0.007818 rad | 0.001473 / 0.008625 rad |
| 1.5 | 0.001305 / 0.007395 rad | 0.001437 / 0.008086 rad |
| 2.25 | 0.001302 / 0.007411 rad | 0.001435 / 0.008103 rad |

The maximum errors exceed tolerance in all three friction groups. This must not be described as every individual condition failing. A further 160→320 check covered 48 representative conditions selected using earlier errors and response magnitudes. The maximum angle differences remained **0.00469–0.00515 rad**, or **0.00491–0.00541 rad** under identical-action replay. This was a selected follow-up, not a 320-substep conclusion for the complete set of 1,170 conditions.

More importantly, primary-action authority kept shrinking. Under the default friction, the mean peak angular response with preloaded support, relative to matched zero-primary-action controls, was:

| substeps | Primary-action angular response (rad) |
|---|---:|
| 10 | 0.031889 |
| 20 | 0.018659 |
| 40 | 0.010516 |
| 80 | 0.005673 |
| 160 | 0.002962 |

The response decreased by approximately **84%** from 20 to 160 substeps. This occurred despite actual support reaction forces being present. For example, at default friction and ss160, the ordinary-anchor condition had an actual dual-contact fraction of 96.67%. Although the dedicated recovery audit had stable contact outcomes, its complete T trajectories still showed maximum angle differences of approximately 0.0092–0.0097 rad from 80 to 160 substeps.

These checks with 20-step pushes cannot establish unreachability for every policy that pushes continuously for 60 steps, or for every primary policy. Complete composition executions are measured separately in the next section.

The [Single-Body Friction Diagnostic](../vmas_pusht/friction_resolution_probe.py) also isolated a numerical mechanism. With a constant torque equal to half the angular-friction cap, angular displacement over 6 simulated time units was 0.029263/0.014816/0.007454/0.003738/0.001872/0.000937 rad at substeps=10/20/40/80/160/320, respectively, shrinking approximately with the integration step size. A control above the friction cap retained nontrivial rotation of approximately 0.99 rad. At each substep, the engine computes friction from the current velocity before integrating the external torque. This can produce discrete creep below the friction cap; per-environment-step drag introduces a smaller correction. This isolates a mechanism, but is not a continuous-limit theorem for the complete multibody contact system. The original environment was not modified to manufacture success.

Data: [Complete Numerical Audit](../data/novelty/abstraction_convergence_verified.json), [Representative 320-Substep Follow-Up](../data/novelty/abstraction_convergence_refinement320_verified.json), [Single-Body Friction Diagnostic](../data/novelty/friction_resolution_probe.json).

## 4. Composition Reruns Under a Common Budget

Four prespecified schemes used the same goals, +0.04 → +0.08 rad, an angle tolerance of 0.015, and a position limit of 0.05. Phase budgets were 60/20/80/60, totaling 220 steps. Every failure and timeout was retained.

- `timer_baseline`: the old contact-point PD controller, with verification under the actual fixed-anchor handoff law during the last 5 recovery steps and termination on a fixed schedule.
- `corrected_timer`: the new recovery controller, still using the entire fixed duration of each phase.
- `guarded_recovery`: fixed durations for the first two phases, followed by a transition as soon as the recovery conditions are met.
- `guarded_all`: terminate the corresponding primitive after the rotation goal holds for 5 consecutive steps, retreat clearance holds for 3 consecutive steps, or recovery is confirmed for 5 steps.

Rotation completion means satisfying the goal over a finite window; it does not claim rest or indefinite maintenance. Padding recorded after early completion does not count toward completion time or control cost. Passive-mode and zero-primary-push controls are retained, and the helper actuator does not secretly continue working after a passive handoff.

The models and normalization remained frozen. To avoid the additional confound of presenting high substeps directly as a model input outside its training distribution, the substeps feature in model queries was fixed at the training reference value of 20 while the simulator was refined independently. This tests deployment of the reference-resolution model; it does not claim that a high-accuracy physics model has been trained. No retraining was performed to adapt to the test data in this round.

There were 864 new composition executions:

| Setting | Strict composition success | Contact recovery |
|---|---:|---:|
| ss20, each of the four schemes | 48/108 for all four | 72/108 for all four |
| guarded_all, ss40 | 12/108 | 72/108 |
| guarded_all, ss80 | 12/108 | 72/108 |
| guarded_all, ss160 | 0/108 | 72/108 |
| corrected_timer, ss160, full 220-step budget | 0/108 | 72/108 |

The 72 successful recovery cases cover all active-support conditions. The 36 passive cases do not satisfy the new sustained-handoff criterion. At ss20, both the WM planner and ordinary angle feedback with active support completed their respective 12/12 conditions. Neither the passive controls nor the zero-primary-push controls completed the full task. All four schemes achieved the same success count, so **there is no evidence that the new recovery controller increased success rates**. The actual benefit of event-based termination was a lower mean execution length: fixed schedules required 220 steps, guarding recovery alone required approximately 180.7, and guarding all phases required approximately 120.0. These are descriptive means over the designed grid, not confidence intervals from independent randomized trials.

At ss160, recovery still succeeded, but every final rotation goal failed. The control with the full 220-step budget also achieved zero successes. The current failures therefore cannot be attributed simply to ending recovery too early or allowing less time. This result covers only the specified macro-action family, finite controllers, and budget; it does not prove that every possible policy fails.

Data: [Four Controls at ss20](../data/novelty/composition_v2_ss20.json), [Executions at 40/80/160 Substeps](../data/novelty/composition_v2_refinement.json), [Full-Budget Control at 160 Substeps](../data/novelty/composition_v2_timer160.json).

## 5. Assessment of the Strong Claim

After correcting the measurements, the original experiments already demonstrate some composition capability; 0/216 can no longer serve as counterevidence. This round also confirms that recovery and handoff with bounded forces can be implemented reliably. However, within the current physics parameters, primary-action family, and budget, no regime was found that simultaneously provides numerical stability, sufficient rotational control authority, and reliable composition after freezing the model. Actual contact reaction forces persist while control authority weakens under refinement. Contact being present and a reusable, effective dynamics interface are therefore distinct requirements.

The current evidence does not support the claim that cooperation has formed a composable physical abstraction that reduces total world-model complexity. Nor can this negative result be generalized to show that the idea is impossible in all physical systems. The next step should first establish a task that passes convergence checks and has a torque budget sufficient for nontrivial motion, then train only on local data and test composition after freezing. Changes to friction or force must be explicitly identified as a new physics setting, rather than silently replacing the current failure conditions. There is currently no basis for scaling up helper RL to compensate for this issue.

## 6. Reproduction and Audit Files

```bash
# Correct measurements of the original trajectories and validate recovery development
.venv/bin/python -m vmas_pusht.rebrace
.venv/bin/python -m vmas_pusht.contact_audit

# Numerical audit; verified files retain the final script version without overwriting earlier raw artifacts
.venv/bin/python -m vmas_pusht.abstraction_convergence --out data/novelty/abstraction_convergence_verified.json
.venv/bin/python -m vmas_pusht.abstraction_convergence \
  --representative-from data/novelty/abstraction_convergence.json --substeps 160 320 \
  --out data/novelty/abstraction_convergence_refinement320_verified.json
.venv/bin/python -m vmas_pusht.abstraction_convergence --recovery --substeps 20 40 80 160 \
  --horizon 100 --compare-horizons 80 100 --out data/novelty/abstraction_convergence_recovery.json
.venv/bin/python -m vmas_pusht.friction_resolution_probe

# Common-budget controls and numerical refinement with the frozen model
.venv/bin/python -m vmas_pusht.abstraction_composition --frictions .75 1.5 2.25 --substeps 20 \
  --output data/novelty/composition_v2_ss20
.venv/bin/python -m vmas_pusht.abstraction_composition --frictions .75 1.5 2.25 --substeps 40 80 160 \
  --variants guarded_all --output data/novelty/composition_v2_refinement
.venv/bin/python -m vmas_pusht.abstraction_composition --frictions .75 1.5 2.25 --substeps 160 \
  --variants corrected_timer --output data/novelty/composition_v2_timer160
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

The complete suite passed **110 tests**, covering comparisons against engine contact reaction forces, recovery and handoff, identical complete initial states, common timestamps, phase timeouts and incorrect completion, frozen-model inputs, and the discrete-friction diagnostic. Source code, raw data, and frozen-model fingerprints were verified.

The experiment JSON files and corresponding `.pt` files preserve raw states, actions, phases, control budgets, model/source fingerprints, and cases that fail the thresholds. Original historical results are retained; corrections are presented through separate audit files and this document. At completion of this audit, all changes remained on the local `codex/novelty` branch and had not been pushed.
