# Cooperation as Physical Abstraction: Validation Record

2026-10-23, `codex/novelty`. This is an experimental audit of the idea that a helper can make a coarse state representation sufficient for prediction and control through physical contact.
All code retains VMAS 1.5.2's finite mass, finite contact stiffness, and bounded force; the helper is not converted into an infinitely rigid fixture.

**Contact-measurement correction (2026-10-23): the earlier claims of “0/216 composed-task successes” and “only 3/216 cases in contact at the end of recovery” are withdrawn.**
The old geometric threshold omitted the thickness of VMAS's collision boundary. In the original trajectories, 216/216 cases actually had dual contact at the end of recovery.
Keeping the other original conditions gives 116/216 successes; adding a requirement for sustained contact before and after handoff gives 93/216.
The relevant passages below have been corrected. Subsequent controller repairs and numerical refinement are documented in the [latest reassessment](rebrace-convergence-audit.md).

**The evidence supports local physical effects and execution of some new goals, but does not yet validate a stable, composable abstraction interface that reduces total system complexity.**
Fixed, handcrafted compliant support does reduce the influence of some hidden disturbances while retaining primary-action control authority. However, simple fixed rules and full-state models remain strong baselines,
the angular-response benefit is sensitive to the integration step size, and model errors on unseen actions remain much larger than the conditional ambiguity at identical inputs.
A frozen small model improves from 16/72 to 48/72 on local new-goal tasks; composition results must use the contact correction above.
Subsequent numerical refinement did not preserve these control effects, so success at the original resolution cannot establish a stable physical interface.

## I. Prior work and the research claim that remains defensible

The closest predecessor is not limited to fixture learning. [COMBO-Grasp (2025)](https://arxiv.org/abs/2502.08054)
already uses one policy to generate support poses and another to reorient and grasp an object, with results on a real bimanual robot and unseen objects.
The general idea that two agents, one supporting and one moving, make manipulation easier is not itself a new contribution.

| Primary source | What it already covers | Distinction this project still needs to establish |
|---|---|---|
| [Learning to Scaffold, ICRA 2020](https://arxiv.org/abs/1911.00969) | Learning fixture placement to reduce manipulation uncertainty through contact constraints | A mobile helper with limited capabilities, counterfactual action prediction, and reuse |
| [COMBO-Grasp, 2025](https://arxiv.org/abs/2502.08054) | Learning bimanual support, reorientation, and grasping | Predictive sufficiency of a coarse representation and a reduction in total resources |
| [Robust Predictable Control, NeurIPS 2021](https://arxiv.org/abs/2109.03214) | Jointly learning a model and policy that favor predictable states | Continued validity for a different family of primary actions after freezing the helper |
| [Feedback Linearization for Uncertain Systems via RL, ICRA 2020](https://tyler-westenbroek.github.io/pdfs/ICRA_2020_FBL.pdf) | Using feedback to make a complex system exhibit simpler dynamics for higher-level reference control | An independent helper implementing the interface solely through constrained physical contact |
| [Approximate Simulation, Automatica 2009](https://www.sciencedirect.com/science/article/pii/S0005109808004731) | Planning with a simple abstract system and refining its behavior to the real system through an interface | The multi-body contact mechanism that learns to form the interface, and its empirical advantage |
| [Environment/agent/joint world models, 2026 preprint](https://arxiv.org/html/2608.20401v1) | Closed-loop coupling can reduce predictive complexity on a restricted support set | The result does not automatically cover arbitrary primary-action counterfactuals; separate intervention tests are required |
| [Learning Extrinsic Dexterity, ICRA 2024](https://arxiv.org/abs/2310.17785) | Learning and composing manipulation primitives that exploit environmental contact | Predictability and controllability of unseen compositions after freezing the helper and model |

These correspondences compare the papers' contents; they do not prove that no prior work has studied the intersection of these ideas.

The mathematical motivation is valid. If the same `(phi(x), u)` corresponds to two deterministic responses `y,y'` separated by distance delta,
any deterministic predictor must incur an error of at least `delta/2` on one of the states. For two equally likely states, the optimal mean squared error is `delta²/4`.
Increasing the parameter count cannot recover information absent from the input, whereas adding history or sensing may resolve the problem.

The helper's closed loop must reduce response variation under **the same representation and the same primary action**, while preserving primary-action control authority.
That alone does not establish a complete MDP abstraction: rewards/goals must remain distinguishable, the property must cover subsequently reachable states, and the helper's mode, memory, and anchor must be included in the complete state.
A reduction in one-step or fixed-horizon conditional variance does not establish multi-step Markov sufficiency.

## II. Physical protocol and representation

The primary agent applies force above the left side of the T's horizontal bar; the helper disk supports the T near its right concave corner.
VMAS sphere–box contact provides normal reaction forces without tangential fingertip friction. Floor friction must not be interpreted as fingertip friction.
Dual contact approximates a pivot only within an action range where reaction-force directions are feasible, force does not saturate, and contact does not separate.

- `phi` contains 12 dimensions of geometry: both agents' positions, plus bar/stem positions and orientation sin/cos.
- The small model additionally receives the primary action, duration, substeps, and helper anchor, for a total of 18 dimensions.
- The full input additionally includes velocities/angular velocities and friction parameters, for a total of 32 dimensions.
- **This is not an information-loss experiment on the released 15D WM.** Its 15D input already includes bar and agent velocities. This round deliberately defines a coarser observation interface.
- Initial hidden velocities are small, rigid-body-consistent perturbations without initial penetration. The main experiment treats them as admissible initial conditions, not as states claimed to have been reached through prior action trajectories.
  The support-removal experiment separately generates intermediate states through actual bounded-action histories.

The main experiment comprises 31,104 trajectories comparing four conditions: a passive disk, PD control around a fixed world anchor, PD tracking of a T-relative contact point, and velocity damping.
The support rules read neither the goal nor the primary action plan and have no learned parameters. Fixed gains are `kp=30,kd=4`, with an additional gain=8 for the damping rule.
Every condition uses the same physical initial state and primary action, with a control-norm limit of 1.

Training actions use directions −20°/0°/20°, strengths 0.3/0.6/0.9, and durations 5/10/20.
Unseen actions use directions ±35°, strengths 0.45/0.75, and durations 15/30. An initial orientation of 30° and larger combined velocity perturbations are also held out.
friction_ratio takes values 0.75/1.5/2.25, and substeps takes values 5/10/20. substeps is explicitly included in the input, and conditional variance is stratified by substeps,
rather than pooling different numerical integration settings as an unobserved physical state.

At the time, a geometric-gap threshold incorrectly gave contact rates of approximately 15%–18% / 5%–8% for ordinary PD, prompting a supplemental check:
**move the finite PD reference point inward by 0.01 scene units along both normals of the concave corner**. This changes the reference point, not the body's position or its stiffness to infinity.
The fixed-preload supplement comprises 23,328 trajectories, with another 15,552 stress-test trajectories for an outward position error of `(+0.003,−0.003)`.
This supplement was motivated by an incorrect low-contact interpretation and is retained as a finite-preload control; it no longer supports the claim that ordinary PD almost never makes real contact.
Remeasurement using actual collision reaction forces gives any-contact/dual-contact rates of 83.04%/78.35% for the ordinary anchor at orientation 0° and ss10, and a dual-contact rate of 99.65% with preload.
No further gain search was conducted to obtain positive results.
All three datasets remained finite and avoided the arena boundary; the actual maximum control norm did not exceed 0.9.

## III. Response variation decreases, but the rotational abstraction benefit is modest

The following table reports results at initial orientation 0°, weighting each duration equally. Primary-action control magnitude subtracts the response under zero primary action in the same condition.

| substeps | Ordinary anchor: reduction in angular-response std | Preloaded anchor: reduction in angular-response std | Primary-action control magnitude with preload / passive | Corrected rate of actual dual contact with preload |
|---|---:|---:|---:|---:|
| 5 | 25.7% | 23.6% | 1.012 | 99.65% |
| 10 | 17.9% | 12.9% | 0.970 | 99.65% |
| 20 | 11.4% | 6.1% | 0.969 | 99.65% |

The corrected contact rate uses actual normal forces and weights all valid trajectories equally. Angular response and control magnitude retain the original duration-grouped aggregation.

Sustained contact is achieved and control authority is largely retained, but angular-response variation decreases only modestly, with the benefit weakening under finer integration.
Tracking the contact point does not outperform a fixed anchor; it instead increases angular-response dispersion.

Across all held-out data at substeps=10, grouping model inputs with **exactly identical numerical values** gives:

| Condition | Conditional translational RMS (scene units) | Conditional angular RMS (rad) |
|---|---:|---:|
| Passive disk | 0.002209 | 0.002862 |
| Ordinary fixed anchor | 0.001045 | 0.002541 |
| Preloaded fixed anchor | 0.000951 | 0.002567 |

The helper therefore reduces some effects of hidden disturbances, particularly translation. This measures ambiguity in the finite-sample conditional distribution, not prediction error after fitting a network.
It does not establish that a previously insufficient representation has become sufficient: this round has not shown that a task's required error threshold is crossed.

Rotation produced by the helper itself must not count as primary-action control authority. With preload and zero primary action, the maximum mean angular changes are approximately
0.028/0.022/0.016 rad at substeps=5/10/20. This is why the zero-action reference must be subtracted.

At the same `+0.02 rad` progress, a preloaded fixed anchor reduces translational std by approximately 9%/26%/50%, but the fraction of cases in which all modes reach that progress is only
93%/78%/43%, and control effort increases by approximately 29%/23%/11%. This progress-matched comparison has limited coverage and cannot be generalized to an unconditional benefit.
Effort is the time integral of squared control and is not labeled physical energy consumption. The data separately record an approximation to absolute actuator work, integrated total force, and travel distance.

## IV. Frozen small models: normalized MSE alone is insufficient

The ordinary-support experiment fits 48 models, and the preload supplement fits 36. Each mode uses the same 1,890 training conditions, 600 optimizer updates, and 3 initialization seeds.
The small coarse model and body-frame small model each have 1,763 parameters, the full-state model has 1,731, and the large coarse model has 19,331.
Normalization is computed from the training set alone, output scales are shared across modes, and architectures and training steps are fixed; models are not selected on test results.
The body-frame transform was introduced before inspecting test results to remove the obvious confound that a world-coordinate MLP lacks a rotational inductive bias.

At substeps=10, the means across three initializations for unseen primary actions are:

| Training group / condition | Input and parameter count | Translational prediction RMS (scene units) | Angular prediction RMS (rad) |
|---|---|---:|---:|
| Ordinary group, passive | Coarse input, 1763 | 0.002323 | 0.021261 |
| Ordinary group, fixed anchor | Coarse input, 1763 | 0.002305 | 0.020481 |
| Ordinary group, passive | Full input, 1731 | 0.001794 | 0.014553 |
| Preload group, passive | Coarse input, 1763 | 0.002296 | 0.025042 |
| Preload group, fixed anchor | Coarse input, 1763 | 0.002116 | 0.018397 |
| Preload group, passive | Full input, 1731 | 0.001971 | 0.013569 |

Normalized MSE in the ordinary group decreases from 0.4354 to 0.3561, approximately 18.2%; however, physical translational RMS improves by only about **0.8%**, and angular RMS by about **3.7%**.
The normalized loss uses training scales for different coordinates; it must not be presented as an across-the-board 18% reduction in physical prediction error.
The preload group improves more substantially, but a small model with full inputs remains a strong baseline. The ordinary and preload groups use different sets of modes and different shared normalization scales,
so normalized MSE is not compared directly across the groups. The physical-unit metrics above are interpretable, but they are not a single-factor ablation of changes to training-loss weights.

Some improvements for the ordinary fixed anchor are larger on unseen initial velocities and orientations: on the held-out orientation test, body-frame translational RMS decreases from 0.002745 to 0.001629.
However, its angular error is approximately 0.0098 rad, and unseen-action angular error is approximately 0.0205 rad, still substantially larger than the same-input ambiguity of approximately 0.0025 rad.
Model fitting, action extrapolation, and multi-step response modeling remain major issues; all failures cannot be attributed to missing information in the representation.

A total-complexity advantage has not been demonstrated. Although the helper has 0 learned parameters, it uses additional velocity feedback, anchor memory, handcrafted gains, and a predefined concave-corner geometry,
and consumes additional actuator work. The full-input baseline even uses fewer model parameters. Poorer OOD performance of the large coarse network does not prove a capacity lower bound.
The full input also reads velocities and friction parameters, making it an information-completion baseline. It is not a deployment option with the same sensing budget as the coarse model,
and its performance alone cannot rule out the value of support under restricted observation.

## V. Contact interventions from the same intermediate state

[abstraction_release.py](../vmas_pusht/abstraction_release.py) captures complete intermediate states after actual bounded-action histories of 1/5/10 steps.
With the primary action fixed, it compares four branches: maintain support, apply zero helper action, disable helper collisions, and physically retreat and approach again.
The tests include unseen constant-force directions and two-segment switching programs. Coarse representations are not guaranteed to match across different histories; only the four intervention branches share exactly the same complete initial state.

Disabling collisions is an analytical `do(contact=off)` intervention, not an executable robot policy. Zero action leaves the physical body in place, so the two interventions are not interchangeable.
For 60 intermediate-state/program combinations with default friction, substeps=10, and an ordinary anchor, disabling contact versus maintaining support gives terminal-difference RMS values of
**0.003814** in translation and **0.013811 rad** in angle. The corresponding values for a preloaded anchor are **0.008028** and **0.026308 rad**.
These are real dynamical effects of paired interventions, not proof of a sufficient abstraction.
The intervention detects changed responses, but does not reconstruct a family of hidden perturbations with strictly identical phi at the intermediate states. A change in mean response therefore cannot be interpreted directly as an increase in conditional ambiguity.

For the ordinary anchor's frozen model on continuations of constant primary actions, angular RMS is approximately 0.0228 rad with support maintained and 0.0325 rad with collisions disabled.
The corresponding values for the preload model are 0.0211 and 0.0403 rad.
Continuation states were absent from reset-only model training, and disabling contact changes the model's operating mode. The increase therefore cannot be attributed entirely to restored hidden-state ambiguity.

Physical retreat followed by reapproach can reestablish contact in this local program. However, making contact again at some point is not equivalent to restoring sustained support, much less completing a full task.
The frozen planning and composition tests below impose separate requirements on goals, retreat, positioning, and related conditions.

## VI. Frozen models with new goals and a different primary controller

After training, model weights and helper rules are fixed while executing 648 local angular-goal microtasks and 216 composed tasks.
New goals increase orientation relative to the initial pose by 0.04/0.08/0.12 rad, with an error tolerance of 0.015 rad and a requirement that bar translation remain below 0.05 scene units.
Primary controllers include a frozen WM selecting 5-step actions, ordinary angular feedback, and zero primary force during rotation phases.
At each decision, the WM selects only from the same 10 constant-force candidates, with no simulator lookahead, retraining, or candidate optimization.
Ordinary feedback reads the true angular velocity, so this is not an algorithm competition with equal sensing or computational budgets.

Each cell contains 72 designed conditions combining friction, substeps, orientation, initial velocity, and new goals:

| Helper condition | Frozen WM | Ordinary angular feedback | No primary push |
|---|---:|---:|---:|
| Passive disk | 16/72 | 15/72 | 0/72 |
| Ordinary fixed anchor | 48/72 | 36/72 | 0/72 |
| Preloaded fixed anchor | 28/72 | 48/72 | 12/72 |

This provides more useful positive evidence than prediction loss alone: fixed physical support improves actual execution of some new goals, and the ordinary-anchor group's benefit is not achieved by the helper acting alone.
It does not establish generalization to new robots, arbitrary new policies, or the full PushT task. The zero-primary-action preload control already completes some 0.04 rad goals;
all no-push controls fail at the larger goals, so the entire preload benefit cannot be credited to primary-action control authority.

Numerical sensitivity remains substantial: ordinary anchor + WM succeeds in 12/36 and 36/36 cases at substeps=10/20, respectively;
preload + ordinary feedback succeeds in 36/36 and 12/36, respectively. An advantage must not be reported by selecting only one integration setting.

The composition program is fixed in advance: 60 steps rotating toward +0.04 rad → 10 steps of physical retreat → 20 steps of rebracing and primary-agent positioning → 60 steps rotating toward +0.08 rad.
All actions are bounded, time is continuous, and neither objects nor agents are repositioned or reset midway through execution.
Strict completion requires the first goal to hold at its phase boundary, actual separation from contact, contact maintained at the end of rebracing, primary-agent positioning, and satisfaction of the final goal.

**The original 0/216 score was incorrect.** Recomputing actual collision reaction forces shows that all 216 trajectories have dual contact at step 90.
Keeping the original requirements on the terminal goal, first goal, retreat, and primary-agent positioning gives **116/216** completions.
Additionally requiring continuous actual dual contact during the final 5 recovery steps and the first 5 steps after handoff gives **93/216**.
These are measurement corrections to old data, not improvements attributable to the later recovery controller.
The old contact/task_success fields in the original JSON retain their historical values and must be interpreted together with the [remeasurement results](../data/novelty/rebrace_calibration_old_contact.json).
All executions remained finite. Boundary events remain in the original records; boundary-failure counts must not be recomputed using the original incorrect success labels.

Reset-only local training did not cover most continuation or repositioned states, and this round did not use these execution data for further training.
Some composition capability exists at the original resolution, but this does not establish stability under numerical refinement; the [subsequent reassessment](rebrace-convergence-audit.md) examines that issue separately.

## VII. Defensible conclusions and investment assessment

The strongest statement currently supported without overreach is:

> Within the specified family of local rotation tasks, restricted observations, and bounded forces, fixed compliant support can reduce the influence of some hidden physical differences
> and improve a frozen small model's control of some new goals. These benefits depend on contact conditions and numerical settings, and do not yet establish a stable, composable dynamics abstraction.

Subsequent work repaired contact measurement, recovery, and completion conditions. The latest results show that numerical refinement substantially attenuates the rotational effect of primary actions,
which is a more immediate bottleneck. A converged physical task with nontrivial control authority should be established before comparing history-augmented/joint models at equal total resources.
Contact mode and helper memory must still be part of the interface.

Large-scale helper RL is not yet warranted. Fixed rules already explain the observed physical benefits,
while COMBO-Grasp, feedback linearization, and physical fixture learning limit the novelty of the general narrative.
If the eventual result is simply learning to place a pivot, or using the helper's additional state sensing to simplify the primary model's problem, the contribution should be positioned at that actual scope.

## VIII. Data and reproduction

Source: [physical_abstraction.py](../vmas_pusht/physical_abstraction.py), [abstraction_models.py](../vmas_pusht/abstraction_models.py),
[abstraction_release.py](../vmas_pusht/abstraction_release.py).
Frozen transfer: [abstraction_transfer.py](../vmas_pusht/abstraction_transfer.py).
All data are in the gitignored `data/novelty/` directory. JSON records configurations, metrics, and provenance fingerprints; the corresponding `.pt` files store samples, trajectories, models, or predictions.

```bash
.venv/bin/python -m vmas_pusht.physical_abstraction --output data/novelty/abstraction_physics.pt
.venv/bin/python -m vmas_pusht.physical_abstraction --preload .01 --modes passive world_anchor track_contact \
  --output data/novelty/abstraction_physics_preload.pt
.venv/bin/python -m vmas_pusht.physical_abstraction --helper-error .003 -.003 --angles 0 --no-trajectories \
  --output data/novelty/abstraction_physics_contact_error.pt
.venv/bin/python -m vmas_pusht.abstraction_models --data data/novelty/abstraction_physics.pt \
  --output data/novelty/abstraction_models
.venv/bin/python -m vmas_pusht.abstraction_models --data data/novelty/abstraction_physics_preload.pt \
  --output data/novelty/abstraction_models_preload
.venv/bin/python -m vmas_pusht.abstraction_release
.venv/bin/python -m vmas_pusht.abstraction_transfer
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

Main experiments: [physical metrics](../data/novelty/abstraction_physics.json), [preload metrics](../data/novelty/abstraction_physics_preload.json),
[contact error](../data/novelty/abstraction_physics_contact_error.json), [model metrics](../data/novelty/abstraction_models.json),
[preload model metrics](../data/novelty/abstraction_models_preload.json), [support removal](../data/novelty/abstraction_release.json),
[frozen-model transfer](../data/novelty/abstraction_transfer.json).
