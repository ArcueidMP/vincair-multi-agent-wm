# Audit of VMAS Cooperation Boundaries and Pairing with Fixed Marginals

2026-10-23, local branch `codex/novelty`, based on `b9102e83cc52276362355add86efb3bdc0313c55`.
Python 3.10.21, VMAS 1.5.2, PyTorch 2.11.0+cpu; the model is the released `wm_full_h_agg2.pt`, without retraining.

This round completed physical boundary measurements, exact snapshot repairs, and pairing experiments. The results support further investigation of **the value and predictability of joint actions under specified conditions and budgets**.
The default VMAS setup cannot justify the premise that no solo policy can complete the task; these experiments also do not establish that the representation has discarded information needed for cooperation.

## 1. State snapshots: repaired, with a real but limited effect

The original 15-dimensional `state(env)` stored only the translational states of the two agents and the state of the T's bar, omitting the independently integrated stem.
`set_state` returned the stem to an ideal rigid-body position and assigned it the bar's velocity, so it could not restore joint residuals produced by contact.

- Added `snapshot.capture/restore`: saves every entity's position, velocity, orientation, and angular velocity, along with agent control buffers, goals, rewards, and step counts.
- Supports duplication and reordering of snapshot rows; checks physical parameters and topology before restoration; snapshots, source environments, and destination environments do not share mutable storage.
- The true-simulator CEM and `--diag` paths in `plan.py` now use complete snapshots.
- Preserves the released model's input format. `set_state` remains available for approximate placement, with the stem's rigid-body velocity corrected to `v_stem = v_bar + omega × r`.
- Corrects tied ranks in the diagnostic Spearman calculation: correlation is undefined when all costs are equal, so JSON records `null` instead of producing a spurious ranking signal.

For 8 orientations reached through actual contact dynamics, replaying the same subsequent action stream from exact snapshots reproduced all physical states, rewards, and done flags exactly at 1/5/25 steps.
At step 25, the old compact-state reconstruction produced a maximum bar-position error of **0.003024** and a maximum orientation error of **0.016966 rad**.
Among the same 16 candidate action sequences, the best candidate changed for 1 of the 8 initial states; mean selection loss was **0.000463**.
The old reconstruction therefore introduced a measurable bias, but this evidence is insufficient to attribute the original planner's low success rate primarily to that bias.

Exact state restoration does not guarantee identical long-term numerical trajectories under arbitrary batch layouts. A different batch size or set of neighboring batch rows can change the floating-point computation path.
The pairing experiments audited this difference separately rather than hiding it by relaxing replay assertions.

Data: [snapshot.json](../data/novelty/snapshot.json). Implementation: [snapshot.py](../vmas_pusht/snapshot.py),
[snapshot_audit.py](../vmas_pusht/snapshot_audit.py).

## 2. Solo versus joint pushing: strict two-agent necessity does not hold

Design: duplicate the same complete state for each condition and compare `00 / 10 / 01 / 11`. An inactive agent receives zero input but remains a collidable body that can be pushed.
All active actions use directions normalized to unit Euclidean norm, avoiding the confound that diagonal actions in `[-1,1]²` exert greater force.
Measurement starts after 10 zero-action steps. The target is a translation of 0.2 along the pushing direction, with position tolerance 0.05 and orientation tolerance 20°.

The sweep covers friction_ratio = 0/0.75/1.5/2.25, substeps = 2/5/10/20, orientation = 0°/45°,
contact layout = wide/side/centered, strength = 0.5/1, B delay = 0/10/25, and H = 5/25/100/200.
There are **576 conditions and 2304 trajectories across the four intervention branches**, with 2304 condition-by-horizon rows recorded; all states remained finite.
VMAS's joint implementation rejected the four friction combinations at substeps=1, which are explicitly recorded as `unsupported`.
These form a designed condition grid, not independent and identically distributed samples; no confidence interval for a success probability is reported.

For the default r=1.5, axial pushing with wide contact, unit actions, no delay, and H=25, **signed center-of-mass displacement along the pushing direction** is as follows. All lengths are in uncalibrated scene units, not SI units.

| substeps | A only | B only | A+B | `11 − 10 − 01 + 00` |
|---|---:|---:|---:|---:|
| 2 | 0.050081 | 0.050081 | 0.126508 | 0.026346 |
| 5 (default) | 0.026792 | 0.026792 | 0.117240 | 0.063655 |
| 10 | 0.017865 | 0.017865 | 0.114684 | 0.078954 |
| 20 | 0.013090 | 0.013090 | 0.113308 | 0.087129 |

This interaction contrast uses physical displacement, not differences in arbitrary latent coordinates. It measures the interaction effect of the specified action interventions; it is not a reachability theorem over all policies.

A clear counterexample to solo impossibility is **r=1.5, substeps=5, centered contact, 45°, a continuous unit-norm push by A, and zero action by B**.
A reaches the target at **step 115**, before the first geometric arena-boundary contact at **step 195**. At H=200, signed center-of-mass displacement is **0.199095**, and the final state still satisfies the goal.
Reducing A's strength to 0.5 under the same conditions also succeeds at **step 236**, before boundary contact at step 392.
From the same initial state, joint unit actions succeed at step 36, showing that the time budget can change the need for cooperation within this action family.

Discretization also changes the assessment: for r=2.25 and joint axial pushing with wide contact, substeps=5/10 succeed at steps 76/140, respectively, whereas substeps=20 fails within 200 steps.
Consequently, friction_ratio alone cannot establish a static-friction threshold or coalition necessity. VMAS 1.5.2 applies zero friction at zero velocity, and its low-speed friction also includes componentwise clipping that depends on the integration step size.
This round retained the original physics implementation; friction was not modified to force two-agent necessity.

`success_ever` and `success_final` are recorded separately to avoid misclassifying trajectories that pass the goal and overshoot. First success, first arena-boundary contact, contact fraction, and finiteness are also recorded.
Boundary flags are particularly important when summarizing long horizons. Different B delays repeat the same A-only control, and these repetitions must not be treated as independent evidence.

Data: [boundary.json](../data/novelty/boundary.json), [centered-contact long-horizon control](../data/novelty/boundary_centered.json).
Implementation: [boundary.py](../vmas_pusht/boundary.py).

## 3. Fixed marginals: pairing gains exist, but the current model recovers only part of them

For each initial state, each agent has a fixed library of 8 complete action sequences. A coupling is a permutation matrix divided by 8, with every row and column summing to exactly 1/8.
Each balanced execution uses every sequence exactly once; it neither selects a single best action pair from the fixed pool nor reoptimizes or resamples individual candidates.
The independent coupling's expectation is the mean outcome over all 64 pairs. Maximum-weight matching identifies the optimal permutation on this finite outcome table.

Methods include independent coupling, 256 random permutations, a synchronization rule that maximizes pulse overlap, a frozen world model, and the best/worst permutations under the true simulator.
The model and synchronization rule choose pairings before observing the true future outcomes.
The model receives a genuinely executed 10-step zero-action history containing 3 frames, rather than a history fabricated by repeating frames.
Each group uses the same candidate table: the model evaluates 64 pairs and the true simulator also evaluates 64 pairs, without extra CEM iterations or candidate expansion.
The true optimum uses privileged information and serves as a diagnostic upper bound. The synchronization rule and model have different actual computational costs, so this is not an algorithm comparison at equal online cost.

A timing-only family was fixed first, followed by a prespecified, harder timing-and-direction family. In the latter, each agent's candidates receive deflections of `−30/−20/−10/0/0/10/20/30°`;
start times and deflection order are independently permuted with fixed seeds. Pulse duration and force norm remain equal. Impulse directions may differ, but the candidate libraries are identical across methods.
Each family tests 6 conditions: wide/narrow/side × 0°/45°. H=25 uses 10-step pulses, and H=60 uses 20-step pulses. Angles were not tuned and conditions were not selected based on outcomes.

The metric is `initial target-pose error − terminal target-pose error`, with the scenario's orientation weighting. The table reports descriptive means over the six designed conditions.
All gains are relative to independent coupling; they are not percentage-point increases in success rate.

| Candidate family / H | Independent progress | Synchronization gain | Frozen WM gain | True-optimum gain | Fraction of optimal gain recovered by WM |
|---|---:|---:|---:|---:|---:|
| Timing / 25 | 0.031943 | 0.018793 | 0.013915 | 0.018793 | 74.0% |
| Timing / 60 | 0.057081 | 0.034268 | 0.017556 | 0.034268 | 51.2% |
| Timing + direction / 25 | 0.019297 | 0.012995 | 0.006668 | 0.014762 | 45.2% |
| Timing + direction / 60 | 0.011114 | 0.019292 | 0.002215 | 0.032853 | 6.7% |

The synchronization rule is optimal in every timing-only condition, so that family alone cannot establish a need for complex cooperation learning.
With directional variation at H=60, synchronization recovers only **58.7%** of the optimal gain, leaving measurable pairing opportunities beyond synchronization.
The current model does not exploit that opportunity: in the direction-varying family, its pairing is worse than independent coupling in 1/6 conditions at H=25 and 2/6 at H=60.
This provides a clear diagnostic target for subsequent training, but it does not identify representation failure rather than dynamics prediction, the decoded cost, or long-horizon error as the cause.

**Across all four pairing experiments, no candidate succeeds on the translation target of 0.2.** This round establishes pairing gains in progress and physical response, not gains in task success rate.
H=60 also exceeds the released model's reported validation horizon of 5 model steps; long-horizon prediction error must be considered separately.

Validation and numerical audit:

- Complete replay within the same batch has zero error for all 24 conditions; hashes of full action sequences, initial states, histories, score matrices, and permutations are saved.
- An additive-matrix negative control consisting of a row term plus a column term has zero pairing gain in every condition.
- Each selected pairing is also executed again as 8 balanced trajectories; every sequence of each agent has a usage count of 1.
- A further 8 cyclic permutations cover all 64 pairs, providing an independent-coupling reference at the same batch size instead of relying only on lookup in the larger outcome table.
- The maximum individual-cell difference in progress after changing batch size is 0.00000523/0.00013012 for timing-only H25/H60 and 0.00001158/0.00217468 for the direction-varying family.
  For the direction-varying family at H60, mean gains in the actual batches of 8 balanced executions are **0.002177** for WM, **0.019251** for synchronization, and **0.032858** for the original large-table optimal permutation; the ranking is unchanged.
  Cross-batch differences were not attributed to a confirmed, specific VMAS bug or treated as evidence of omitted snapshot fields.

Data: [timing H25](../data/novelty/pairing_pilot_h25.json), [timing H60](../data/novelty/pairing_pilot_h60.json),
[direction H25](../data/novelty/pairing_steering_h25.json), [direction H60](../data/novelty/pairing_steering_h60.json).
The corresponding `.pt` files store complete action libraries and snapshots. Implementation: [pairing.py](../vmas_pusht/pairing.py).

## 4. Testable claims for the next stage

A defensible claim is: **with individual action-sequence capabilities and marginal distributions fixed, the pairing relationship has measurable decision value; the released model's ability to exploit that value breaks down as conditions and horizons change.**
The evidence remains a finite diagnostic study. A new method, generalization to unseen teams, minimum-coalition identification, and an advantage from causal representations have not yet been established.

The next step should compare an ordinary joint-action WM with training constrained by interaction effects on matched data, keeping dataset size, candidate libraries, and prediction budgets equal.
Test splits should hold out contact layouts, directions, and relative timing. Synchronization, analytical geometry/torque pairing, and direct outcome-prediction baselines should all remain in the comparison.
The key metrics should be success and progress, pairing regret, and the fraction of attainable gain recovered under unseen conditions, rather than latent MSE alone.
If synchronization or simple geometric rules still explain the gains, or if gains occur only at numerical boundaries that move with substeps, the claim should be narrowed.

## 5. Reproduction and validation

Only this repository and the downloaded state model are required; neither the pixel models, the original training data, nor a GPU is needed.

```bash
uv venv --python 3.10 .venv  # Skip if the environment already exists
uv pip install --python .venv/bin/python -r requirements-novelty.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python -m vmas_pusht.snapshot_audit --out data/novelty/snapshot.json
.venv/bin/python -m vmas_pusht.boundary --output data/novelty/boundary.json
.venv/bin/python -m vmas_pusht.boundary --output data/novelty/boundary_centered.json \
  --frictions 1.5 --substeps 5 --layouts centered --angles 0 45 \
  --strengths 0.5 1 --delays 0 --horizons 25 100 200 500
.venv/bin/python -m vmas_pusht.pairing --checkpoint data/vmas_pusht/wm_full_h_agg2.pt \
  --horizon 25 --pulse-steps 10 --out data/novelty/pairing_pilot_h25.json
.venv/bin/python -m vmas_pusht.pairing --checkpoint data/vmas_pusht/wm_full_h_agg2.pt \
  --horizon 60 --pulse-steps 20 --out data/novelty/pairing_pilot_h60.json
.venv/bin/python -m vmas_pusht.pairing --checkpoint data/vmas_pusht/wm_full_h_agg2.pt \
  --bank-family timing-steering --horizon 25 --pulse-steps 10 --out data/novelty/pairing_steering_h25.json
.venv/bin/python -m vmas_pusht.pairing --checkpoint data/vmas_pusht/wm_full_h_agg2.pt \
  --bank-family timing-steering --horizon 60 --pulse-steps 20 --out data/novelty/pairing_steering_h60.json
```

Tests cover replay of contact and rotational states, replay in the same environment, batch mappings, storage independence, rejection of incompatible configurations, compact-reconstruction error, tied ranks,
matched four-branch interventions, contact geometry, permutation constraints, comparison with brute-force optimal matching, the additivity negative control, candidate norms, and impulses.
Small actual runs also exercised the `true`, `sim_latent`, and `latent --diag` planning paths.
Previously published scores were not rerun and are not combined with this finite diagnostic study into a single benchmark.

Raw outputs occupy approximately 17 MB in the gitignored `data/novelty/` directory. At the time of this audit, the code and this report remained on the local research branch and had not been pushed.
