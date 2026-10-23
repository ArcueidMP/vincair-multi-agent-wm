# Recovery disturbance attribution and closed-loop simulator recovery

2026-10-23, `codex/novelty`. This round contains two diagnostics only. No world model (WM) was retrained or queried; existing models, physics parameters, recovery controllers, and earlier experimental artifacts were preserved.

**Result: recovery overshoot is driven mainly by helper actuation and the closed-loop interaction between the agents. Adding physically executed side changes and reverse contact raises true-simulator MPC recovery from 0/108 to 108/108 among cases that passed the first goal.**
All 108 successes persist when the same actions are replayed at 320 substeps. The original handoff states can therefore be corrected within the original remaining time and force limits. The old action interface imposes a substantial restriction; this does not yet establish a reusable physical abstraction for the frozen small WM.

## 1. Separating recovery actuation from identical retreat endpoints

We use all 216 cases from the previous `dry_light` experiment at 160 substeps, including first-goal failures and passive controls.
Starting from the original complete post-prelude snapshots, we replay the recorded actions in the original two batches of 108, preserving row order, and capture each case's complete physical state at its original retreat endpoint.
Both batches replay 64 steps with bitwise equality of all 24 physical coordinates at every recorded time. The compact model observation is never used as a simulator snapshot.

Each real initial state is branched four ways, retaining the original `RebraceController` staging, acquisition, and confirmation logic:

| Branch | Primary actuator | Helper actuator |
|---|---|---|
| A | Off | Off |
| B | On | Off |
| C | Off | On |
| D | On | On |

Off means setting that agent's final drive command to zero. No body is removed, locked, or teleported; inertia and collisions remain active.
The original support semantics are also preserved: passive support still applies zero helper drive after the controller enters confirmation.
Active support is therefore the primary attribution stratum, with passive cases reported separately rather than treated as an identical intervention.

Every branch receives the same 80-step budget and unit force-norm limit. Both the 160- and 320-substep executions start from the identical complete state captured at 160 substeps, giving **216 × 4 × 2 = 1,728 trajectories**.
At every executed integration substep, we record agent–bar/stem contact normal forces, impulses, contact torques, actions, displacements, and work.
Work is the dot product of the actual actuator force and that substep's displacement; signed, absolute, positive, and negative work are accumulated separately. Values are in scene units, not hardware-calibrated joules.

Two endpoints are reported:

- **Historical D handoff time:** all branches of a source case are compared at that case's original recovery duration. This is the primary endpoint for attributing the historical overshoot.
- **Fixed 80-step endpoint:** all branches continue recovery/holding to the same budget, exposing longer-term feedback effects.

The second endpoint does not replace the first. Both endpoints and every failed case are retained.

### Main results: 75 active-support cases that passed the first goal

All angular increments below are measured from the identical retreat endpoint and evaluated at the historical handoff time. Original recovery durations in this stratum were 14–16 steps.

| Branch / effect | 160 substeps, mean rad | 320 substeps, mean rad |
|---|---:|---:|
| A | −0.000394 | −0.000242 |
| B | 0.004303 | 0.004185 |
| C | 0.062257 | 0.062233 |
| D | 0.099738 | 0.099787 |
| B−A: conditional primary effect | **0.004696** | **0.004427** |
| C−A: conditional helper effect | **0.062651** | **0.062475** |
| D−B−C+A: interaction effect | **0.032785** | **0.033126** |

**Historical handoff overshoot is driven mainly by the helper and amplified by the agents' closed-loop interaction; the primary's isolated repositioning effect is smaller.**
These are finite intervention results for the specified controller and initial-state distribution, not intrinsic properties of every recovery policy.
The interaction term includes both mechanical coupling and intervention-induced changes in controller phases and feedback. It is not purely an open-loop force-superposition term.

Within this time window, D's mean angular increment decomposes by its own controller phase into staging −0.000065, acquisition 0.055025, and confirmation 0.044779 rad.
The mean peak angular speed is approximately 0.277 rad/s, occurring during acquisition about 0.789 seconds after recovery starts.
Disturbance thus continues through contact acquisition and confirmation, rather than occurring only during free-space positioning.

At the fixed 80-step endpoint, C's mean angular increment reaches 0.166015 rad, compared with D's 0.114376 rad; the interaction term becomes −0.058935 rad.
C still has 72/75 cases in acquisition, whereas D has confirmed contact in every case. The interaction's sign at 80 steps therefore cannot replace the conclusion at the original handoff time.

### Numerical and replay limitations

At 160 substeps, D's bar angle at the historical handoff matches the earlier trajectories; the maximum discrepancy across all 24 state coordinates is approximately 7.3e−7.
All branches remain finite and avoid walls. Mean attribution is stable between 160 and 320 substeps, but **2/75** D cases in the first-goal-qualified active-support stratum are sensitive to closed-loop gating.
Their maximum whole-trajectory angular discrepancy is approximately 0.01209 rad, with approximately 0.00543 rad at the historical handoff. We cannot claim that every closed-loop case meets the original `.003 rad` threshold.

Complete states, actions, and substep measurements are in [recovery_attribution.pt](../data/novelty/recovery_attribution.pt).
Per-case endpoints, factorial effects, and resolution comparisons are in [recovery_attribution.json](../data/novelty/recovery_attribution.json).

## 2. Testing recoverability from original handoff states

The inputs remain all 216 complete post-recovery snapshots from the previous experiment. No new preparation is applied.
Each case retains its original goal reference, first-goal outcome, and elapsed step count. A successful continuation cannot erase an earlier first-goal failure.
Events in the original trajectory after the handoff are excluded from the new continuation's prefix safety score.
The two goal sequences remain `.04→.08 rad` and `.06→.12 rad`. These are local bar-pose composition microtasks under fixed designed conditions, not estimates of success on the repository's original full Push-T benchmark; cases are not treated as independent random samples.

Both libraries use the same **60 remaining real steps, unit force-norm bound, original helper control law, and frozen helper anchor**:

- `bank13`: the original zero action plus directions −20/0/20° × strengths 0.15/0.3/0.6/0.9.
- `expanded`: all original actions plus two fixed finite-force routes: withdraw upward from the upper-left contact, move to the upper right, and push downward; or travel around the left exterior to the lower left and push upward. Contact-phase directions and strengths still come from bank13, oriented for the physically acquired face.

Every side change uses ordinary `env.step` calls. Withdrawal, transit, approach, contact confirmation, and braking after a failed route all consume real time, actuation, and work.
The helper anchor is not updated for the route, and no body is locked or teleported.

### MPC search scope

This is closed-loop true-simulator MPC, rather than the earlier one-shot oracle with a 5-step lookahead:

1. Every 5 real steps, predict **all remaining time** from the current complete physical state.
2. Each candidate contains an initial 5 contact-ready pushing steps, followed by zero drive, repeated pushing, or angle feedback quantized to bank13.
3. Expanded candidates simulate the complete side-change route before evaluating the subsequent pushes, so short lookahead does not automatically discard a maneuver that requires positioning first.
4. Execute at most 5 steps, then replan. Preserve route progress, the acquired contact face, and any force macro that began between query times and has not yet completed its 5 steps.
5. Success still requires 5 consecutive real post-step states within the 0.015 rad angular tolerance and 0.05 translation limit, with a valid terminal pose, finite states, and no wall contact.

There are 39 finite templates per baseline query and 117 per expanded query. Physical execution budgets are equal, but search computation is not; this is not an equal-inference-compute comparison.
Each expanded trajectory can execute at most one of the declared side changes, rather than search arbitrary contact-mode sequences.
Failure of this finite-template MPC does not prove global unreachability.

The same 216 source cases are partitioned into four batches of 54. Both libraries use identical partitions, evaluated in eight independent processes.
This changes computational batching only: the MPC implementation and candidate plans are unchanged, and every case is retained. The original batch-of-216 baseline is preserved as a batching-sensitivity control.
The large-batch expanded run was stopped before completion because of computation time; its partial results were not used to select parameters or remove cases, and its interruption log is retained.
Numerical replay preserves each actual execution batch and its row order. Different batches are not merged before claiming bitwise reproduction.

The two routes were validated in an independent stationary development scene without reading composition-task outcomes. They acquired actual contact after approximately 26/32 steps.
For the upper-right route at strength 0.9, net rotation after 60 steps was approximately −0.0644 rad with translation 0.0133; the lower-left route at strength 0.3 produced approximately −0.0251 rad and 0.0317 translation.
Stronger lower-left pushing caused excessive translation and was retained as a failure; scoring thresholds were not relaxed.

### Composition outcomes and numerical checks

In the original large-batch bank13 run, 22 of 216 cases reached the final goal, but all had failed the first goal earlier.
Complete ordered success was 0/216, and recovery among the 108 first-goal-qualified cases was also 0/108.
Replaying identical actions at 160/320 substeps gave maximum active-prefix position and angular discrepancies of 0.000368 and 0.001065 rad, within the `.001/.003` thresholds.
All 22 continuation successes persisted, while complete ordered success remained zero. This was fixed-action replay, not a new closed-loop MPC run at the finer resolution.

The main comparison uses the four matched batches. Complete initial snapshots, per-case goals, and remaining budgets were checked field by field for equality between the libraries.

| Scoring population | Original bank13 MPC | Expanded contact-primitive MPC |
|---|---:|---:|
| All 216 cases: final continuation only | 22/216 | **214/216** |
| All 216 cases: complete ordered success, retaining the first-goal gate | 0/216 | **108/216** |
| The 108 cases that passed the first goal: successful recovery | 0/108 | **108/108** |
| Active-support subset: 75 cases | 0/75 | **75/75** |
| Passive subset: 33 cases | 0/33 | **33/33** |

Active-support recovery is 41/41 for world_anchor and 34/34 for preloaded_anchor.
All 108 cases physically acquired reverse contact: 84 used the upper-right route and 24 the lower-left route. Braking after route failure was not counted as successful side acquisition in this group.
All 108 terminal angles show negative correction relative to the handoff, with a mean of −0.05133 rad and a range of approximately −0.06957 to −0.02093 rad.

The two remaining continuation failures both came from preloaded_anchor / legacy_wm initial states that had already failed the first goal.
Their final angles were accurate enough, but translations of 0.05279/0.05260 exceeded the original 0.05 limit. They were not relabeled as successes by relaxing that limit.

The large-batch and matched-batch bank13 runs have identical per-case success labels and stopping steps.
The largest differences in terminal angular-error and translation metrics are approximately 4.8e−7 rad and 1.4e−7, respectively, leaving the baseline conclusion unchanged.

### Actual correction costs

For the same 108 first-goal-qualified cases, expanded-library execution from the original handoff to completion has the following costs:

| Resource / error | Measurement |
|---|---|
| Real continuation steps | Median 34, mean 37.56, range 29–53; all below 60 |
| Acquiring reverse contact | 26–32 steps, mean 27.33; included in the continuation budget |
| Mean primary / helper path length | 0.55085 / 0.01173 scene units |
| Mean primary / helper actuator-force norm integral | 2.57745 / 0.71703 |
| Mean primary / helper absolute actuator work | 0.39913 / 0.00364 scene work units |
| Final angular error | Mean 0.001345 rad, maximum 0.005253 |
| Final translation | Mean 0.02995, maximum 0.03972, below 0.05 |

All executions remain finite and avoid walls. The maximum action norm is 1.00000012, within floating-point tolerance of the unit bound.
Recontact requires real travel and work, so it is not a free action switch. An equal-progress energy advantage also cannot be claimed when the baseline does not complete the task.

### Same-action verification at 160/320 substeps

Each actual batch of 54 is replayed independently. At 160 substeps, all 24 physical coordinates are reproduced bitwise, as are all original scoring labels.
At 320 substeps, the last five active states and terminal pose are evaluated at the **same original stopping time**:

| Check | bank13 | Expanded primitives |
|---|---:|---:|
| Maximum active-prefix position discrepancy | 0.000368 | 0.000301 |
| Maximum active-prefix angular discrepancy, rad | 0.001065 | 0.001263 |
| Continuation successes, 160→320 | 22→22 | **214→214** |
| Complete ordered successes, 160→320 | 0→0 | **108→108** |

Both pass the `.001/.003` thresholds, with no lost or gained successes. Maximum joint position/angular residuals in the expanded trajectories are approximately 0.004369/0.056105, also below the previously declared 0.01/0.08 limits.
This verifies the saved joint-action sequences; it does not claim a closed-loop MPC run at 320 substeps.
The object remains VMAS's finite-stiffness bar/stem assembly, and scoring uses bar pose. Passing the numerical checks is not validation of an ideal rigid body or real hardware.

Main results: [bank13](../data/novelty/recovery_mpc_parallel_bank13.json), [expanded primitives](../data/novelty/recovery_mpc_parallel_expanded.json).
Numerical audits: [bank13](../data/novelty/recovery_mpc_parallel_numerics_bank13.json), [expanded primitives](../data/novelty/recovery_mpc_parallel_numerics_expanded.json).

## 3. Supported conclusions

This round provides constructive evidence: for all 108 real handoff states whose first goal had already succeeded, **expanded contact primitives suffice to recover within the original 60 remaining steps and unit force bound**.
The earlier composition failure therefore cannot be interpreted as physical irrecoverability. Overshoot produced by the current recovery protocol does place a substantial demand on the old control interface.
Failure of the tested long-horizon bank13 MPC under the same physical budget supports prioritizing contact modes, reverse primitives, and recovery transitions before simply increasing WM capacity.

Three boundaries remain:

- bank13 uses a finite template search. Its 0/108 result is not an unreachability proof over all long bank13 action sequences. Search computation also differs between libraries.
- The expanded planner uses complete physical state and extensive true-simulator prediction. It establishes that solutions exist, not that a small WM can predict or select them.
- Passive cases also achieve 33/33 recovery, so these gains cannot be uniquely attributed to cooperation reducing model complexity.

The stronger claim still requires a learning experiment: make contact modes and recovery transitions explicit, freeze the local WM/helper, then vary the primary planner, goals, and primitive compositions.
Additional sensing, memory, planning computation, and physical travel must be counted together. As requested, this round did not perform that retraining.

## Implementation and reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.recovery_attribution
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.recovery_mpc_batch
# Replay each original batch/order independently; use separate output prefixes for the two libraries.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.recovery_mpc_numerics \
  --sources data/novelty/recovery_mpc_parallel_bank13_batch{0,1,2,3}.pt \
  --out data/novelty/recovery_mpc_parallel_numerics_bank13
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.recovery_mpc_numerics \
  --sources data/novelty/recovery_mpc_parallel_expanded_batch{0,1,2,3}.pt \
  --out data/novelty/recovery_mpc_parallel_numerics_expanded
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

These commands require the previous round's complete local trajectories in `data/novelty/stable_composition_selected_matched.pt`, which can be regenerated using that audit's commands.
Outputs live under the gitignored `data/novelty/` directory. Use separate output prefixes to preserve current artifacts.

New implementations: [recovery attribution and read-only substep observation](../vmas_pusht/recovery_attribution.py),
[finite-force contact routes](../vmas_pusht/recovery_primitives.py), [true-simulator MPC](../vmas_pusht/recovery_mpc.py),
[matched-batch execution](../vmas_pusht/recovery_mpc_batch.py), and [fixed-action numerical replay](../vmas_pusht/recovery_mpc_numerics.py).

The [summary and hash audit](../data/novelty/recovery_experiments_summary.json) contains 97 checks of source, input, model, and artifact hashes, with no mismatches.
Complete initial snapshots match between libraries, and both original WM model packs remain unchanged. **168 tests pass.**
