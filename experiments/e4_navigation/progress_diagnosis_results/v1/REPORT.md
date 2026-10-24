# E4 navigation progress diagnosis on NEW development geometries

## Result and scope

On all126 sampled states there is a safe H8 candidate reducing total goal distance by more than0.05 VMAS units. This establishes local candidate coverage at these snapshots only, not complete trajectory feasibility or closed-loop success.

For validation states visited by the frozen invariant scorer,32.14% retain a safe progressing option through its predicted filter yet select a non-safe-progressing candidate. Replacing only the cost ranking with true cost increases safe-progress selection67.86% ->92.86%, but unsafe choices also rise3.57% ->7.14%. Retaining predicted cost while supplying true safety gives64.29% safe progress. The true-safety/true-cost reference gives100%. These local counterfactuals support prioritizing cost/progress ranking while retaining safety checks; they do not justify replacing the filter or claim a uniquely isolated cause.

On states visited by the true-simulator reference, validation predicted filtering excludes all safe-progress candidates14.58% of the time. Thus filtering remains unreliable on a different state distribution. Diagnostic flags can overlap in general and are not additive causal percentages.

## Coverage and failure flags

Within each split and origin, average snapshots within geometry, then geometries equally. Different origins visit different states and have different numbers of snapshots; no direct paired causal comparison between origins.

| Split/origin | Geometries | Snapshots | Safe progress exists | Filter excludes all progress | Cost miss within filter |
|---|---:|---:|---:|---:|---:|
| train/invariant_repair | 8 | 56 | 100.0% | 0.0% | 42.9% |
| train/true_sim_safety | 8 | 29 | 100.0% | 16.7% | 15.6% |
| validation/invariant_repair | 4 | 28 | 100.0% | 0.0% | 32.1% |
| validation/true_sim_safety | 4 | 13 | 100.0% | 14.6% | 31.2% |

Safe progress means native-substep minimum separation>=0.30 AND sum(current goal distances)-sum(H8 terminal goal distances)>0.05. Cost miss within filter means at least one safe-progress candidate survives the frozen risk/minsep conjunction but the learned choice is not safe-progressing. The criterion includes both unsafe and nonprogressing choices; it is not a count of stationary agents.

## Counterfactual decisions

| Split/origin | Rule | Safe-progress choice | Unsafe choice | Mean H8 progress |
|---|---|---:|---:|---:|
| train/invariant_repair | learned | 57.1% | 1.8% | 0.1259 |
| train/invariant_repair | true_cost_predicted_filter | 96.4% | 3.6% | 0.3574 |
| train/invariant_repair | true_safety_predicted_cost | 66.1% | 0.0% | 0.1992 |
| train/invariant_repair | true_safety_oracle | 100.0% | 0.0% | 0.4140 |
| train/invariant_repair | safe_progress_oracle | 100.0% | 0.0% | 0.4140 |
| train/true_sim_safety | learned | 78.1% | 0.0% | 0.2456 |
| train/true_sim_safety | true_cost_predicted_filter | 86.5% | 7.3% | 0.3397 |
| train/true_sim_safety | true_safety_predicted_cost | 84.4% | 0.0% | 0.3354 |
| train/true_sim_safety | true_safety_oracle | 100.0% | 0.0% | 0.4488 |
| train/true_sim_safety | safe_progress_oracle | 100.0% | 0.0% | 0.4488 |
| validation/invariant_repair | learned | 67.9% | 3.6% | 0.1497 |
| validation/invariant_repair | true_cost_predicted_filter | 92.9% | 7.1% | 0.3393 |
| validation/invariant_repair | true_safety_predicted_cost | 64.3% | 0.0% | 0.2246 |
| validation/invariant_repair | true_safety_oracle | 100.0% | 0.0% | 0.4173 |
| validation/invariant_repair | safe_progress_oracle | 100.0% | 0.0% | 0.4173 |
| validation/true_sim_safety | learned | 54.2% | 8.3% | 0.2396 |
| validation/true_sim_safety | true_cost_predicted_filter | 68.8% | 16.7% | 0.3237 |
| validation/true_sim_safety | true_safety_predicted_cost | 85.4% | 0.0% | 0.4054 |
| validation/true_sim_safety | true_safety_oracle | 100.0% | 0.0% | 0.4894 |
| validation/true_sim_safety | safe_progress_oracle | 100.0% | 0.0% | 0.4894 |

`true_cost_predicted_filter` uses true total soft cost only inside the original predicted-accepted set; when that set is empty, it keeps the frozen learned fallback. `true_safety_predicted_cost` chooses predicted cost inside the true-safe set, keeping learned fallback if none exists. `true_safety_oracle` uses true safety-first ordering with original soft cost; `safe_progress_oracle` maximizes true progress within true-safe candidates. No counterfactual was run as a new closed-loop controller. No future label was used by the learned parent policy.

## Parent trajectories and sampling

12 newly generated geometries (seed2026104901):8 train/4 validation whole groups. Each has one200-step-budget trajectory under the frozen invariant model and one under the existing privileged true-sim reference, using a fixed candidate seed schedule. Snapshot execution indices0,10,30,60,100,150,199; indices after termination are skipped.24 parent episodes, 126 snapshot pools x72 = 9072 candidate labels, H8. Approach distance0.6..0.9, crossing angles40..140deg, random orientation, initial route speeds0.08..0.25. Native VMAS contract unchanged. These are development data; there is no new untouched confirmation set in this block.

| Split/origin | Episodes | Joint goal success count | Safe joint success count |
|---|---:|---:|---:|
| train/invariant_repair | 8 | 0 | 0 |
| train/true_sim_safety | 8 | 8 | 8 |
| validation/invariant_repair | 4 | 0 | 0 |
| validation/true_sim_safety | 4 | 4 | 4 |

No prior consumed closed-loop/stress/test data were opened for fitting or selection. Weights and policies remained frozen; no training, threshold search, or successful learned-world-model navigation claim. The source code and protocol fix sampling and diagnostic definitions before generation. The new data may support later development fitting, but validation labels must not be used as training labels; a later repair requires new confirmation geometries.

## Verification and provenance

37 targeted tests pass. Independent NumPy recomputes all physical cost/safety/progress labels, model forward predictions and630 counterfactual decisions; parent/snapshot input matching, hashes, episode goal/safety outcomes and equal-geometry aggregates pass. Maximum cost formula discrepancy1.91e-06, forward discrepancy4.58e-05. Every snapshot candidate rollout repeated in fresh native VMAS with0 state/substep error; every parent executed trajectory replayed exactly. An initial independent RMSE check differed by1.7e-8 because portable NumPy scalar arithmetic promoted precision; the auditor now explicitly matches the producer's Python-scalar/float32 arithmetic. No result or tolerance was changed to bypass that check.

Models remain external in safety_repair_v1/state_action_ranker_v1. Raw parent trajectories, separated snapshot inputs/future labels/predictions remain in this external run directory, identified by SHA256. Source, protocol, summaries, all counterfactual decisions, parent outcomes and verification are published. Runtime101.8s excluding independent verification. No Navigation-specific LeWM checkpoint was obtained or used.

## Next executable block

1. Freeze a goal-progress/cost repair experiment using only this block's train geometries for fitting and validation geometries for selection. Keep the current risk model as a frozen comparator. Compare a trained cost correction with a simple analytic goal-progress baseline, using the same candidates and explicit safety/goal metrics.
2. Include development states visited by both learned and privileged policies; track the unequal snapshot density and weight geometries explicitly. Separate current-state/action inputs from future supervision. Validate one-step versus H8 ranking and receding-horizon progress rather than cost RMSE alone.
3. Freeze selected model/policy before generating and activating NEW confirmation geometries; test bounded closed-loop safe goal success, not just local safety. Preserve original72 candidate baseline and isolate any proposal change as its own ablation.
4. Navigation LeWM checkpoint, loader/config/normalization/history/action/agent contract and state readout remain pending from teammates. Keep all direct-scoring baselines separate from LeWM transfer and three-agent claims.
