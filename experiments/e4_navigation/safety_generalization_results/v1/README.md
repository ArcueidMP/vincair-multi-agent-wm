# Frozen scorer: failure diagnosis and new-scenario generalization

## Completed blocks and conclusions

1. Post-hoc diagnosis of the already-consumed validation/test data: old test has12 states, including one all-unsafe pool and one avoidable unsafe selection. The avoidable failure is a false risk acceptance; seven other states select a safe action while excluding the lowest-cost true-safe action. Categories follow a fixed hierarchy and may involve interacting mechanisms, not unique causal explanations.
2. Frozen new-scenario evaluation:12 base geometries x3 conditions x2 distances x64 candidates =4608 new labeled candidates. Existing checkpoint epoch126/threshold0.25 is unchanged, no optimizer, retraining or threshold search. New geometry specifications were saved before old-case inspection. Both post-hoc diagnostics and new results are distinct from model-selection data.

| New condition | Cost-only unsafe* | Safety-rule unsafe* | No safe pool | False-accept selections | Unsafe fallback selections |
|---|---:|---:|---:|---:|---:|
| angle_orientation | 45.83% | 37.50% | 0 | 3 | 6 |
| velocity_asymmetry | 29.17% | 37.50% | 0 | 3 | 6 |
| arrival_offset | 37.50% | 45.83% | 0 | 4 | 7 |

*First average two states within each base geometry, then equally across12 base geometries. All72 states contain at least one safe candidate. In all three conditions combined there are10 false-accept unsafe selections and19 unsafe fallback selections. Thus current new-scenario unsafe decisions are not explained by missing safe proposals. Baseline8.33% from the earlier test does not transfer: new-condition rates37.5/37.5/45.83%. Scenario distributions differ, so these are stress/generalization comparisons, not a randomized estimate of one causal factor.

Within a base geometry and distance-grid index, candidate actions and goals are EXACTLY shared across the three conditions. Velocity condition changes only initial velocity; arrival condition redistributes approach offsets while preserving velocity. The latter also changes instantaneous initial separation, so it is not a timing intervention at fixed separation; actual separation is exported in the new-case CSV. Across base geometries, angles45/75/105/135deg and full-circle orientations vary. All initial separations exceed0.30. These are offline H8 tests, not closed-loop flights or three-agent/3D validation.

## Mechanism checks

`true_risk_predicted_cost` substitutes future true safety for the risk filter; all conditional unsafe rates become0 by construction. `predicted_risk_true_cost` substitutes true soft cost but keeps the predicted risk filter/fallback; unsafe rates remain high and can worsen because the soft objective itself favors some unsafe actions. These are explicitly oracle counterfactual diagnostics, not deployable methods or additive causal attribution.

The model can both reject real safe candidates and overestimate fallback quality.19/29 new unsafe decisions occur after predicting no candidate safe, even though a safe option exists. In19/19 such failures, the selected predicted minimum separation is nevertheless>=0.30, with mean optimistic error0.1837 VMAS length units. This is direct evidence of unreliable fallback estimates, not a guaranteed-safe rejection mechanism. Purely increasing soft-cost accuracy or adding a safety penalty is not an established repair. Coordinate/agent-order sensitivity and narrow training coverage are hypotheses suggested by raw44D input and changed distributions; no rotational-invariance intervention was performed, so they are NOT proven causes.

Recommended next repair block: keep readout/true-simulator controls, predefine invariant/relative state-action representations and explicit safe-fallback coverage, select using training-only splits, and obtain NEW confirmation scenarios. Do not tune the current checkpoint or threshold on these now-consumed stress scenarios. Navigation-specific WM materials remain missing; this is a direct scorer baseline, not LeWM transfer.

## Validation and evidence

All4608 native VMAS candidate trajectories/substeps were run twice from fresh environments, error0. Independent NumPy checks recompute feature44D, cost and safety labels, forward inference (max error3.81e-06), selected decisions, paired actions/goals and scene-mean rates; original checkpoint hash unchanged. Twenty unit tests pass. Cost formula max CPU/GPU reduction difference1.91e-06; tolerance1e-5. No rows were excluded.

`generalization_pairs.svg` shows every geometry pair, with black mean lines and fixed0–100% context. `old_test_cases.svg` uses the predeclared first-name selection rule (avoidable unsafe, all-unsafe, safe), not favorable case selection. Native-substep paths are H8 snippets, not complete goal trajectories; true-safe alternatives are oracle references. Underlying case CSVs, source protocol, figure hashes and full raw external artifacts are preserved. Raw trajectories, predictions and weights are not committed to Git.

## Reproduce

```bash
python -m unittest discover -s tests -v
python experiments/e4_navigation/safety_generalization.py --data outputs/conflict_ranking_v1 --model outputs/state_action_ranker_v1 --output outputs/safety_generalization_v1
python experiments/e4_navigation/report_safety_generalization.py --run outputs/safety_generalization_v1 --model outputs/state_action_ranker_v1 --old-data outputs/conflict_ranking_v1
```

CUDA VMAS labels and CPU scoring use the unchanged prior environment. The prior checkpoint and its portable NPZ are external prerequisites; a new output directory is required. This block consumes the new evaluation scenarios, not a new training set. No closed-loop experiments were launched.
