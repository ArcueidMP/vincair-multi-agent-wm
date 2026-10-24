# Research branch state

## Completed: E4 teammate baseline reproduction

- Base: teammate branch `feature/e4-navigation-baselines`, commit `a4ff32623f4287966aef878d44f01d3521e660af`.
- Personal branch: `codex/e1-ranking-e4-integration`.
- Headless entry point, targeted regression tests and saved-array verifier added; original notebooks and planner functions retained.
- 21 actual episodes: S1 PD/priority/independent CEM plus18 centralized runs on3 geometries x3 planner seeds x2 costs.
- 18/18 centralized runs match teammate cached steps, goal success, step violation counts and minimum separation within1e-5. Latency is hardware dependent.
- Goal success: centralized9/9 for both cost variants. Safe joint success at every native substep: no-safety0/9, soft-safety3/9. All21 native trajectory replays have0 state error.
- Four targeted tests pass, including exact trajectory agreement against original notebook functions. Independent saved-array verification passes.

Small results and trajectory hashes are under `experiments/e4_navigation/reproduction_results/v1/`. Full NPZ trajectories are retained externally; no weights or raw private conversation material is committed.

## Prior local E1 context

MultiPushT mixed-proposal work used24 training scenes and8 new confirmation scenes (4096 labels). A frozen low-dimensional ridge residual improved confirmed global regret by6.7%, with2/8 scene wins. It failed the predeclared closed-loop gate. Readout remains the baseline; this is not a Navigation result. The full prior E1 source/data archive is outside this first E4 reproduction commit and must be migrated with its own provenance, rather than presented as already published here.

## Next complete execution block

1. Rebuild or obtain the original conflict-pilot NPZ and safety-first label files, checking hashes/schema and candidate labels.
2. Reuse the team's native VMAS contract and preserve the original soft-cost baseline. Add safety-first oracle as a diagnostic; do not call it learned performance.
3. Freeze scene/geometry/episode grouping and keep future labels out of model inputs. Random decision windows lacking conflicts are not enough to train avoidance.
4. Integrate learned Navigation rollout/readout only after a matching checkpoint and metadata are supplied. The current notebook adapter raises NotImplementedError. Existing PushT weights are not Navigation weights.

Runs proceed in complete bounded blocks without step-by-step approval; request input for missing essential materials, irreversible actions or a changed resource budget. No Navigation learned-model experiment or three-agent claim is included in this reproduction.


## Completed: E4 conflict data and safety-ranking targets

- Rebuilt the12-state/768-candidate public pilot from the notebook rather than claiming to own the missing original NPZ; all12 cached pilot comparisons match.
- Generated18 new transformed geometries with four conflict offsets each:4608 candidates. Geometry-grouped split:train12 geometries/3072 candidates, validation3/768, sealed test3/768. Shared route families are a limitation; this is not held-out-family transfer.
- Preserved original step soft labels and added initial+native-substep safety labels and deterministic all-unsafe fallback. Train includes one all-unsafe pool; it is explicitly not safe.
- Under substep labels, scene-mean soft-cost unsafe selection conditional on a safe option is75% train and91.67% validation. Safety-first oracle is0% by definition, not a learned result.
- Input and future-supervision NPZs are separate. Train96737 and validation24186 preference pairs are group-correlated. No test preference pairs or method metrics were generated.
- All5376 candidates (including pilot) were rolled out twice with fresh native VMAS, replay error0. Independent cost/safety/order/pair/split/hash checks pass; maximum independent cost difference1.91e-6. Nine targeted unit tests pass.

Source and small audit artifacts: `experiments/e4_navigation/conflict_ranking.py`, `verify_conflict_ranking.py`, `conflict_results/v1/`. Raw NPZs remain external and are identified by hashes. This commit adds no weights or private communications.

The earlier next-block items1-3 are now complete. Next material dependency: Navigation-specific checkpoint, loading API/config/normalization, supported history/frameskip/agents, and a compatible position/velocity state head. Keep the new test split sealed until a scorer and selection rule are frozen. Learned-model training/evaluation and closed-loop safety gains have not been claimed.


## Completed: direct observed-state/action safety scorer baseline

Navigation WM weights are still missing (confirmed by user). A separately named5059-parameter44→64→32→3 MLP was actually trained on current state/relative goals/actions only, without any world-model checkpoint or future-feature input. Train-only normalization and cost statistics; CPU seed0, validation-selected epoch126 and risk threshold0.25; checkpoint/policy frozen before one test activation.

Same-network cost-head-only versus safety-policy ablation: validation conditional unsafe selection16.67%→0%; test16.67%→8.33%. Test soft regret0.6822→0.6987. Predeclared success(>=10 percentage-point reduction AND>=2/3 geometries better) was not met. This is limited direct-scoring evidence, not LeWM transfer, a separately optimized cost-only comparison, or a closed-loop improvement. No post-test refit or closed-loop run.

Sixteen targeted tests pass. Independent NumPy forward reconstruction reproduces all selected test candidates/regrets; maximum prediction difference1.91e-6, training statistics exact, frozen checkpoint unchanged. Source/report/provenance in `experiments/e4_navigation/state_action_results/v1/`; raw model weights and predictions remain external with hashes.

**Test status update:** the geometry test split from conflict_ranking_v1 has now been consumed by this frozen experiment. Earlier statements of sealed status describe the preceding block only. Do not reuse it for threshold/feature/model selection; subsequent changes require new confirmation geometries.

Next material dependency remains Navigation checkpoint, loader/config/normalization, history/frameskip/action/agent contract and a compatible state head. Keep the direct scorer as a distinct baseline when those arrive. Do not present its weights as the missing Navigation world model.


## Completed: failure diagnosis and frozen new-scenario generalization

User requested both blocks1/2; no retraining or risk-threshold changes. Existing epoch126/threshold0.25/checkpoint unchanged. New specifications were frozen before examining old failures.

Old consumed test:12 states;1 pool has no safe candidate and1 avoidable unsafe choice is a false risk acceptance. Seven safe selections exclude the lowest-cost true-safe alternative. Categories are hierarchical diagnostics, not unique causal attribution.

New12 base geometries with angles45/75/105/135deg and new orientations x3 conditions x2 initial distance indices x64 actions =4608 new candidate labels. Paired conditions share EXACT actions/goals. Conditional unsafe selection(cost-head-only vs safety rule):new angle45.83%→37.50%; velocity asymmetry29.17%→37.50%; arrival offset37.50%→45.83%. All72 pools contain safe candidates. Arrival-offset intervention also changes actual separation; it is not fixed-separation timing isolation.

Among29 new unsafe selections,10 are false accepts and19 occur after predicting every candidate unsafe then choosing an unsafe fallback. All19 fallback choices had predicted minimum separation>=0.30 but actual violation; average optimistic minsep error0.1837 VMAS units. Risk filtering and fallback estimates are unreliable under shift. Coordinate/agent-order sensitivity is a hypothesis, not proven by an invariance intervention.

All4608 trajectories/substeps repeated in fresh native VMAS with0 error; independent features, predictions, choices, labels, pairings and aggregates verified.20 targeted tests pass. All12 geometry pairs and deterministic old-test cases are plotted with raw tables/provenance. Source/results under `experiments/e4_navigation/safety_generalization_results/v1/`; full trajectories/predictions remain external with hashes.

These new stress scenarios are now consumed evaluation data, not a new untouched confirmation set. Do not tune current weights/threshold on them. No closed-loop runs, LeWM-transfer claim, model refit or new Navigation weights. Next repair needs training-only selection and NEW confirmation geometries; dedicated Navigation checkpoint/readout remains missing.

## Completed: bounded direct-scorer repair with fresh confirmation

Generated44 NEW geometry groups (train24/validation8/confirmation12), six states each, original64 plus4 fixed open-loop zero/brake proposals:17,952 candidate rollouts. Two fresh VMAS replays agree exactly. No old consumed test or stress files used for selection. Train-only normalization; seed0 MLP training; raw selected epoch56/risk.1/margin.06 and canonical symmetric model epoch63/risk.5/margin0, frozen before confirmation method access.

On original64 candidates, frozen old safety rule unsafe30.56%, fresh raw consistency repair6.94%, invariant consistency repair6.94%. Primary old->invariant reduction23.61pp with9/12 geometry wins passes the predeclared offline gate. Invariant soft regret0.3097 vs fresh raw0.6853. Safety improvement cannot be credited specifically to invariance: new training distribution and selector also changed. Same invariant model with legacy selector6.94% unsafe/regret0.2815 shows no confirmation safety benefit of consistency gating. Appended proposals help old model30.56%->18.06%, but not either repaired model.64/68 use different cost optima and both repair models trained on68 pools.

27 targeted tests pass; independent NumPy reproduces576 choices and all aggregate metrics, all physical labels and train-only moments. Max cost error1.91e-6, forward error2.24e-5, native replay0. Rigid-coordinate and agent-order feature invariance tests pass; numerical prediction translation discrepancy2.10e-5. These are scorer representation properties, not global rotation invariance of clipped VMAS dynamics.

Artifacts: experiments/e4_navigation/safety_repair_results/v1/; full weights and raw NPZ remain external under results/e4/safety_repair_v1, identified by hashes. Confirmation is now consumed: no tuning on it. No closed-loop or Navigation-WM claim. Next block: freeze a NEW bounded paired closed-loop pilot, smoke-test receding-horizon integration first, then evaluate. Navigation-specific checkpoint/loading/normalization/history/action/state-head materials remain missing. Preserve all negative ablations.