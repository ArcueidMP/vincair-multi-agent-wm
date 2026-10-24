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

## Completed: fresh frozen closed-loop direct-scoring pilot

Six NEW geometry groups x2 candidate seeds x5 arms =60 episodes, max200 steps; five separate5-step smoke runs passed first. Native VMAS contract retained. Replan every step with H8/72 pool:64 route/noise,4 prior zero/brake and4 current-PD proposals; learned models and thresholds frozen. Same initial states and seed schedule, but policy-dependent pools diverge with visited states. PD has no pool; true simulator reference has privileged rollout access.

Joint goal/safe joint success/any substep separation violation:PD100%/0%/100%; original scorer0%/0%/8.33%; raw repair0%/0%/16.67%; invariant repair0%/0%/0%; true-sim safety reference100%/100%/0%. The predeclared closed-loop gate FAILED: invariant repair avoids violations but does not reach goals. Offline gains have not transferred to complete navigation. Mean final goal distance original4.0615, raw2.7592, invariant0.8059, true-sim0.0618 (post-hoc descriptive diagnostics). Do not attribute failure uniquely to one cause without further intervention.

32 targeted tests pass. Independent audit checks7200 model decisions,664 freshly recomputed native oracle rankings,8728 actual steps and first-action execution, all physical outcomes and equal-geometry aggregates. Max NumPy forward error3.81e-5, oracle-cost formula error1.91e-6, state and native-substep replay error0. Model hashes unchanged. Original pilot wall146.7s excluding independent verification; no retraining or tuning.

Published source/compact results/all60 episode table/figure/verification at experiments/e4_navigation/closed_loop_results/v1/. Raw trajectories/candidates/predictions remain external at results/e4/closed_loop_pilot_v1 with hashes. All six geometries are now consumed evaluation data; do not tune on them. Next block should diagnose goal-progress/cost ranking and proposal coverage using NEW training/validation trajectories before another frozen confirmation. Do not expand long runs. Missing Navigation-specific WM checkpoint/loading/config/normalization/history/action/state-readout still blocks LeWM transfer; no three-agent claim.

## Completed: NEW development-trajectory goal-progress diagnosis

12 NEW geometry groups (8 train/4 validation), frozen invariant and true-sim parent policies:24 episodes,126 fixed-index snapshots,9072 H8/72 candidate labels. Every snapshot rollout repeated in fresh native VMAS with0 state/substep error. No prior closed-loop confirmation/stress/test file was read, no model fitting or policy tuning. Development data, not a new confirmation set.

Safe H8 progress candidates (>0.05 sum-distance reduction and native-substep separation>=0.30) exist at ALL126 snapshots. On validation invariant-policy snapshots, cost miss within surviving filter32.14%, no exclusion of all safe-progress candidates. Local safe-progress selection learned67.86%; true cost inside predicted filter92.86% but unsafe3.57%->7.14%; true safety with predicted cost64.29%; true safety+cost reference100%. On validation true-sim-policy states, filter excludes all safe-progress candidates14.58%. Local substitutions prioritize cost/progress repair while preserving safety checks; they do not identify a unique cause or establish a new controller's closed-loop success. Counterfactuals and different state-origin rates are not additive causal effects.

37 targeted tests pass. Independent NumPy verifies630 counterfactual choices, all9072 physical labels and forward predictions, parent/snapshot matches, hashes and split/origin/equal-geometry aggregates. Max cost discrepancy1.91e-6, forward4.58e-5. Initial independent RMSE discrepancy1.7e-8 traced to NumPy scalar promotion; matched production scalar precision without changing results or check tolerance. Weights unchanged. Run wall101.8s excluding independent audit.

Source and complete decision tables:experiments/e4_navigation/progress_diagnosis_results/v1/. Raw parent trajectories and separated snapshot inputs/labels/predictions remain external in results/e4/progress_diagnosis_v1, identified by hashes. Next:freeze a cost/progress repair using only these8 train groups, validation4 for selection, retain frozen risk comparator, then activate NEW confirmation geometries. Keep72-proposal baseline unchanged for attribution. Navigation-specific LeWM weights/loader/config/normalization/history/action/state-readout still missing; no LeWM or three-agent performance claim.

## Completed: frozen goal-cost correction and NEW closed-loop confirmation

Fit92-feature weighted ridge residual on previous development8 training geometries/85 snapshots/6120 candidates; validation4 geometries/41 snapshots/2952 candidates select lambda10 from six fixed values. Train-only weighted moments; equal geometry/origin/snapshot/candidate weighting. Goal-action cost target excludes soft safety cost. Frozen risk/minsep heads, threshold, margin, fallback and72 candidates unchanged. Hand-coded native free-motion goal-cost baseline ignores inter-agent contact forces and is not LeWM.

NEW6 confirmation geometries x2 candidate seeds x4 arms =48 episodes after four5-step smoke runs. Safe goal arrivals:original invariant0/12, analytic-goal plus frozen risk12/12, trained ridge0/12, true-sim reference12/12. No arm has separation violation or collision proxy in this small confirmation. Mean steps analytic65.25 vs true-sim55.25; learned arms hit200. The predeclared primary RIDGE gate FAILED; do not substitute the analytic success for that gate. Analytic comparator was separately predeclared. All6 geometries now consumed; no post-confirmation tuning.

Validation:analytic unsafe11.90%/safe-progress80.80%; original unsafe5.95%/safe-progress61.01%; selected ridge unsafe0%/safe-progress53.57%. Analytic validation risk evidence prevents general safety claims despite12/12 fresh closed-loop success. Related synthetic two-agent geometry family and only6 independent scenes; two seeds are nested repeats. No three-agent/learned Navigation WM claim.

42 targeted tests pass, including free-motion recurrence vs native contact-free rollout, agent symmetry and unchanged risk heads. Audit reconstructs train-only weighted moments exactly, ridge normal equation residual2.00e-14, every validation grid entry/selection,5583 model choices,663 fresh oracle rankings,6246 executed steps and physical/aggregate outcomes. Max feature formula difference4.77e-7, NumPy prediction difference1.83e-4, oracle-cost error1.91e-6, native state/substep replay0. Auditor float32 vs Python-float validation averaging mismatch was resolved by matching arithmetic, without changing outcomes or tolerances.

Published source/results:experiments/e4_navigation/navigation_cost_results/v1/. Raw coefficients/model weights and trajectory/candidate/prediction NPZs remain external at results/e4/navigation_cost_repair_v1 and prior runs, identified by hashes. Next:retain analytic hybrid as successful development baseline and failed ridge as negative control; freeze NEW bounded robustness cases before broader claims, including speeds/timing/terminal neighborhoods. Do not tune on these confirmation scenes. Navigation-specific LeWM checkpoint/loading/config/normalization/history/action/state-readout still not obtained in this block.

## Completed: frozen hybrid robustness and collaborative publication

72 NEW robustness episodes:6 independent base geometries x4 correlated conditions(nominal,fast approach,arrival offset,terminal neighborhood) x3 arms, one candidate seed per base. Analytic-goal+frozen risk24/24 safe goal arrivals; true-sim reference24/24; original invariant scorer0/24. All72 episodes have0 substep separation violation/collision proxy. Screen gate passes, but only6 clusters; terminal cases are final approach, not crossing conflict; arrival intervention also changes speed/separation. Earlier analytic offline unsafe-selection11.90% remains a limitation. No training/threshold/proposal/physics changes, no LeWM/three-agent or global safety guarantee.

46 targeted tests pass. Audit verifies5897 model choices,964 fresh oracle rankings,6861 executed steps and per-condition aggregates. Max NumPy prediction difference6.10e-5, oracle-cost1.91e-6, native state/substep replay0. JSON exports only our5059-parameter direct risk scorer; every parameter/statistic matches source checkpoint and all three portable5-step smoke state/action/candidate/prediction arrays match exactly. Portable model JSON SHA256 fe57201a74edee4b8b29b7865c58fb3dc477d9bb50b4ada5392814f78030c527. Teammates can independently reproduce the current E4 hybrid without local author-weight paths. Full separate-machine72-episode reproduction is still pending.

Team entry:COLLABORATION_START_HERE.md and experiments/e4_navigation/REPRODUCE_SHARED.md. Latest results:experiments/e4_navigation/robustness_results/v1/. Historical migration adds23 selected E0/E1 reports/source files plus manifest; all10 final E1 phase22 sources match original frozen hashes. Reports only normalize UTF-8/LF, original files retained. Historical trials not rerun here, and complete raw arrays/authors' weights/old environments/earlier source stages remain external. Do not claim a complete clone reproduces all history.

Next P0:teammate clone and portable smoke/full paired repeat, review metrics/provenance and prepare team PR review. P1:obtain Navigation WM material contract, integrate with loading/rollout checks and compare with current analytic and true-sim references. P2:shared large-artifact index/storage and legacy E0/E1 portability. Main and teammate feature branch remain unchanged; no PR/merge or teammate messages automatically created. New robustness bases and variants are consumed evaluation data; no tuning on them.

## Completed: team readiness, main synchronization and NEW pressure boundary

A NEW clean GitHub clone of published d2d0af7 ran46 tests and the complete72-episode portable robustness suite on the SAME machine/existing environment. All episode metrics match;720 array fields match exactly, hashes/replay verified. This is not cross-machine independent reproduction or fresh scientific confirmation. Teammate-facing comparison CLI and five new readiness tests added;51 total tests pass after synchronizing main's a5c2d76 readout commit into the personal branch. Main/teammate feature refs unchanged; no PR/teammate messages created. Upstream synchronization preserves a5c2d76 as merge parent.

NEW pressure6 bases x2 related variants x3 arms=36 episodes. Analytic hybrid near-conflict6/6 safe joint arrival; high-inertia6/6 goal arrivals but0/6 safe, all6 separation violations. True-sim reference12/12 safe. Original frozen scorer0/12 goal arrival, high-inertia6/6 violations. No collision proxies. Stress screen FAILS, limiting prior24/24 success. Every high-inertia initial pool has1..4 true-safe H8 options, but analytic initially selects an accepted unsafe H8 plan; predicted minima.424..486 vs actual.267..299. Actual violations begin at execution step1/2. Local H8 labels alone do not prove first-action unavoidability. No tuning on these consumed6 bases/variants.

Independent pressure audit:3087 model choices,543 fresh native oracle rankings,3630 executed steps, paired initial pools and all physical/condition metrics. Max NumPy forward difference4.58e-5, oracle-cost1.91e-6, state/substep replay0; portable model unchanged. Source/results:experiments/e4_navigation/pressure_results/v1/. Raw NPZ remain external.

Read-only external asset index snapshots3723 scientific files/345669154 bytes in selected results/e0,e1,e4 extensions.3602 freshly hashed,96 declaration-only,25 unhashed; author manifest6 model/config items plus2 readouts freshly checked. Scan predates new pressure completion and excludes data/datasets/cache/private images. No new raw-data/model upload or storage URL invented.

Deliverables:docs/WORK_SUMMARY_2026-10-25.md, docs/PR_REVIEW_READY.md, docs/ASSET_SHARING.md, team_readiness_results/v1/ comparison and asset indexes. Pending external tasks:teammate cross-machine run/review, Navigation WM delivery, shared-storage decision and full historical-asset/path migration. Next scientific block needs NEW risk-calibration development data before another frozen confirmation; current pressure failures cannot tune that repair.
