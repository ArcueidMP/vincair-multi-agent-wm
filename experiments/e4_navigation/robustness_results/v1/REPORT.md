# E4 frozen hybrid robustness and collaborative reproduction

## Result

Descriptive robustness screen **passed**. Across24 condition-specific episodes, analytic free-motion goal cost plus the frozen learned risk filter reaches goals safely24/24; frozen original scorer0/24; privileged true-sim safety reference24/24. Each condition has6 episodes from the SAME six base geometries. Treat them as6 clusters with four related variants, not24 independent geometries. No population-level safety, LeWM transfer or three-agent claim.

| Condition | Arm | Joint goal | Safe joint goal | Any separation violation | Mean steps |
|---|---|---:|---:|---:|---:|
| nominal | invariant_repair | 0.0% | 0.0% | 0.0% | 200.00 |
| nominal | analytic_goal | 100.0% | 100.0% | 0.0% | 62.33 |
| nominal | true_sim_safety | 100.0% | 100.0% | 0.0% | 56.83 |
| fast_approach | invariant_repair | 0.0% | 0.0% | 0.0% | 200.00 |
| fast_approach | analytic_goal | 100.0% | 100.0% | 0.0% | 63.50 |
| fast_approach | true_sim_safety | 100.0% | 100.0% | 0.0% | 54.50 |
| arrival_offset | invariant_repair | 0.0% | 0.0% | 0.0% | 200.00 |
| arrival_offset | analytic_goal | 100.0% | 100.0% | 0.0% | 53.33 |
| arrival_offset | true_sim_safety | 100.0% | 100.0% | 0.0% | 45.67 |
| terminal_neighborhood | invariant_repair | 0.0% | 0.0% | 0.0% | 200.00 |
| terminal_neighborhood | analytic_goal | 100.0% | 100.0% | 0.0% | 3.67 |
| terminal_neighborhood | true_sim_safety | 100.0% | 100.0% | 0.0% | 3.67 |

All72 episodes have zero separation violations and collision proxies. Original scorer failure is lack of goal arrival despite200-step budgets. Arrival-offset changes both start positions and speeds; it is not a pure timing intervention at fixed geometry/separation. Terminal-neighborhood starts0.20 before each goal and tests final approach; it is not a crossing-conflict trial. Position-only success tolerance0.12 has no terminal-speed requirement. Report all conditions, not just the fast/nominal cases.

## Frozen protocol

Seed2026105101 generates6 NEW base geometries, crossing angles45..135deg and full-circle orientations. Conditions:nominal velocities0.12/0.24; fast0.45/0.60; arrival offsets0.90/0.45 with velocities0.20/0.20; final approach0.20 before goals with velocities0.30/0.45. One candidate RNG seed per base shared across conditions/arms, plus execution step. Actual candidate pools diverge with policy state. H8/72 candidates,200 maximum steps, native VMAS1.5.2 dt0.1/two substeps/radius0.1/world drag0.25. Safety includes initial and native substeps, threshold0.30. No training, thresholds/margins/proposals/physics changes or old consumed evaluation files read.

Three independent5-step smoke runs preceded72 full episodes. Both learned-risk arms share the same frozen5059-parameter direct state/action scorer; analytic arm changes only goal-cost ranking through known free-motion recurrence, ignoring contact forces. True-sim arm has privileged candidate rollouts and is not compute/information matched to learned arms. Screen criterion was fixed before run:analytic safe-goal rate>=95% in EACH condition and violations no higher than true-sim in EACH condition. This is a descriptive screen on a bounded synthetic set, not proof of universal safety. Earlier offline validation unsafe-selection evidence (analytic11.90%) remains relevant and must not be omitted.

## Verification and reproducibility

46 targeted tests pass. Independent audit checks5897 learned choices and964 fresh native oracle rankings, all6861 executed first actions, physical outcomes and condition aggregates. Native state/substep action replay error0; max NumPy forward difference6.1e-05; independently computed oracle-cost error1.91e-06. Run wall170.8s excluding independent audit.

Portable JSON exports ONLY our locally trained direct risk scorer, with exact source parameter/statistic equality and source hashes. All state/substep/action/candidate/prediction/selection arrays from the three portable smoke runs exactly match the original model runs. Teammates can clone this branch and run [the shared entry point](../../REPRODUCE_SHARED.md) without the author's LeWM checkpoints or our machine-specific weight paths. Complete72-episode independent reproduction on a DIFFERENT machine remains a team task; only local portable smoke was separately rerun here.

Published source, protocol, compact summaries, every episode CSV/JSON, tested environment, verification and hashes accompany the portable model. Raw trajectories/candidates/predictions remain local, indexed by SHA256. Historical E0/E1 reports and exact final E1 phase22 source were separately migrated with original/published hashes; they were not rerun during this publication.

## Next work and collaboration

1. Teammate independently clones personal branch, checks model/physics/version hashes, runs portable smoke and then72-episode repeat; compare every episode outcome/steps/minimum separation, excluding latency from exact matching.
2. Review positive hybrid evidence together with failed trained-ridge and original-scorer controls, and the unresolved offline risk errors. Prepare a team PR review; main and teammate feature branch remain untouched.
3. Obtain Navigation-specific LeWM checkpoint/loading/config/normalization/history/actions/agent contract/state head and dataset splits. Integrate only after real loading and state/rollout checks, then compare against this hybrid and true-sim reference at fixed candidate budget.
4. If pursuing robustness claims, freeze NEW independent geometry/speed/near-contact/obstacle or terminal-velocity evaluation before expansion. Do not tune on these six bases or their24 variants. Current screen has no obstacles and no terminal-speed constraint; add either only as an explicit new task/metric change.

See root [COLLABORATION_START_HERE.md](../../../../COLLABORATION_START_HERE.md) for proposed owners, acceptance artifacts, historical migration scope and external asset dependencies. No messages were sent to teammates, and no PR/merge was created automatically.
