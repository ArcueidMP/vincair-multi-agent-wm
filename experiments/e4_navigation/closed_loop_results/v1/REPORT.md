# E4 frozen closed-loop pilot

## Result

Predeclared descriptive gate: **not passed**. Offline safety-ranking gains must not be presented as closed-loop navigation success. No retraining, threshold changes, or online true-label access for learned arms.

| Arm | Joint goal success | Safe joint success | Any separation violation | Collision proxy | Mean planning ms |
|---|---:|---:|---:|---:|---:|
| pd | 100.0% | 0.0% | 100.0% | 100.0% | 0.02 |
| original | 0.0% | 0.0% | 8.3% | 0.0% | 3.33 |
| raw_repair | 0.0% | 0.0% | 16.7% | 0.0% | 3.26 |
| invariant_repair | 0.0% | 0.0% | 0.0% | 0.0% | 3.46 |
| true_sim_safety | 100.0% | 100.0% | 0.0% | 0.0% | 55.24 |

All rates average the two seeds within each geometry then the six geometries equally. Each arm has12 episodes, but only6 independent geometries. Collision proxy means center distance<0.20, not physical contact event instrumentation. Any separation violation uses initial and native substeps and threshold0.30. Position-only joint success tolerance0.12; terminal speed is reported separately, not a success requirement. Latencies are measurements on this laptop and unequal information/computation budgets; PD does not generate candidates, oracle uses true rollout.

## Frozen design

New seed2026104801 generates6 crossing geometries, starts0.75 from center and initial route speeds0.12/0.24. Two candidate RNG schedules per geometry. Five methods x6 geometries x2 seeds =60 episodes, max200 steps each. H8/72 candidates: original64 route/noise,4 zero/brake,4 repeated-current-PD/priority/half-PD actions. This expands the earlier offline68 pool. Same pool-generation rule and random seeds; actual pools diverge as policy states diverge. All learned models/policies frozen from prior blocks. Direct-scoring arms access current observed state/goals/proposed actions only, without simulator rollouts or future state features. True-sim safety arm is a privileged finite-pool H8 reference, not a globally optimal or certified safe controller. Replan each environment step and execute only the first action.

Native VMAS1.5.2:dt0.1,substeps2,radius0.1,world drag0.25,component action limit1. No extra speed limit, dynamics intervention or automatic stop after unsafe events. Every executed trajectory and native substep was replayed in a fresh environment with zero error. Five separate5-step smoke runs passed before the full suite.

## Post-hoc behavior diagnostics

These descriptive summaries were added after seeing pilot outcomes; they are not tuning criteria or evidence of a uniquely identified cause.

| Arm | Mean final per-agent goal distance | Last20-state mean speed |
|---|---:|---:|
| pd | 0.1044 | 0.0816 |
| original | 4.0615 | 0.2106 |
| raw_repair | 2.7592 | 0.1564 |
| invariant_repair | 0.8059 | 0.0676 |
| true_sim_safety | 0.0618 | 0.2471 |

The full trajectory panels show all six geometries for the first predeclared seed. Both seeds and all episode outcomes remain in episodes.csv; no favorable-case selection. Terminal dots are not necessarily goal arrivals. Failed navigation may reflect goal-progress ranking, proposal coverage, and distribution shift under repeated replanning; this block does not isolate their causal contributions.

## Verification and provenance

32 targeted tests pass. Independent NumPy reconstructs7200 learned choices and corresponding network predictions. Fresh native rollouts independently verify664 oracle rankings and cost/separation labels. Every executed first action, goal/safety/collision metric, model checksum and equal-geometry aggregate was checked. Max NumPy forward error3.81e-05; max independently computed oracle-cost error1.91e-06. Native state/substep replay error0. Total experiment wall time146.7s, including smoke and action replay, excluding independent audit.

Full model weights remain in prior external run directories. Raw closed-loop traces/candidates/predictions remain in this external run directory and are identified by SHA256; source, compact results, all episode CSV, frozen protocol, verification and vector figure are published. Raw SVG retained; published SVG strips trailing whitespace only. No renderer or video was claimed.

## Next block

If learned safe-goal success remains poor, first diagnose goal-progress/cost ranking on NEW training/validation trajectories and check candidate coverage using the existing privileged reference. Do not tune on these six consumed confirmation geometries, and do not expand long-duration runs before fixing the observed failure. Separate scoring and candidate-generation ablations in the next frozen protocol. Navigation-specific LeWM checkpoint/config/normalization/history/action/state-readout remains missing; this experiment is a direct-scoring baseline, not LeWM transfer. No three-agent or learned-world-model evaluation was run.
