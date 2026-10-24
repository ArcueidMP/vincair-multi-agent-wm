# Frozen Navigation pressure screen: high-inertia failures

## Result

Predeclared stress screen **FAILED**. The analytic-goal/frozen-risk hybrid is safe6/6 in the near-conflict condition but safe0/6 in high inertia, despite reaching goals6/6. True-sim safety reference is safe6/6 in both. The previous four-condition24/24 success cannot be generalized to this harder state distribution.

| Condition | Arm | Joint arrival | Safe joint arrival | Any separation violation |
|---|---|---:|---:|---:|
| near_conflict | invariant_repair | 0.0% | 0.0% | 0.0% |
| near_conflict | analytic_goal | 100.0% | 100.0% | 0.0% |
| near_conflict | true_sim_safety | 100.0% | 100.0% | 0.0% |
| high_inertia | invariant_repair | 0.0% | 0.0% | 100.0% |
| high_inertia | analytic_goal | 100.0% | 0.0% | 100.0% |
| high_inertia | true_sim_safety | 100.0% | 100.0% | 0.0% |

All collision proxies remain0. A separation violation is distance<0.30 at initial/native-substep states; collision proxy is center distance<0.20. Neither zero collisions nor successful arrival cancels a separation violation. Position-only joint goal tolerance0.12, no terminal-speed constraint.

## Protocol

6 NEW independent base geometries seed2026105201, angles75..135deg/full orientations, two correlated variants each, one candidate seed per base:near conflict initial separation0.42/speeds0.10,0.20; high inertia separation0.36/speeds0.35,0.50.12 variants x3 arms=36 episodes, max200 steps,H8/72. Native VMAS1.5.2 dt0.1/two substeps/radius0.1/drag0.25 unchanged. Both variants initially satisfy separation>=0.30. Conditions jointly change speed and separation, so this is not an isolated velocity causal estimate. Only6 independent clusters, not12.

No fitting/threshold/physics/proposal changes or old consumed confirmation data used. Only published portable direct scorer JSON was loaded; no author or Navigation WM checkpoint. Same initial candidate pools across all3 arms independently verified. All six bases and their variants are now consumed evaluation data and must not tune a later model.

## Post-hoc initial-pool diagnostics

Each high-inertia initial pool has1..4 truly safe H8 candidates, while the analytic policy initially selects a predicted-accepted but truly unsafe H8 sequence. Predicted risk remains below the frozen0.5 threshold and predicted minimum separation exceeds0.30. Actual episode violations begin at zero-based execution step1 or2. These results expose unsafe acceptance/optimistic separation under high inertia. The local H8 label alone does not prove an executed first action must violate; full trajectories supply the actual outcome. Finite candidate coverage is not proof of global physical feasibility or optimality.

| High-inertia geometry | True-safe options initially | Chosen true H8 minsep | Predicted risk | Predicted minsep | First violation step (0-based) |
|---|---:|---:|---:|---:|---:|
| pressure_00_high_inertia | 4 | 0.2945 | 0.0000 | 0.4411 | 2 |
| pressure_01_high_inertia | 4 | 0.2852 | 0.0052 | 0.4264 | 1 |
| pressure_02_high_inertia | 2 | 0.2829 | 0.3911 | 0.4518 | 1 |
| pressure_03_high_inertia | 4 | 0.2709 | 0.3572 | 0.4243 | 1 |
| pressure_04_high_inertia | 3 | 0.2987 | 0.0135 | 0.4311 | 2 |
| pressure_05_high_inertia | 1 | 0.2666 | 0.4988 | 0.4865 | 1 |

Do not remove the safety filter, tune a threshold on these six cases, or report prior successes without this negative screen. New risk calibration/training needs separate NEW development geometry and then a fresh frozen confirmation set.

## Verification

51 targeted tests pass after upstream readout synchronization. Independent NumPy audits3087 model choices, fresh native rollout audits543 oracle choices, and all3630 executed first actions/physical outcomes/condition aggregates. Max NumPy forward error4.58e-05, oracle-cost formula error1.91e-06. State and native-substep action replay error0; paired initial pools exact; portable model hash unchanged. Original run wall90.9s excluding independent audit.

Raw arrays remain external; per-episode tables, protocol, diagnostics, hashes and verification are published. No learning-world-model, three-agent or universal-safety claim.
