# E4 direct-scorer repair: fresh confirmation

## Result

The predeclared offline gate passed: original64 conditional unsafe30.56% -> invariant repair64 6.94% (23.61 percentage points), with9/12 geometry wins. This is a direct observed-state/action scorer, NOT a Navigation LeWM or a closed-loop improvement.

| Method | Unsafe given safe | Soft-cost regret |
|---|---:|---:|
| old_safety_64 | 30.56% | 1.1620 |
| old_safety_68 | 18.06% | 1.5361 |
| raw_consistent_64 | 6.94% | 0.6853 |
| raw_consistent_68 | 6.94% | 0.6944 |
| raw_legacy_selector_64 | 12.50% | 0.6145 |
| invariant_consistent_64 | 6.94% | 0.3097 |
| invariant_consistent_68 | 6.94% | 0.3126 |
| invariant_legacy_selector_64 | 6.94% | 0.2815 |

## What the ablations support

- Fresh raw-coordinate training plus the consistency selector also reaches6.94% unsafe. Therefore safety improvement over the old model cannot be attributed specifically to invariance; broader training, normalization and changed selection are confounded in that comparison.
- Invariant repair improves mean regret0.6853 ->0.3097 relative to the freshly trained raw control, with equal unsafe rate. Each representation has its own validation-selected checkpoint/policy; this is a method comparison, not an isolated causal estimate.
- The invariant model with the legacy selector has the same6.94% unsafe and slightly lower regret0.2815. The consistency selector shows no safety benefit for this model on this confirmation set. Preserve this negative ablation; do not retrospectively switch the preregistered primary result.
- Appending four zero/brake candidates helps the frozen old model30.56% ->18.06%, but does not improve the repaired model6.94% ->6.94%. Braking is a fixed open-loop sequence and is not certified safe. Regret comparisons between64/68 use different pool optima.

## Data and selection

44 new geometries:24 train,8 validation,12 confirmation; three initial-state conditions and two distances per geometry.264 states x68 candidates =17,952 native VMAS rollouts, each repeated in a fresh environment. Labels include initial and native substep separation. Whole geometries remain in one split. No previous consumed test/stress files were read for training or selection. Confirmation labels were generated and hashed before training but method access was guarded until model/policy freeze.

Both models use5059-parameter44->64->32->3 MLPs. Invariant features center positions, align inter-agent displacement and average both agent-order predictions. Train-only feature/cost moments. Seed0 CPU training, validation-only epoch/risk/margin selection; raw epoch56, invariant epoch63. Original checkpoint hash unchanged.

Raw and invariant models both train on the expanded68 proposal pool. The64 versus68 comparisons change evaluation candidates only; they do not isolate the effect of expanded-pool training. The three synthetic conditions share geometry families across splits. One seed and12 confirmation geometries do not establish broad generalization. Arrival-offset also changes separation. This confirmation set is now consumed and must not tune later models.

## Verification and artifacts

27 targeted tests pass. Independent NumPy verification reproduced576 method decisions and every aggregate; rebuilt training moments and all17,952 physical cost/safety labels. Maximum cost error1.91e-06; forward error2.24e-05; native replay error0. Numerical translation prediction error2.1e-05; rigid-coordinate and agent-swap feature invariance unit tests pass. Coordinate invariance is a property of the scoring representation, not a claim that clipped VMAS action dynamics are globally rotation invariant.

Full weights/raw NPZ remain in this external run directory. Published files contain source, frozen protocol, hashes, compact summaries, all confirmation decisions, verification and figure. Raw SVG exports are retained; published SVG only strips trailing whitespace.

## Next block

Freeze a small NEW closed-loop confirmation protocol using native VMAS, matched seeds/budgets and original/repair scoring arms. First validate a real receding-horizon integration and state/action scaling with a short smoke test, then run a bounded paired pilot. Do not extrapolate H8 offline gains to successful navigation. Navigation-specific WM checkpoint/loader/config/normalization/history/action contract/state readout remain missing; request these from teammates before any LeWM transfer claim. No three-agent or Navigation-WM experiment was run here.
