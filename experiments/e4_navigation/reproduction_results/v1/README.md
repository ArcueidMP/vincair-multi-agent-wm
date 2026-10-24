# E4 baseline reproduction v1

Actual local runs based on teammate commit `a4ff32623f4287966aef878d44f01d3521e660af`. This is true-VMAS reproduction, not learned world-model planning.

## Results

| Method | Episodes | Goal success | Safe goal success | Substep loss of separation | Mean latency(ms) |
|---|---:|---:|---:|---:|---:|
| centralized_no_safety | 9 | 9 | 0 | 9 | 119.07 |
| centralized_safety | 9 | 9 | 3 | 6 | 116.14 |
| independent_cem | 1 | 1 | 0 | 1 | 254.37 |
| independent_pd | 1 | 0 | 0 | 1 | 0.17 |
| priority_pd | 1 | 1 | 1 | 0 | 0.14 |

Two PD policies and independent CEM cover S1 seed42 only. Centralized arms each cover three fixed layouts and three planner RNG seeds. These are different evaluation scopes: do not compare the counts as a common generalization benchmark. Safety means separation >=0.30 at every recorded native integration substep, including initial state; goal success preserves the original position-only tolerance0.12. There is no terminal-speed requirement in this reproduction.

Cached upstream comparisons: **18/18** match step count, goal success, step-level violation count and minimum separation within1e-5. Latency is hardware dependent and is not expected to match. Substep metrics are additional diagnostics, not retroactively attributed to the team's old results.

## Validation

All21 action sequences were replayed in fresh native VMAS environments, maximum state error **0.0**. Saved-array checks independently recompute success, separation, bounds, substep endpoints and hashes. Four targeted unit tests compare original notebook functions against this entry point and check scenario-goal isolation and substep safety. Initial development errors in optional VMAS metadata access (`substeps`, absent `close()`, nullable agent drag) were corrected before collecting results; no physics or planner parameters changed.

## Environment and limitations

Versions: `{"vmas": "1.5.2", "torch": "2.11.0+cu128", "numpy": "2.2.6"}`. Local runtime is CUDA on an RTX4060 laptop; planning uses native VMAS with dt0.1, two substeps, agent radius0.1, action component limits±1, world drag0.25, and no added speed/acceleration cap. This differs from earlier local bounded-navigation pilots; their success figures must not be pooled with these.

The learned Navigation adapter in notebook04 remains unimplemented. No Navigation weight was downloaded or trained. Soft safety cost is not a safety guarantee. Existing decision-ranking data lack close conflicts; conflict-pilot labels are the next priority, with raw NPZ provenance and scene-group splits checked first.

## Raw artifacts

All state, velocity, executed action, native-substep state, goals and latency arrays are retained in the local run directory. Large/raw artifacts are excluded from Git; `trajectory_manifest.json` identifies every NPZ by SHA256 and bytes. No model weights, private chat screenshots, or absolute personal filesystem paths are included in this result bundle. Full rerun commands are in the parent `REPRODUCTION.md`.
