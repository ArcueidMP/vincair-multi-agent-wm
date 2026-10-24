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
