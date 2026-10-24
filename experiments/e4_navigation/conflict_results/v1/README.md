# Conflict ranking dataset and oracle audit v1

Reconstructed from teammate notebook04 at commit a4ff326. No original private NPZ was downloaded: these are newly generated, provenance-recorded artifacts. The12 public pilot states match cached unsafe candidate counts, minimum step separation and both best indices. Total5376 candidates: public pilot768 and new geometry-grouped data4608.

| Split | Geometries | States | Mixed states | No safe candidate | Soft unsafe selection* | Safety-first oracle* |
|---|---:|---:|---:|---:|---:|---:|
| public_pilot | 3 | 12 | 12 | 0 | 83.33% | 0.00% |
| train | 12 | 48 | 47 | 1 | 75.00% | 0.00% |
| validation | 3 | 12 | 12 | 0 | 91.67% | 0.00% |

*Conditional on at least one safe candidate; first average within geometry over eligible offsets, then equally over eligible geometries. Zero safety-first oracle error follows from the priority definition; it is not learned-model performance. No test method metrics are computed. Train has48 groups/3072 candidates, validation12/768, sealed test12/768. Geometry rotations/translations/scales differ across splits; all four offsets and64 candidates stay within one geometry group. Route families remain shared: this is not held-out-family generalization.

## Targets

Preserve original goal/safety/action weighted soft cost5/100/.01 using environment-step states. New safety labels include the initial state and all native integration substeps at separation threshold0.30. Safe candidates precede unsafe candidates; safe candidates sort by soft cost. When all are unsafe, maximize minimum separation, then minimize squared violation exposure, soft cost, and stable index. No feasible action is invented. Never describe an all-unsafe fallback as safe.

Training/validation preference kinds:0 safe-vs-unsafe,1 safe cost,2 unsafe robust fallback. Exactly tied candidates do not produce preference pairs. Pairs:train96737, validation24186; group-correlated, not independent sample counts. No test preferences were generated.

## Validation and storage

All5376 candidates were rolled out twice in fresh native VMAS environments, state/substep replay error0. Independent NumPy verification checks separated inputs, bounds, native substep endpoints, cost/separation labels, tuple-based ordering, every preference direction and geometry splits. Maximum soft-cost formula difference1.91e-06; tolerance1e-5 absolute/relative accounts for GPU/CPU floating reductions. Original data were not altered. Nine targeted unit tests (including the prior4 reproduction tests) pass.

The labels are true-simulator/oracle supervision. No Navigation weights were trained or evaluated; no closed-loop claim. Finite H8 safe candidates do not guarantee safety beyond the window. Raw data remain external with SHA256 manifest; committed files are source, contracts and audit tables. `inputs.npz` contains only initial state/goals/actions/mode, and `labels.npz` contains future state/substep trajectories and supervision. Never use labels as online model features.

## Reproduce

```bash
python -m unittest discover -s tests -v
python experiments/e4_navigation/conflict_ranking.py --output outputs/conflict_ranking_v1
python experiments/e4_navigation/verify_conflict_ranking.py --run outputs/conflict_ranking_v1
```

Current entry point uses CUDA with VMAS1.5.2 to match the public pilot. The protocol is saved before collection, all original source functions come from the checked-in notebook, and existing outputs are preserved/rejected. The test split is generated and integrity-checked only, not used for target tuning, model selection or method metrics. A Navigation checkpoint, compatible state readout/normalization and actual learned adapter remain the next material dependency.
