# Cooperative aerial transport — completed mission_v2 evidence

[Research report](../../../docs/uav-transport-world-model-audit.md) ·
[Branch overview](../../../README.md)

This package organizes the existing physical audit, local-model study and
final 192-mission evaluation. It retains the original result values and model
files. `mission_v2` is the current protocol, with the 20 mm / 0.040 m/s internal
arrival guard and the original external task requirements.

| File | Contents |
|---|---|
| [summary.json](summary.json) | Named field excerpts from the physical, data, local-selection, mission and numerical reports |
| [mission_v2.json](mission_v2.json) | Byte-identical copy of the final protocol and all 192 mission result records |
| [uav_v2_provenance.json](uav_v2_provenance.json) | Original source/artifact verification, including archived source resolutions |
| [mission_v2_boundary_dt0005.log](mission_v2_boundary_dt0005.log) | Recorded targeted 0.5 ms check of the tension-boundary case |
| [models/](models/) | All six frozen NumPy checkpoints, metadata, manifest and training registration |
| [index.json](index.json) | Original file hashes, selected summary fields, published file hashes and combined-branch regression results |

## Main outcome

All missions deliver their payload. Completion with the 5 cm excursion and
0.6 N donor-tension limits changes from 0/36 to **35/36** for the small model,
and from 0/36 to **36/36** for the large model. Exact-option prediction and the
complete geometric controller each achieve **12/12** with reconfiguration.
All 192 whole-mission refinements pass the trajectory-error checks; the
fine-resolution small-model constrained score is 36/36. The nominal result is
retained in the main comparison.

The `summary.json` section names are the original filenames. Each section
records its source SHA256 and the exact fields copied. Complete raw candidate
arrays, trajectories and the earlier `mission_v1` archives remain under
`data/uav_transport/`, as recorded in the index and source manifests.

## Use the existing frozen models

From the repository root, restore the original cache layout in a fresh checkout:

```bash
mkdir -p data/uav_transport
cp -n experiments/uav_transport/v2/models/* data/uav_transport/
```

The six weight and metadata pairs match the original manifest hashes. With the
UAV environment described in the research report, run a fresh mission output:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.mission --models data/uav_transport/models_v1_manifest.json \
  --out data/uav_transport/mission_reproduction --workers 8
```

Runtime prediction uses NumPy. Full data-collection, fitting and detailed
training-data audits use the commands and raw inputs listed in the research
report. Published mission summaries and historical validation counts remain
associated with their original runs; the index separately records the
regression checks performed when consolidating this branch.
