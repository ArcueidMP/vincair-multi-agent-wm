# Archived self-built UAV transport baseline — v2

**Status: archived, frozen comparison baseline.** This package preserves the
self-built MuJoCo quadrotor study before work on the official CrazyMARL and
MuJoCo Menagerie Crazyflie 2 models. The original experiment results, source
bytes, and frozen local predictors are retained together.

The snapshot was captured on 2026-10-25 from `codex/novelty` at Git HEAD
`300d7eccd0a65329bbf45c8ce9f4a43b17b21a0b`. It includes the then-current working
tree version of `README.md` and the untracked `uav_transport/viewer.py`; their
exact hashes are in [capture.json](capture.json).

## Contents and integrity

- [repo/uav_transport/](repo/uav_transport/) contains the 14 captured Python modules.
- [repo/tests/](repo/tests/) contains the nine captured UAV test files.
- [repo/requirements-uav.txt](repo/requirements-uav.txt) pins the CPU simulation environment.
- [Archived research report](repo/docs/uav-transport-world-model-audit.md) and
  [archived project README](repo/README.md) preserve the original prose.
- [Compact evidence and six frozen models](repo/experiments/uav_transport/v2/)
  retain the published v2 records byte for byte.
- [manifest.json](manifest.json), its detached checksum, and `verify.py` cover
  the captured files and archive metadata.
- [external_artifacts.json](external_artifacts.json) indexes 1,266 retained
  raw scientific artifacts, with paths, sizes, and verified SHA256 hashes.
  These approximately 5.90 GB of raw records remain in the original ignored
  `data/` tree; the archive contains references rather than another copy.

There are 46 exact captured files totaling 1,954,725 bytes. Checksum-covered
archive files are read-only. Make a separate working copy for future changes
or new experiment outputs. Runtime caches are outside the checksum inventory.
The copied project documents preserve their original links, including links
to broader repository documentation and separately retained raw records.

## Verified reproduction

Verification used Python 3.10.21, MuJoCo 3.3.0, NumPy 2.2.6 and SciPy 1.15.3 in
an isolated environment. Imports resolved to `repo/uav_transport` in this
archive, independently of the active UAV package.

- **83 archived tests passed**, with two optional Torch training tests skipped.
- A complete recorded v2 mission replay reproduced all physical states,
  integration states, and scoring labels **bit for bit**.
- A fresh frozen-small-model mission also reproduced the historical complete
  physical/integration trajectories, joint rotor commands, external forces,
  and published task scores **bit for bit**.
- Every copied file and every referenced external raw artifact passed its
  SHA256 check.

The representative case is `wm_small`, seed 0,
`mass026_donor0_route0`. Detailed verification outputs are in
[verification_mission.json](verification_mission.json),
[verification_replay.json](verification_replay.json), and
[verification_tests.log](verification_tests.log).

## Run without the original raw data

The compact models and archived source are sufficient to execute a fresh
representative mission. Starting at the containing repository root:

```bash
UAV_ARCHIVE="$(pwd)/archives/uav_transport/self_built_v2"
UAV_RUN="$(mktemp -d)"
uv venv --python 3.10 "$UAV_RUN/.venv"
uv pip install --python "$UAV_RUN/.venv/bin/python" \
  -r "$UAV_ARCHIVE/repo/requirements-uav.txt"
python "$UAV_ARCHIVE/verify.py"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$UAV_RUN/.venv/bin/python" "$UAV_ARCHIVE/smoke.py" \
  --mode mission --out "$UAV_RUN/mission.json"
```

`smoke.py` explicitly inserts the archived package path before import. To call
other archived modules directly, use `PYTHONPATH="$UAV_ARCHIVE/repo"` and run
from `"$UAV_ARCHIVE/repo"` or a separate copy of that directory. For a full
matrix, first copy `repo/` into a writable working directory, then restore the
model cache layout there as described in the archived evidence README. Use a
fresh output prefix for every experiment.

Run the archived tests independently of the active repository's pytest setup:

```bash
PYTHONPATH="$UAV_ARCHIVE/repo" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$UAV_RUN/.venv/bin/python" -m pytest -q -c /dev/null \
  --rootdir "$UAV_ARCHIVE/repo" --confcutdir "$UAV_ARCHIVE/repo" \
  "$UAV_ARCHIVE/repo/tests" --basetemp "$UAV_RUN/pytest"
```

## Check or replay the separately retained raw records

Set `UAV_ORIGINAL_ROOT` to the original repository or a relocated checkout
containing the same `data/uav_transport/` files:

```bash
UAV_ORIGINAL_ROOT="/path/to/repository-with-original-data"
python "$UAV_ARCHIVE/verify.py" --external-root "$UAV_ORIGINAL_ROOT"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$UAV_RUN/.venv/bin/python" "$UAV_ARCHIVE/smoke.py" \
  --mode replay --raw-root "$UAV_ORIGINAL_ROOT" --out "$UAV_RUN/replay.json"
```

The optional `--raw-root` argument also lets fresh mission execution compare
its entire trajectory to the historical raw mission. The archived GUI viewer
accepts an explicit raw pickle path:

```bash
PYTHONPATH="$UAV_ARCHIVE/repo" "$UAV_RUN/.venv/bin/python" \
  -m uav_transport.viewer \
  "$UAV_ORIGINAL_ROOT/data/uav_transport/mission_v2_wm_small_seed0_mass026_donor0_route0.pkl"
```

The archive does not modify active source files or original scientific data.
Its historical scores remain attached to the self-built v2 plant family.
