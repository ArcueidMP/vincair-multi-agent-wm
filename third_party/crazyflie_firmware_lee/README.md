# Pinned upstream Lee position/attitude controller

These seven source/license files are copied byte for byte from
[IMRCLab/crazyflie-firmware](https://github.com/IMRCLab/crazyflie-firmware/tree/b08d9f3cbcc581d8e7f988cabad22c26d35f4a58),
commit `b08d9f3cbcc581d8e7f988cabad22c26d35f4a58`, the firmware revision pinned by
[coltrans-planning](https://github.com/IMRCLab/coltrans-planning/tree/146b205e33e20ac26975993d32ec89b140754550).
[UPSTREAM.json](UPSTREAM.json) records each original path, Git blob identifier,
SHA256, and observed file-level license notice.

`controller_lee.c` implements the geometric position and attitude controller
based on Lee, Leok and McClamroch, CDC 2010. It is compiled directly by
`uav_transport.external_controller`; the control equations are not rewritten
in Python. Each UAV has its own complete `controllerLee_t` state, including
the upstream position and attitude integrals.

The integration uses the study's existing payload-force allocator, formation
references, recovery protocol and bounded rotor mixer. It is labeled
**shared payload/formation allocation plus upstream Lee position/attitude
feedback**. The separate upstream `controller_lee_payload.c`, with its generated
OSQP workspaces, is not used in this adapter.

The wrapper supplies the calibrated airframe mass/inertia and configures the
source scheduler for the experiment's 50 Hz control interval; upstream's
default attitude rate is 500 Hz. All published position and attitude gains are
retained. Native libraries are built with a local C compiler and cached under
the ignored `data/uav_transport/external_controller_build/` directory, with
source, compiler and binary hashes.

## License notices

The controller C file and `math3d.h` carry MIT notices. Firmware interface
headers carry GPL version 3 notices; the original [LICENSE.txt](LICENSE.txt)
is retained. All copyright and license text remains intact. This bundle is
not distributed as MIT-only code.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m pytest -q tests/test_uav_external_controller.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 data/uav_transport/.venv/bin/python \
  -m uav_transport.external_experiment \
  --out data/uav_transport/menagerie_external_lee_v1 --workers 4
```

Use a fresh output prefix to preserve existing runs. The experiment exposes
only `rule_fixed` and `rule_reconfigure`, with the same complete mission and
physical limits used by the other Menagerie comparisons.
