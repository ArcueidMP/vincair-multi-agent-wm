# Headless E4 baseline reproduction

## Scope

Reproduce the teammate's native VMAS Navigation baseline at commit
`a4ff32623f4287966aef878d44f01d3521e660af`, without Colab, Google Drive,
notebook installation cells, rendering, or cached notebook outputs.

`reproduce_baselines.py` loads only the original function definitions for the
environment and independent/centralized CEM planners from the checked-in
notebooks. It preserves CEM horizon8, population64, iterations3, elite fraction
0.125, goal/safety/action weights5/100/0.01, component action clipping, and the
original execute-first-action rule. The priority PD logic is regression-tested
against notebook01. No production dependency was added.

## Run from the repository root

Use an existing Python environment with VMAS1.5.2, NumPy and PyTorch. Recorded
local versions are included in the result protocol. No checkpoint is required.

```bash
python -m unittest discover -s tests -v
python experiments/e4_navigation/reproduce_baselines.py --suite smoke --device cuda --output outputs/e4_reproduction_v1
python experiments/e4_navigation/reproduce_baselines.py --suite full --device cuda --output outputs/e4_reproduction_v1
python experiments/e4_navigation/summarize_reproduction.py --run outputs/e4_reproduction_v1
```

`--device cpu` is supported, but PyTorch CPU/CUDA RNGs can produce different
CEM samples; do not require CPU runs to numerically match cached CUDA results.
Use the same output path for smoke and full. The immutable protocol checks
source/notebook hashes, configuration and runtime versions before resuming.
Full suite reuses the two completed S1 centralized episodes, producing21 unique
episodes total. Existing incomplete directories are preserved and rejected.
Each invocation has a1800-second wall budget checked between episodes.

## Metrics and verification

- Original endpoint-level metrics and position-only joint success are preserved.
- Additional diagnostics record every native integration substep, separation,
  safe joint success, terminal speed and full action/state trajectories.
- The instrumentation calls the original integrator without altering forces,
  limits, drag or collisions. Every executed sequence is replayed in a fresh
  environment; every raw trajectory has a SHA256 hash.
- `summarize_reproduction.py` independently checks saved arrays and compares
  the18 centralized episodes to the teammate's cached CSV. Latency is excluded
  from numerical equivalence checks.
- The three geometries are fixed. Repeating seeds varies planner randomness;
  it is not a held-out geometry generalization study.

## Publication boundary

Commit source, tests, this guide and the small `reproduction_results/v1/`
tables, provenance and reports. Keep full trajectories and model weights in
external storage with hashes. Earlier MultiPushT E1 work remains a separate
experiment: its latest confirmed residual improvement was6.7%, insufficient
for the predeclared closed-loop gate. Do not portray it as validated
Navigation performance.

Next: audit conflict labels and scene-level splitting, then integrate safety
ranking. The Navigation-specific learned checkpoint/readout is still missing;
notebook04's learned adapter is a contract, not an implemented model.
