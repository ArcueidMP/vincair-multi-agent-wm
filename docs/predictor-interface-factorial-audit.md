# Contact-Mode Coverage Controls Planning Capability Across Predictors

2026-10-25, `codex/novelty`.

**Executable contact coverage is a causal determinant of composed manipulation
capability. Its bottleneck persists with an elevenfold increase in network
capacity and with exact simulator option prediction. Expanding contact modes
recovers performance even when the candidate budget is held fixed.**

The completed predictor × interface experiment provides three mutually
supporting results: large paired gains with frozen weights, the same effect
under exact dynamics, and persistent gains with 39 candidates on both sides.
It extends the [completed action-interface audit](completed-action-interface-audit.md)
with 1,260 complete task executions.

## Complete predictor × interface matrix

Ordered-goal scores from the nominal 160-substep evaluation:

| Predictor | Direct39 | Expanded117 | Shared successful first stages |
|---|---:|---:|---:|
| Matched velocity model, 1,949 parameters per support network | 6/108 | **64/108** | 74/108 |
| Larger velocity model, 22,149 parameters per support network | 1/108 | **64/108** | 66/108 |
| True-option simulator prediction | 1/36 | **35/36** | 35/36 |

Each learned row pools three frozen model seeds over the same 36 conditions.
Seed-level direct → expanded successes are `1→24, 1→15, 4→25` for the matched model
and `0→23, 1→25, 0→16` for the larger model, each out of 36. The direct interface
remains a severe bottleneck for all three predictors. Adding executable contact
routes recovers 58, 63, and 34 complete tasks respectively.

Conditional on the common successful first stage, the corresponding direct to
expanded counts are **6/74→64/74**, **1/66→64/66**, and **1/35→35/35**. These conditional
counts are supplementary; the main score above retains every failed prefix.

**The practical consequence is that contact-mode coverage deserves to be
optimized alongside prediction quality.** Across the tested predictors,
expanding this physical action interface is the decisive intervention for
recovering composed-task capability.

## Frozen protocol

All models remain frozen throughout evaluation. Each support-specific matched
velocity network has 1,949 parameters; its larger counterpart has 22,149. A
policy family contains three support-specific networks per fit seed. Both use the same previously
trained local data and frozen checkpoint pack:
`447460169e669319b7874de6d3f1c91dfbeae95479af198a0e23dc37e8a135a4`.
The true-option predictor uses complete simulator snapshots and evaluates one
complete option with the same endpoint cost. Every planner uses one-option
lookahead over the declared finite action bank.

All tasks start from the initial scene and execute rotation, retreat, rebracing,
and final rotation. Physics, force limits, helper controllers, prelude, stage
budgets, and termination rules are unchanged: dry_light at 160 substeps, unit
force bound, 10 prelude steps, 60/20/80/60 task steps, 0.015-rad angular tolerance,
0.05 translation tolerance, and five consecutive qualifying states per goal.
Travel, contact acquisition, and pushing all consume the actual stage budget.
The helper, routing, and recovery controllers are fixed hand-written policies.
The task grammar permits one contact-side change, with passive, world-anchor,
and preloaded-anchor support conditions.

The 36 physical conditions combine three supports, two initial orientations,
three velocity perturbations, and two goal pairs. Each learned predictor uses
all three previously frozen fit seeds. Simulator controls run once per physical
condition. Six unrestricted cells plus nine allocated-budget arms produce
**1,260 complete task executions**. Seeds and allocations are repeated paired
controls on these same physical conditions.

## Candidate budget control

Direct control offers 39 programs: zero drive or one of 12 direction/force
combinations, each at 5, 10, or 20 pushing steps. Unrestricted expansion attaches
each program to each of three routes, yielding up to 117 choices.

The budget39 interface keeps the exact original candidate-parameter multiset
of 39 direction, force, and duration programs. Executed actions and effort are
recorded separately. All three zero-drive programs stay direct. For
nonzero force index `f=0..11`, duration index `d=0..2`, and prespecified shift
`s=0..2`, route assignment is `(d + f + s) % 3`. Each table therefore contains
15 direct, 12 upper-route, and 12 lower-route candidates. Across the three
allocations, every nonzero program visits every route. All options were already
present in the frozen local training bank.

All three prespecified allocations are reported.
The first rotation always has the original direct39 legal action bank. After
acquiring a reverse face, or when a route is immediately unavailable, budget39
uses the full direct39 bank on the current face. This fallback depends only on
current geometry and controller memory.

The matched budget is **39 distinct candidate slots per decision**. Learned
predictors evaluate all slots; simulator predictors roll out only immediately
legal candidates whose declared pushing duration fits the remaining budget.
Actual route durations are checked after prediction. Decision counts, total
candidate evaluations, active rollout steps, padded simulator integration work,
and wall time are recorded separately. The experimental budget match applies
per decision; episode-level resource totals are empirical outcomes.

The unrestricted control preserves the historical shared-batch behavior: when
another row has entered the final stage, a first-stage row can receive 117 model
predictions with routed candidates masked. Its legal choices remain direct39.
The budget39 arm always allocates 39 slots.

### All prespecified allocations

| Predictor | Direct39 | Expanded39, shift 0 | Expanded39, shift 1 | Expanded39, shift 2 | Expanded117 |
|---|---:|---:|---:|---:|---:|
| Matched velocity | 6/108 | **64/108** | **62/108** | **59/108** | 64/108 |
| Larger velocity | 1/108 | **56/108** | **57/108** | **65/108** | 64/108 |
| True-option simulator | 1/36 | **35/36** | **35/36** | **35/36** | 35/36 |

The interface advantage survives every fixed allocation. In particular, exact
simulator prediction reaches the same 35/36 ordered successes with 39 candidate
slots as with 117. **At a fixed search budget, changing the available contact
modes recovers 34 additional complete tasks.** This isolates contact coverage
as the effective intervention in the matched-budget comparison.

Allocation and fit-seed sensitivity remain visible. Per-seed counts for shifts
0/1/2 are:

| Predictor | Shift 0: seeds 0/1/2 | Shift 1: seeds 0/1/2 | Shift 2: seeds 0/1/2 |
|---|---|---|---|
| Matched velocity | 28 / 11 / 25 | 21 / 16 / 25 | 23 / 12 / 24 |
| Larger velocity | 20 / 23 / 13 | 16 / 25 / 16 | 23 / 25 / 17 |

Each count in this table has denominator 36. All allocations retain large gains
over direct control. The seed-level results characterize how each frozen
predictor uses the different candidate allocations.

## Paired attribution and resource accounting

The audit makes 1,008 within-predictor comparisons against the corresponding
direct baseline, including all allocation repeats. Every pair agrees on first
goal success and all pre-final phase endpoint clocks. The maximum pre-final
discrepancy is **2.403e−7** across the complete 24D physical state and **5.96e−8**
in joint commands, below the declared absolute tolerance of 1e−6. Complete
initial and handoff snapshots, action buffers, real prelude, and input history
also pass the comparison.

For unrestricted expansion, there are **58 / 63 / 34 paired improvements** for
the matched / large / simulator predictors, with **zero paired regressions**.
Every fixed-budget allocation also retains every direct success across this
matrix.

The 360 executions whose exact arms existed in the preceding audit reproduce
their entire physical trajectories and joint actions **bit for bit**, with all
scoring labels unchanged. All 1,260 nominal executions remain finite, avoid
walls, and have no failed-option events.

Navigation has a substantial physical cost. Means below include the real
prelude and every executed navigation/recovery step, exclude padding, and use
the measured substep actuator force–displacement work in simulator units.

| Predictor | Mean steps, direct → expanded | Primary absolute actuator work, direct → expanded |
|---|---|---|
| Matched velocity | 135.16 → 119.15 | 0.04377 → 0.44304 |
| Larger velocity | 140.70 → 121.20 | 0.04142 → 0.47366 |
| True-option simulator | 120.89 → 97.64 | 0.03364 → 0.43324 |

Side changes use more actuator work and finish tasks in fewer steps. All arms
share the instantaneous force bound and physical step budget; force integrals
and actuator work quantify the execution tradeoff.

The budget39 simulator arms also use fewer evaluated options over their actual
episodes: **4,745–4,901**, versus **13,832** for direct39 over the same 36 cases.
Their padded integration work is **139,503–142,623** candidate-environment
steps versus **270,855** for direct39. Thus the observed gains also come with
fewer total oracle evaluations and less integration work. Active rollout steps, padded integration steps,
substeps, and selector wall time are retained separately in the audit.

Sources: [complete matrix](../data/novelty/interface_factorial_v1.json),
[pairing, candidate-budget, and compute audit](../data/novelty/interface_factorial_v1_analysis.json),
and [prespecified local registration](../data/novelty/interface_factorial_v1_registration.json).

## Numerical validation

All **1,260/1,260** trajectories pass the prespecified active-prefix refinement
thresholds of 0.001 position units and 0.003 rad. Maximum discrepancies between
160 and 320 substeps are **0.00064140** and **0.00265336 rad**. Nominal replays
reproduce every full 24D state and scoring label exactly. Both resolutions remain
finite and clear of walls; maximum weld residuals are approximately 0.006293
position units and 0.063043 rad.

The refinement check starts from identical complete post-prelude snapshots,
holds the recorded joint actions fixed, and evaluates the original five-state
dwell windows and terminal times. The resulting 320-substep scores are:

| Predictor | Direct39 | Expanded117 | Expanded39, shifts 0 / 1 / 2 |
|---|---:|---:|---|
| Matched velocity | 6/108 | **64/108** | **63 / 62 / 60**, each out of 108 |
| Larger velocity | 1/108 | **64/108** | **57 / 57 / 62**, each out of 108 |
| True-option simulator | 1/36 | **34/36** | **34 / 34 / 34**, each out of 36 |

**Every interface comparison retains its advantage under numerical refinement,
and all 1,008 fine-resolution pairs retain zero regressions against direct
control.** Eight ordered labels change from success to failure and two change
from failure to success at the original dwell boundaries. The exact-simulator
arms share the same first-stage boundary case responsible for 35→34. All 37
changes across first/final/terminal/ordered metrics, spanning 20 executions,
are retained in the audit.

Reports: [complete numerical replay](../data/novelty/interface_factorial_v1_numerics.json)
and [per-arm scores and paired refinement summary](../data/novelty/interface_factorial_v1_numerics_summary.json).

## Research claim

**The available physical interactions form an independent axis of world-model
planning capability. A stable bottleneck in contact-mode coverage survives
both increased neural capacity and exact dynamics; supplying the missing
contact transitions restores most tasks with a successful first stage under the same candidate
budget.**

The interface and locally trained models remain frozen across both goal pairs,
three support conditions, and all model seeds. The true-option expanded control
achieves 12/12 with passive support; the matched and larger models achieve
23/36 and 25/36 there. Contact coverage is therefore a capability mechanism
across support regimes. For the physical-abstraction research program, this
provides a concrete mechanism to build on: physically organizing contact
changes which task corrections a fixed predictive system can realize.

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.interface_factorial \
  --output data/novelty/interface_factorial_v1 --workers 8
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.interface_factorial_analysis \
  --manifest data/novelty/interface_factorial_v1.json \
  --previous data/novelty/interface_complete_v1.json \
  --out data/novelty/interface_factorial_v1_analysis
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m vmas_pusht.interface_numerics \
  --manifest data/novelty/interface_factorial_v1.json --fine-substeps 320 \
  --workers 8 --out data/novelty/interface_factorial_v1_numerics
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

Use fresh output prefixes to preserve completed artifacts. The frozen local
model pack and earlier audit manifests must already exist. The previous source
files were copied, with matching SHA256 hashes, to the ignored local directory
`data/novelty/interface_factorial_v1_previous_sources/` before extending the task
runner. Historical source hashes continue to identify those archived versions.
Raw data, model weights, and Chinese summaries remain local; this report and
the implementation are in English.

**235 tests pass.** Experiment sources and all model packs retain their
registered hashes; the paired audit verifies 235 input files. The new selector
also reproduces the original planner on matched test queries, and the historical
full-task replication checks its behavior across the complete original cells.
