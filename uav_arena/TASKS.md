# UAV arena task board

Who builds which row of the benchmark, what each row delivers and when it is done. The rules every
row follows are in [PARTICIPANTS.md](PARTICIPANTS.md); this board assigns work and does not change
them.

| Row | Owner | Status |
|---|---|---|
| Entity-token JEPA `full`, `no_cross`, `flat` | Yihao Xu (@Yihao-Xu-777) | not started |
| Autoregressive state model (W-ar) | june (@june250630) | not started; builds on the JEPA backbone |
| PETS ensemble | june (@june250630) | not started |
| TD-MPC2, DreamerV3 | unassigned | blocked on the Gymnasium wrapper |
| Evaluation harness and recorder ([HARNESS.md](HARNESS.md)), analysis plan ([ANALYSIS.md](ANALYSIS.md)), Gymnasium wrapper, reference and CrazyMARL rows on `test` | maintainer (@ArcueidMP) | analysis plan frozen in `arena-v0.3`; recorder and harness built and validated on `dev`; the rest open |

## First day, every owner

1. Branch from the integration branch: `git fetch origin && git switch -c <handle>/<row> origin/uav-arena`.
   Its frozen files are those of the tag `arena-v0.3`. Since `arena-v0.2`, only `collect.py` has
   changed: the standard data are now `explore-v2`.
2. Build the environment ([README](README.md#environment)): Linux or WSL2, a C compiler, Python 3.10.
3. Collect and verify the data, about a minute on 24 cores. On Ubuntu 22.04 the regenerated
   episodes differ from the references, so copy the data there instead:

   ```bash
   PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.collect --seeds 200000:202000 \
     --out data/uav_arena/datasets/explore_v2
   PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.collect --seeds 299000:299100 \
     --out data/uav_arena/datasets/explore_v2_val
   for d in explore_v2 explore_v2_val; do
     PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.collect --verify data/uav_arena/datasets/$d
   done
   ```

4. Run the arena tests ([README](README.md#reproduce)); all must pass.
5. Run the example participant ([`participants/example/`](participants/example/)) end to end and
   compare with [its numbers](#example-participant). This is the path your model will take:

   ```bash
   PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.example.train \
     --data data/uav_arena/datasets/explore_v2 --out data/uav_arena/participants/example/mlp.pt
   PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop --predictor uav_arena.participants.example.model:make \
     --predictor-options '{"checkpoint": "data/uav_arena/participants/example/mlp.pt"}' \
     --data data/uav_arena/datasets/explore_v2_val --out data/uav_arena/results/example_mlp_openloop_val.json
   PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run --predictor uav_arena.participants.example.model:make \
     --predictor-options '{"checkpoint": "data/uav_arena/participants/example/mlp.pt"}' \
     --suite dev --seeds 0:5 --out data/uav_arena/results/example_mlp_dev_s0-5.json
   ```

## Two phases per row

**Phase A: make it work.**
- Put the model in `uav_arena/participants/<row>/` the way the example is laid out: a `make`
  factory, `reset`, `observe` and `__call__`. Add tests for the call shapes (1, 512, 15, 3, 3),
  (1, 1, 15, 3, 3) and (1, 555, 15, 3, 3), for `reset` clearing episode state (and reseeding the
  random generator, if the model is stochastic), and for never reading the payload mass.
- Train on the first 250 episodes with a short schedule. Iterate with `openloop.py --stride 10` on
  the validation set, and fly `dev` once (40 episodes).
- Bar: a better selection score than the example trained the Phase A way, on the first 250
  episodes for 5,000 steps and scored with `openloop.py --stride 10`: 8.26 cm on `explore-v2`
  ([below](#example-participant)).
- Deliverable: a draft pull request into `uav-arena` with the code, tests, commands and these
  numbers. The maintainer reviews it before Phase B.

**Phase B: the protocol.**
- Up to 8 hyperparameter configurations, each trained on the 2,000 standard episodes. Choose by the
  selection score on the full validation set (`openloop.py`, every decision, four latencies), and
  record every configuration with its score.
- Train the chosen configuration with 3 new seeds, not the selection run, and run each once on
  `test` (all 8 conditions, the frozen planner).
- Train the same configuration with 3 more seeds on the first 500 standard episodes (seeds
  200000–200499), with the same number of steps and no new selection, and run each once on `test`
  too.
- In the pull request: the open-loop table per latency, the `dev` and `test` tables per condition
  for both data budgets, the parameter count, training GPU-hours and the time of one planning call
  (512 × 15). Result JSON stays under `data/`.
- Done when the maintainer merges the pull request and adds the row to the README.

## Row cards

### Entity-token JEPA `full`, `no_cross`, `flat` (Yihao)

- **Start from** `vmas_pusht/model.py` (on `main` and this branch): the entity-token LeWM with the
  same three `attn` variants, SIGReg, a readout head and a k-step rollout loss. Keep its recipe;
  change the entities and the actions.
- **Tokens:** three aircraft tokens (position, velocity, two rotation columns, body rates: 15
  values) and one payload token (position, velocity: 6) per observed frame, from `windows.py`
  (`history` ≥ 1). Take positions relative to the observed payload position, as the example does.
- **Actions and delay:** each aircraft's setpoints, the four in force (`padded_recent`) and the 15
  candidates, go to its own token. Give the model the observation's `age`: the setpoints already
  sent cover the delay, and the model learns to bridge it.
- **Readout:** the planner's cost needs the aircraft positions, the payload position and the
  payload velocity at the end of each of the 15 decisions. Train the readout head with the model,
  as `vmas_pusht` does.
- **Variants:** `full`, every token attends to every token; `no_cross`, aircraft tokens do not
  attend to other aircraft tokens (they still see themselves and the payload); `flat`, one token
  per frame, with the total latent budget matched and the parameter difference reported. One code
  base; the variants differ only in the mask or tokenization.
- **Budget:** each variant is its own row: the same grid of up to 8 configurations, selected
  separately, then 3 seeds at each of the two data budgets (2,000 and 500 episodes).
- **Shared backbone:** publish the tokenizer and transformer as a module that W-ar (next card) can
  import, and agree on its interface with june in the first week.

### Autoregressive state model, W-ar (june)

- **What it is:** the state-space reference for the JEPA rows: the same entity-token backbone,
  history, `full` mask and data as JEPA `full`, trained to predict normalized physical-state
  increments decision by decision with a multistep state loss; no latent prediction, no SIGReg.
  Against JEPA `full` it isolates latent against direct state prediction. It is a transformer,
  not an MLP: an MLP would mix in an architecture difference and duplicate PETS.
- **Start from** Yihao's backbone module. Until it lands, build the training loop, the state heads
  and the multistep loss on the example's data path.
- **Capacity:** match JEPA `full`'s parameter count as closely as possible and report both.

### PETS ensemble (june)

- **As published** (Chua et al., NeurIPS 2018): an ensemble of 5 probabilistic MLPs (4 hidden
  layers of 200 units, Swish), each predicting a Gaussian over the next state increment, trained
  with the Gaussian negative log-likelihood (bounded log-variance) on bootstrap resamples; inputs
  and outputs normalized with training statistics. The 8 configurations may widen or deepen the
  members up to the transformer rows' parameter count; report it.
- **Delay:** the first predicted step bridges the observation's `age` plus one decision; give that
  step the age and the setpoints in force (`windows.py` provides both).
- **Planning:** PETS propagates 20 particles per candidate and averages the cost over them; the
  shared planner scores one predicted trajectory per candidate. Propagate each candidate with one
  ensemble member drawn at random per planning call (TS∞ with one particle), and use that member's
  mean at every step, without Gaussian sampling. State both deviations from published PETS in the
  results: one particle instead of 20, and mean propagation. In june's code (PR #2) this is
  `sample_gaussian=False`; the default there is `True`.
- **Why the mean** (decided on 2026-10-28, before the tag; ANALYSIS.md, Section 8):
  - the validation selection score (`openloop.py --stride 10`): on `explore-v2`, 9.42 cm without
    sampling against 18.6 cm with it; on `explore-v1`, 3.04 against 5.32 cm;
  - the shared planner scores one trajectory per candidate, so nothing averages the samples the way
    published PETS averages its cost over 20 particles, and for that single trajectory a sample is
    a noisier estimate than the mean.
- **Randomness:** the member draw is still random, so set `stochastic = True`. Draw the member
  choice from torch's global generator, and reseed it in `reset()`
  ([PARTICIPANTS.md](PARTICIPANTS.md), Section 2). The harness then uses the same draws for a plan
  and its counterfactual, and averages layers 1 and 3 over 5 repeats.
- **Phase A so far:** PR #2's configuration (250 episodes, 5,000 steps per member), rerun by the
  maintainer on `explore-v2` without sampling, scores 9.42 cm against the 8.26 cm bar: still
  1.2 cm short.
- **Why an MLP:** PETS is defined with MLP members, and MLP ensembles remain the standard dynamics
  model for low-dimensional states. The transformer rows are the JEPA variants and W-ar.
- **Precedent:** PETS has been used for multi-UAV payload transport (Khursheed, Virginia Tech MS
  thesis 2024; [related work](../docs/uav-arena-related-work.md)).

### TD-MPC2 and DreamerV3 (unassigned)

Blocked until the maintainer adds the Gymnasium wrapper (README Roadmap, step 6). Then: the
official code and default settings, trained online on training layouts, evaluated as published
with their own planners, plus state probes for layers 1–3.

## Example participant

`participants/example/` is a template, not a benchmark row: a direct multi-step MLP (3 hidden
layers of 512 units) that maps the delayed state, its age, the setpoints in force and the 15
candidate setpoints to the team state at the end of each decision. Trained on the 2,000 standard
episodes for 20,000 steps (batch 1,024) on an RTX 5070 Laptop GPU:

| | Example MLP | Physics (nominal 25 g) | Physics + mass estimation |
|---|---|---|---|
| Training | 4 min, 0.75M parameters | – | – |
| Selection score (validation) | 6.24 cm | 4.01 cm | 0.25 cm |
| Phase A bar: 250 episodes, 5,000 steps (`--stride 10`) | 8.26 cm | – | – |
| Layer 1 on U's `dev` states ([HARNESS.md](HARNESS.md)) | 10.8 cm | 4.3 cm | 0.35 cm |
| `dev` closed loop | 21/40 | 28/40 | 40/40 |

- On `explore-v2` the validation score orders the example and the nominal physics reference as the
  closed loop does. On `explore-v1` it did not. There the example scored 1.46 cm against the
  physics reference's 2.93 cm, yet flew 7 of 40 `dev` episodes, 23 of its 33 failures being
  aircraft-separation events, and its layer-1 error on U's states was 14.6 cm: the plans U flies lay
  outside the `explore-v1` data (PARTICIPANTS.md, Section 5).
- On U's `dev` states the harness gives the example a cross-agent error of 4.0 cm, costly decisions
  at 32.5% of the questions and a pairwise ranking accuracy of 0.67 (HARNESS.md, Section 7).
- Scoring it takes about 10 s open loop and 30 s on `dev` (8 ms per planning call, CPU).

## Compute on a laptop

Measured on the maintainer's laptop (Ryzen 9 7845HX, RTX 5070 Laptop 8 GB) with stand-in models of
about 0.8M parameters, training at batch 256 with a 15-step rollout loss:
- 100k steps take about 10 minutes for a PETS-sized MLP ensemble and 80 minutes for an entity
  transformer.
- Phase B (14 training runs, open-loop selection, 6 `test` runs) comes to about 3.5 hours for an
  MLP row and about 21 hours for a transformer row at 100k steps per run; the three JEPA variants
  together take about 64 hours, close to three days of GPU time.
- `dev` closed loop takes minutes: planning costs 8–25 ms per decision for such models.
