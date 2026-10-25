# Released LeWM baselines and reproduction

[Back to the current research overview](../README.md).

This guide preserves the pixel and entity-token LeWM release, checkpoints, and
original evaluation results. Run commands from the repository root. Current
contact-interface and aerial-transport results are organized in the main README.
The historical VMAS planning scores precede the complete-snapshot correction
documented in the [cooperation audit](novelty-audit.md).

Can a [LeWM](https://github.com/lucas-maes/le-wm)-style world model (a JEPA that plans with CEM in latent space) plan
coordinated pushing by two agents? The two settings use different physics; whether cooperation is necessary must
be checked for the particular contact conditions, action bounds and time budget:

- **MultiPush-T** (pixels): stable-worldmodel's Push-T with two agents and a force threshold; pixel LeWM fine-tuned
  from the official single-agent checkpoint.
- **VMAS Push-T** (state): a related task in [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator), with an
  entity-token LeWM (one token per agent and one for the T). VMAS 1.5.2 uses velocity-dependent friction, not a
  strict static-friction threshold. The default parameters allow measurable solo motion and some solo successes.

Weights: [huggingface.co/fuisloy/vincair-multi-agent-wm](https://huggingface.co/fuisloy/vincair-multi-agent-wm).

## Install

The pixel pipeline uses [uv](https://docs.astral.sh/uv/), Python 3.10 and an NVIDIA GPU (PyTorch CUDA 12.8 wheels).
The released VMAS planners also support CPU through `--device cpu`.

```bash
bash scripts/install.sh              # .venv + all checkpoints under ./data (~220 MB)
bash scripts/install.sh /big/disk    # or keep data elsewhere, linked as ./data
source .venv/bin/activate
python scripts/download.py --vmas-data   # VMAS data (~300 MB): needed by wm_full_v2's probe cost and for retraining
python scripts/download.py --coop-heldout   # held-out MultiPush-T scenes (700 MB): only to refit the readouts
```

## Checkpoints

| Checkpoint | Model | For |
|---|---|---|
| `lewm_coop2_ftfull` | Pixel LeWM (ViT-tiny, 224 px), fine-tuned on cooperative data, own action | baseline planning, rollout-error analysis |
| `lewm_coop2_joint` | Same, action = [self, partner] | centralized planning, cross-agent counterfactuals |
| `lewm_coop2_*/readout/cost_probe.pt` | State readout of each pixel model: MLP probe z → [self xy, T xy, T angle] | readout cost, cost diagnostics |
| `wm_full_v2` | VMAS entity-token LeWM, full attention over agent and T tokens | planning, scalability |
| `wm_full_h_agg2` | Same, plus a jointly trained state readout head; trained on scripted + planner-visited data | cost function, decision fidelity |

Normalization ships with the models. The pixel models use the official Push-T z-scores in `assets/pusht_stats.json`
(the joint model tiles the action stats per agent), and `plan.py` applies them. Each VMAS `.pt` holds its `config`,
`mean`, `std` and `delta_std`.

### `lewm_coop2_ftfull`
```bash
python plan.py policy=lewm_coop2_ftfull                      # 50 episodes, independent CEM per agent, ~8 min
python scripts/open_loop.py lewm_coop2_ftfull                # do plans that win in the model win in the simulator?
```

### `lewm_coop2_joint`
```bash
python plan.py policy=lewm_coop2_joint planning=centralized  # one CEM over both agents' actions
python plan.py policy=lewm_coop2_joint planning=decentralized
```

Load either in Python (run from the repo root, so that `jepa.JEPA` resolves):
```python
import stable_worldmodel as swm                              # reads $STABLEWM_HOME/checkpoints/<name>/{weights.pt,config.json}
model = swm.wm.utils.load_pretrained("lewm_coop2_joint").eval()
```

### State readout of the pixel models
Each coop2 model has an MLP probe (`readout.py`) from its latent z (encoder CLS + projector, as the planner
encodes frames) to the success test's quantities: this agent's xy, T xy, T angle (sin, cos), in px. It was fit on
the model's encoder latents of the 100 held-out coop scenes, split by scene 70/10/20. Test scenes:

| readout | r (self xy / T xy / T angle) | median error: self, T, angle |
|---|---|---|
| `lewm_coop2_ftfull` | 0.97–0.98 / 0.95 / 0.96–0.97 | 11.7 px, 9.1 px, 3.9° |
| `lewm_coop2_joint` | 0.98 / 0.95–0.96 / 0.96–0.98 | 11.4 px, 8.4 px, 3.0° |

```bash
python plan.py policy=lewm_coop2_ftfull cost=probe           # plan on the readout cost instead of ||z_hat - z_goal||^2
python scripts/cost_audit.py                                 # true vs readout vs latent cost along the scripted team's eval episodes
python scripts/fit_readout.py lewm_coop2_ftfull              # refit (needs download.py --coop-heldout)
```
```python
from readout import load_probe, pose_cost, state_pose
probe, info = load_probe("data/checkpoints/lewm_coop2_ftfull/readout/cost_probe.pt")  # info: test metrics, split
pose = probe(z)                                              # (..., 6): self x, y, T x, y, sin, cos
readout_cost = pose_cost(probe(z), probe(z_goal))            # the cost=probe cost, px^2; latent cost: ((z - z_goal)**2).sum(-1)
true_cost = pose_cost(state_pose(env.state(), agent), state_pose(env.core.goal_state, agent))
```
All three costs use one form, |Δself xy|² + |ΔT xy|² + (Δangle · 20 px / 20°)². Agent 1's goal spot is its start.
The readout was fit on encoder latents of real frames. With `cost=probe` it reads the predictor's output, so
the probe also sees the model's prediction error.

### `wm_full_v2`
```bash
python -m vmas_pusht.plan --ckpt data/vmas_pusht/wm_full_v2.pt --cost decoded --probe_on predicted \
    --subgoal scripted --lookahead 15 --window 30 --horizon 3 --budget 600 --episodes 128
```

### `wm_full_h_agg2`
```bash
python -m vmas_pusht.plan --ckpt data/vmas_pusht/wm_full_h_agg2.pt --cost head --beta 2 \
    --subgoal scripted --lookahead 15 --window 30 --horizon 3 --budget 600 --episodes 128
```

Both VMAS commands plan with the learned model toward oracle subgoals, for 128 episodes from seed 42, and write
`data/vmas_pusht/eval/out.json` (`--out` to change). Add `--device cpu` without a GPU, and `--episodes 8` for a quick
run. Load in Python:
```python
from vmas_pusht.plan import load
model, mean, std = load("data/vmas_pusht/wm_full_h_agg2.pt", "cuda")   # EntityJEPA, state z-scores
z = model.encode((state - mean) / std)                                 # (B, n_agents + 1, 64) token latents
features = model.decode(z)                                             # readout head (wm_full_h_agg2 only)
```

## Results

MultiPush-T, budget 150, seeds 0–2, the 95 episodes where the T must move:

| model | planning | success | T must move |
|---|---|---|---|
| LeWM, zero-shot | independent | 27% | 1 / 95 |
| `lewm_coop2_ftfull` | independent | 33% | 6 / 95 |
| `lewm_coop2_joint` | decentralized / centralized | 31% / 31% | 1 / 95, 2 / 95 |
| scripted team | — | | 83 / 100 |

VMAS, the commands above (128 episodes, one H100):

| model | cost | solved | time |
|---|---|---|---|
| `wm_full_v2` | state decoded by probes fit on predictions | 8 / 128 | 38 min |
| `wm_full_h_agg2` | readout head, colored-noise CEM | 57 / 128 (56 with `--seed 7`) | 8 min |

In MultiPush-T the planner exploits model error: CEM plans beat the recorded actions in the model in 97–100% of
held-out windows and lose to them in the simulator just as often. Even on the true simulator with the true cost, a
25-step CEM horizon solves 5 of 30 (`scripts/sim_cem.py`). In VMAS, the cost readout was the main limit: a readout head
trained with the model, data from the planner's own visits and colored-noise CEM together lift planning from 8 to 57
of 128.

## Reproduce

```bash
# MultiPush-T (start from the official LeWM checkpoint, keep its z-scores)
python collect.py                                                     # 1000 scenes, ~6.4 GB, CPU
python collect.py seed=1 num_episodes=100 output.name=multipusht_2a_coop_heldout.h5
python scripts/make_joint_dataset.py multipusht_2a_coop.h5 multipusht_2a_coop_heldout.h5   # action = [self, partner]
python train.py                                                       # -> checkpoints/lewm_coop2_ftfull, ~4 h
python train.py data=coop_joint output_model_name=lewm_coop2_joint

# VMAS
python -m vmas_pusht.collect --episodes 4096 --batch 1024 --device cuda --out data/vmas_pusht/train.pt
python -m vmas_pusht.train --attn full --out data/vmas_pusht/wm_full_v2.pt                  # --attn no_cross | flat: ablations
python -m vmas_pusht.train --attn full --head 1 --data data/vmas_pusht/train_agg2.pt --out data/vmas_pusht/wm_full_h_agg2.pt
```

`train_agg2.pt` is `train.pt` plus two rounds of planner-visited episodes (`vmas_pusht.plan --record`, merged by
`vmas_pusht.merge_data`). To plan with a MultiPush-T model you trained, pass its weight file:
`policy=lewm_coop2_ftfull/weights_epoch_30.pt`.

## Baseline source layout

| Path | Role |
|---|---|
| `world.py` | `MultiPushT`: N-agent Push-T, force-threshold physics, per-agent views |
| `policies.py`, `collect.py` | Scripted team and data recording in the layout of LeWM's Push-T dataset |
| `jepa.py`, `module.py` | The LeWM world model, unchanged from the official code |
| `readout.py` | State readout of the pixel models (probe, pose cost); `scripts/fit_readout.py`, `scripts/cost_audit.py` |
| `train.py`, `plan.py` | Fine-tuning (joint actions, scene-level validation split); MPC with CEM for two agents |
| `vmas_pusht/` | VMAS scenario, scripted team, entity-token model (`model.py`), training, planner, subgoals |
| `config/`, `scripts/` | Hydra configs; install, download, diagnostics, upload |

## Historical release changes

- Fix `PushTN._placed` (`world.py`): it restored each body's position before its angle, and pymunk keeps the centre of
  gravity fixed when the angle changes, so rendering any state with another T angle moved the real T by about
  45 px × the angle difference. The reset path renders at the current pose and was not affected; code that renders
  subgoals or goals from the live env was.
- Add the VMAS Push-T code and the two VMAS checkpoints; `scripts/download.py` points at the released weights.
- Add the state readouts of the two pixel models with their fit script, `plan.py cost=probe` and the cost audit.

## Acknowledgements

The world model, training loop and eval protocol are [LeWM](https://github.com/lucas-maes/le-wm)'s; the Push-T
environment and the planners come from [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), and the
VMAS setting from [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator).
