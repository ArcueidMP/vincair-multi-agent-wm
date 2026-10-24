# Multi-agent Push-T world models (LeWM)

> Collaboration branch: see [COLLABORATION_START_HERE.md](COLLABORATION_START_HERE.md) for current E0/E1/E4 results, portable Navigation reproduction, known limitations, and team tasks. The Navigation hybrid is distinct from the Push-T LeWM models described below.

Can a [LeWM](https://github.com/lucas-maes/le-wm)-style world model (a JEPA that plans with CEM in latent space) plan for
**two agents that must push a T together**? The T is too heavy for one agent; it moves only when both push. Two
settings:

- **MultiPush-T** (pixels): stable-worldmodel's Push-T with two agents and a force threshold; pixel LeWM fine-tuned
  from the official single-agent checkpoint.
- **VMAS Push-T** (state): the same task in [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator), with an
  entity-token LeWM (one token per agent and one for the T).

Weights: [huggingface.co/fuisloy/vincair-multi-agent-wm](https://huggingface.co/fuisloy/vincair-multi-agent-wm).

## Install

Requires [uv](https://docs.astral.sh/uv/), Python 3.10 and an NVIDIA GPU (PyTorch CUDA 12.8 wheels).

```bash
bash scripts/install.sh              # .venv + all checkpoints under ./data (~220 MB)
bash scripts/install.sh /big/disk    # or keep data elsewhere, linked as ./data
source .venv/bin/activate
python scripts/download.py --vmas-data   # VMAS data (~300 MB): needed by wm_full_v2's probe cost and for retraining
```

## Checkpoints

| Checkpoint | Model | For |
|---|---|---|
| `lewm_coop2_ftfull` | Pixel LeWM (ViT-tiny, 224 px), fine-tuned on cooperative data, own action | baseline planning, rollout-error analysis |
| `lewm_coop2_joint` | Same, action = [self, partner] | centralized planning, cross-agent counterfactuals |
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

## Layout

| Path | Role |
|---|---|
| `world.py` | `MultiPushT`: N-agent Push-T, force-threshold physics, per-agent views |
| `policies.py`, `collect.py` | Scripted team and data recording in the layout of LeWM's Push-T dataset |
| `jepa.py`, `module.py` | The LeWM world model, unchanged from the official code |
| `train.py`, `plan.py` | Fine-tuning (joint actions, scene-level validation split); MPC with CEM for two agents |
| `vmas_pusht/` | VMAS scenario, scripted team, entity-token model (`model.py`), training, planner, subgoals |
| `config/`, `scripts/` | Hydra configs; install, download, diagnostics, upload |

## Changes in this release

- Fix `PushTN._placed` (`world.py`): it restored each body's position before its angle, and pymunk keeps the centre of
  gravity fixed when the angle changes, so rendering any state with another T angle moved the real T by about
  45 px × the angle difference. The reset path renders at the current pose and was not affected; code that renders
  subgoals or goals from the live env was.
- Add the VMAS Push-T code and the two VMAS checkpoints; `scripts/download.py` points at the released weights.

## Acknowledgements

The world model, training loop and eval protocol are [LeWM](https://github.com/lucas-maes/le-wm)'s; the Push-T
environment and the planners come from [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), and the
VMAS setting from [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator).
