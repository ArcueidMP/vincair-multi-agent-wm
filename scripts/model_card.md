---
tags:
  - world-model
  - jepa
  - lewm
  - multi-agent
  - push-t
  - planning
---

# Multi-agent Push-T world models (LeWM)

Checkpoints for [fuisl/vincair-multi-agent-wm](https://github.com/fuisl/vincair-multi-agent-wm): LeWM-style JEPA world
models for cooperative Push-T, where the T moves only when two agents push it together. Code, install and the full
commands are in the GitHub README.

| Path | Model | Use |
|---|---|---|
| `lewm_coop2_ftfull/` | Pixel LeWM (ViT-tiny, 224 px), fine-tuned from [quentinll/lewm-pusht](https://huggingface.co/quentinll/lewm-pusht) on MultiPush-T, own action | baseline planning, rollout-error analysis |
| `lewm_coop2_joint/` | Same, action = [self, partner] | centralized planning, cross-agent counterfactuals |
| `lewm_coop2_{ftfull,joint}/readout/cost_probe.pt` | State readout of each pixel model: MLP probe z → [self xy, T xy, T angle], fit on held-out scenes (test median error ~9 px T, 3–4°, 11 px self) | readout cost vs latent cost vs true cost |
| `datasets/multipusht_2a_coop_heldout.h5` | The 100 held-out MultiPush-T scenes the readouts were fit and tested on | refitting the readouts |
| `vmas_pusht/wm_full_v2.pt` | Entity-token LeWM on VMAS Push-T, full attention over agent and T tokens | planning, scalability |
| `vmas_pusht/wm_full_h_agg2.pt` | Same, with a jointly trained state readout head, trained on scripted + planner-visited data | cost function, decision fidelity |
| `vmas_pusht/{train,heldout,train_agg2}.pt` | VMAS data: scripted train (4096 episodes), held-out, train + planner visits | probes for `wm_full_v2`, retraining |

Normalization: the pixel models use the official Push-T z-scores (`assets/pusht_stats.json` in the code repo; the
joint model tiles the action stats per agent). Each VMAS file holds its own `config`, `mean`, `std` and `delta_std`.

```bash
git clone https://github.com/fuisl/vincair-multi-agent-wm && cd vincair-multi-agent-wm
bash scripts/install.sh && source .venv/bin/activate      # downloads these checkpoints into ./data
python plan.py policy=lewm_coop2_ftfull
python plan.py policy=lewm_coop2_ftfull cost=probe          # plan on the readout cost
python scripts/cost_audit.py                                # true vs readout vs latent cost
python -m vmas_pusht.plan --ckpt data/vmas_pusht/wm_full_h_agg2.pt --cost head --beta 2 \
    --subgoal scripted --lookahead 15 --window 30 --horizon 3 --budget 600 --episodes 8
```
