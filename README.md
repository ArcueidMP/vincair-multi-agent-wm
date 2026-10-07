# MultiPush-T LeWM

Can a world model trained on one agent pushing a T plan for **two agents that must push it together**?

This repo puts [LeWM](https://github.com/lucas-maes/le-wm) (a JEPA world model that plans with CEM in latent space) into a cooperative version of Push-T. The T is too heavy for one agent: it moves only when both agents push at the same time. Like LeWM, the repo holds only the core code; [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) runs data, planning and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) runs training.

**Status (Oct 2026):** fine-tuned LeWM models predict the joint push, but planning with them almost never produces one: 6 of 95 episodes solved where the T must move, against 83 of 100 for a scripted team. With the real simulator as its model, the same CEM search solves 5 of 30 even with the true task cost, so the search is the main limit. See [Results](#results).

## Install

Requires [uv](https://docs.astral.sh/uv/), Python 3.10 and an NVIDIA GPU (the lock file uses PyTorch CUDA 12.8 wheels).

```bash
bash scripts/install.sh              # .venv + checkpoints under ./data
bash scripts/install.sh /big/disk    # or keep data elsewhere, linked as ./data
source .venv/bin/activate
```

## Quick start

```bash
python scripts/scripted_team.py --seed 0               # the task is solvable: a scripted team (CPU)
python plan.py policy=lewm/pusht                        # official single-agent LeWM, zero-shot, for both agents
python plan.py policy=lewm_coop2_ftfull                 # fine-tuned on cooperative data
python plan.py policy=lewm_coop2_joint planning=centralized   # one planner for both agents' actions
```

Each run writes `data/plan/<run>/results.json` and one video per episode: the audit view (true agent identities), each agent's own view (self blue, partner orange) and the goal. A run is 50 episodes, about 8 minutes on one H100 MIG slice (`lewm_coop2_ftfull`, seed 0: 32% success, 2 of 30 where the T must move).

## The environment

`world.py` builds `MultiPushT`, a PettingZoo `ParallelEnv`, on stable-worldmodel's single-agent Push-T. Physics, T, walls, goal and rendering are the same, so a single-agent checkpoint plugs into every agent.

- **Views.** Each agent sees a 224 px frame in the Push-T layout: itself blue, the partner orange, the T gray, the goal green.
- **Force threshold** (`agent_force=1e5`). Each agent pushes with at most `agent_force`. The T has floor friction of 1.5 × that force and 1.1 × that torque, so one agent can neither slide nor turn it, and two can.
- **Goals and success.** Agent 0's goal is the T at its goal pose and agent 0 at its goal spot; this is the original Push-T test. Agent 1's goal spot is its start.
- **Eval episodes.** The 50 start/goal pairs per seed of LeWM's Push-T eval, exported to `assets/eval_episodes.npz`, with the partner placed away from the T. In 13–22 of the 50, the T already starts within tolerance; *T must move* counts the rest.

## Layout

| File | Role |
|---|---|
| `world.py` | `MultiPushT`: N-agent Push-T, force-threshold physics, per-agent views |
| `policies.py` | Scripted policies: `CoopPolicy` (two-agent team, the data source) and `HeuristicPolicy` |
| `collect.py` | Record scripted rollouts in the layout of LeWM's Push-T dataset, one episode per agent |
| `jepa.py`, `module.py` | The LeWM world model, unchanged from the official code |
| `train.py` | LeWM training, plus fine-tuning from a checkpoint, joint actions and a scene-level validation split |
| `plan.py` | MPC with CEM for two agents: `independent`, `decentralized` or `centralized` planning |
| `episodes.py`, `utils.py` | Eval episodes, z-scores (`assets/`), preprocessing |
| `config/` | Hydra configs for `collect`, `train`, `plan` |
| `scripts/` | Install, download, joint-action dataset, scripted team, the two diagnostics below |

## Reproduce the models

```bash
python collect.py                                                     # 1000 scenes, ~6.4 GB, CPU
python collect.py seed=1 num_episodes=100 output.name=multipusht_2a_coop_heldout.h5
python scripts/make_joint_dataset.py multipusht_2a_coop.h5 multipusht_2a_coop_heldout.h5   # action = [self, partner]
python train.py                                                       # -> checkpoints/lewm_coop2_ftfull, ~4 h
python train.py data=coop_joint output_model_name=lewm_coop2_joint
```

Both start from the official LeWM checkpoint and keep its z-scores (`assets/pusht_stats.json`). The joint model's action encoder is widened so that it starts as the same function, with the partner's action ignored. To plan with a model you trained, give `plan.py` the weight file, e.g. `policy=lewm_coop2_ftfull/weights_epoch_30.pt`.

## Diagnostics

```bash
python scripts/open_loop.py lewm_coop2_ftfull       # does the plan that wins in the model win in reality? (GPU)
python scripts/sim_cem.py --cost true --procs 16    # CEM on the true simulator with the true cost (CPU)
python scripts/sim_cem.py --cost lewm_coop2_ftfull  # ... with a model's latent cost (GPU)
```

## Results

Budget 150, seeds 0–2, 95 episodes where the T must move:

| model | planning | success | T must move |
|---|---|---|---|
| LeWM, zero-shot | independent | 27% | 1 / 95 |
| `lewm_coop2_ftfull` | independent | 33% | 6 / 95 |
| `lewm_coop2_joint` | decentralized | 31% | 1 / 95 |
| `lewm_coop2_joint` | centralized | 31% | 2 / 95 |
| scripted team | — | | 83 / 100 |

Seed 0, the 30 episodes where the T must move:

| planner | solved |
|---|---|
| CEM in the world model (`lewm_coop2_joint`, centralized) | 1 / 30 |
| CEM in the true simulator, latent cost | 3–6 / 30 |
| CEM in the true simulator, true task cost | 5 / 30 |

- **Model error.** In the model, CEM's plans beat the recorded actions in 97–100% of held-out windows. Executed in the simulator, they lose in 97–100%: the planner exploits model error.
- **Search.** Even with perfect dynamics and cost, a 25-step CEM horizon rarely finds a plan where both agents walk to the T and push together.
- **Tried, with no gain:** more CEM samples, iCEM, a 50-step horizon, a cost read by a position probe, and keeping plans near the data's actions.

## Acknowledgements

The world model, training loop and eval protocol are [LeWM](https://github.com/lucas-maes/le-wm)'s; the Push-T environment and the planners come from [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).
