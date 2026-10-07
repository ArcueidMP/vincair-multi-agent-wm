"""Plan with LeWM world models in cooperative Push-T (model-predictive control with CEM, as in LeWM's eval).

Every agent sees its own view and plans toward its own goal image (the T at its goal pose, the agent at its
goal spot; agent 1's goal spot is its start). Three ways to plan:

    independent     each agent runs its own model + CEM on its own action (own-action checkpoints)
    decentralized   each agent plans the joint action [self, partner] on its own view, executes its half
    centralized     agent 0 plans [agent 0, agent 1] and both execute it (joint-action checkpoints)

    python plan.py policy=lewm/pusht                                        # zero-shot single-agent LeWM
    python plan.py policy=lewm_coop2_ftfull
    python plan.py policy=lewm_coop2_joint planning=centralized seed=1
-> $STABLEWM_HOME/plan/<run>/results.json (+ one video per episode)
"""

import os
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parent / "data"))

import json
import time
from types import SimpleNamespace

import cv2
import hydra
import imageio
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms

from episodes import eval_episodes, reset
from utils import pusht_stats
from world import MultiPushT


class LazyWorldModelPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy that skips image preprocessing on steps where it does not replan (same actions, faster)."""

    def _prepare_info(self, info_dict):
        dead = np.asarray(info_dict.get("terminated", np.zeros(self.env.num_envs)), dtype=bool)
        if all(len(buf) > 0 or d for buf, d in zip(self._action_buffer, dead)):
            return {"terminated": info_dict.get("terminated")}
        return super()._prepare_info(info_dict)


def scalers(action_tile):
    """z-score scalers for the planner's inputs: the official Push-T data's (assets/pusht_stats.json)."""
    process = {}
    for col in ("action", "proprio", "state"):
        mean, std = pusht_stats(col, action_tile if col == "action" else 1)
        s = StandardScaler()
        s.mean_, s.scale_, s.var_, s.n_features_in_ = mean.astype(np.float64), std.astype(np.float64), std.astype(np.float64) ** 2, len(mean)
        process[col] = s
        if col != "action":
            process[f"goal_{col}"] = s
    return process


def img_transform(size):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=size),
    ])


def load_planner(cfg, n_envs, seed, action_dim):
    """One LeWM planner: its own model instance and CEM solver, batched over the n_envs episodes."""
    model = swm.wm.utils.load_pretrained(cfg.policy).to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    solver = hydra.utils.instantiate(cfg.solver, model=model, seed=seed)
    tf = img_transform(cfg.img_size)
    policy = LazyWorldModelPolicy(solver=solver, config=swm.PlanConfig(**cfg.plan_config),
                                  process=scalers(action_dim // 2), transform={"pixels": tf, "goal": tf})
    policy.set_env(SimpleNamespace(  # the planner only needs the batched action space
        num_envs=n_envs,
        action_space=spaces.Box(-1.0, 1.0, (n_envs, action_dim), np.float32),
        single_action_space=spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
    ))
    return policy


def agent_info(obs, agent, goals, last_action, done):
    """Per-agent info dict in the format WorldModelPolicy gets from swm.World."""
    stack = lambda k: np.stack([o[agent][k] for o in obs])[:, None]
    return {
        "pixels": stack("pixels"), "proprio": stack("proprio"), "state": stack("state"),
        "action": last_action[:, None],
        "goal": goals["goal"][:, None], "goal_proprio": goals["goal_proprio"][:, None], "goal_state": goals["goal_state"][:, None],
        "terminated": done.copy(),
    }


def panel(env, obs, goal_img, text, size=448):
    """Video frame: audit view (true identities) | each agent's own view | agent 0's goal."""
    tiles = [env.core.render_audit(text)] + [obs[ag]["pixels"] for ag in env.possible_agents] + [goal_img]
    labels = ["audit"] + [f"agent {i} view" for i in range(env.core.n_agents)] + ["goal"]
    out = []
    for tile, label in zip(tiles, labels):
        tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, label, (10, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        out.append(tile)
    return np.concatenate(out, axis=1)


@hydra.main(version_base=None, config_path="./config", config_name="plan")
def run(cfg: DictConfig):
    assert cfg.planning in ("independent", "decentralized", "centralized"), cfg.planning
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.budget
    states, goal_states = eval_episodes(cfg.seed)
    n = min(cfg.num_episodes, len(states))

    envs = [MultiPushT(max_episode_steps=10 * cfg.budget, **cfg.env) for _ in range(n)]
    obs, infos, placed = [], [], np.zeros(n, dtype=bool)
    for e, env in enumerate(envs):
        o, i, placed[e] = reset(env, cfg.seed, e, states[e], goal_states[e])
        obs.append(o)
        infos.append(i)
    agents = envs[0].possible_agents
    goals = [{k: np.stack([inf[ag][k] for inf in infos]) for k in ("goal", "goal_proprio", "goal_state")} for ag in agents]

    joint = cfg.planning != "independent"
    n_planners = 1 if cfg.planning == "centralized" else len(agents)
    planners = [load_planner(cfg, n, cfg.seed + i, action_dim=4 if joint else 2) for i in range(n_planners)]

    done = np.zeros(n, dtype=bool)
    success_step = np.full(n, -1)
    last = [np.zeros((n, 2), dtype=np.float32) for _ in agents]
    contact = np.zeros((n, cfg.budget, len(agents)), dtype=bool)  # agent i touches the T
    steps = np.zeros(n, dtype=int)
    frames = [[panel(env, obs[e], goals[0]["goal"][e], "t=0")] for e, env in enumerate(envs)] if cfg.video else None

    t0 = time.time()
    for t in range(cfg.budget):
        if cfg.planning == "centralized":
            a = planners[0].get_action(agent_info(obs, agents[0], goals[0], np.concatenate(last, 1), done))
            actions = [a[:, :2], a[:, 2:]]
        elif cfg.planning == "decentralized":  # plan [self, partner] on own view, execute own half
            actions = [p.get_action(agent_info(obs, ag, goals[i], np.concatenate([last[i], last[1 - i]], 1), done))[:, :2]
                       for i, (ag, p) in enumerate(zip(agents, planners))]
        else:
            actions = [p.get_action(agent_info(obs, ag, goals[i], last[i], done)) for i, (ag, p) in enumerate(zip(agents, planners))]
        last = [np.nan_to_num(a) for a in actions]

        for e, env in enumerate(envs):
            if done[e]:
                continue
            obs[e], _, term, _, info = env.step({ag: actions[i][e] for i, ag in enumerate(agents)})
            steps[e] = t + 1
            contact[e, t] = [info[ag]["block_contact"] for ag in agents]
            if term[agents[0]]:
                done[e], success_step[e] = True, t + 1
            if frames is not None:
                frames[e].append(panel(env, obs[e], goals[0]["goal"][e], f"t={t + 1}" + (" SUCCESS" if done[e] else "")))
        if done.all():
            break

    out_dir = Path(os.environ["STABLEWM_HOME"], "plan", cfg.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for e, env in enumerate(envs):
        block_err, angle_err, agent_err = env.core.errors(env.core.goal_state, env.state())
        c = contact[e, : steps[e]]
        episodes.append({"episode": e, "success": bool(done[e]), "success_step": int(success_step[e]),
                         "T_placed_at_start": bool(placed[e]), "T_pos_err": block_err, "T_angle_err": angle_err,
                         "agent_goal_dist": agent_err.tolist(), "T_contact_frac": c.mean(0).tolist(),
                         "both_touch_T_frac": float(c.all(1).mean())})
        if frames is not None:
            imageio.mimsave(out_dir / f"ep{e:02d}_{'success' if done[e] else 'fail'}.mp4", frames[e], fps=10, macro_block_size=1)
    summary = {
        "success_rate": float(done.mean()),
        "T_must_move": f"{int(done[~placed].sum())} / {int((~placed).sum())} solved",  # the cooperative part of the task
        "T_placed_at_start": f"{int(done[placed].sum())} / {int(placed.sum())} solved",
        "both_touch_T_frac": float(np.mean([ep["both_touch_T_frac"] for ep in episodes])),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print(json.dumps(summary, indent=2))
    (out_dir / "results.json").write_text(json.dumps(
        {"config": OmegaConf.to_container(cfg, resolve=True), "summary": summary, "episodes": episodes}, indent=1))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    run()
