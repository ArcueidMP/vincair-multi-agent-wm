"""The scripted team (policies.CoopPolicy) on the eval episodes: shows two agents can solve the task.

It reads the true simulator state and plans with the true friction thresholds, so it is a solvability check
and the data source, not a baseline.

    python scripts/scripted_team.py --seed 0 --budget 150    # -> $STABLEWM_HOME/plan/scripted_team_s0/
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data"))

import imageio
import numpy as np

from episodes import eval_episodes, reset
from policies import CoopPolicy
from world import MultiPushT

ENV = dict(n_agents=2, agent_force=1e5, solver_iterations=50, jpeg_quality=95)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=150)
    p.add_argument("--videos", type=int, default=4, help="save the first n episodes as videos")
    args = p.parse_args()
    out = Path(os.environ["STABLEWM_HOME"], "plan", f"scripted_team_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    res = []
    for e, (state, goal) in enumerate(zip(*eval_episodes(args.seed))):
        env = MultiPushT(max_episode_steps=args.budget, **ENV)
        _, _, placed = reset(env, args.seed, e, state, goal)
        team = CoopPolicy(np.random.default_rng(e))
        team.reset(env.core, 0)
        frames = [env.render()] if e < args.videos else None
        success = False
        while env.agents:
            _, _, term, _, _ = env.step({ag: team(env.core, i) for i, ag in enumerate(env.possible_agents)})
            success = term["agent_0"]
            if frames is not None:
                frames.append(env.render())
        res.append({"episode": e, "success": bool(success), "T_placed_at_start": bool(placed)})
        if frames is not None:
            imageio.mimsave(out / f"ep{e:02d}_{'success' if success else 'fail'}.mp4", frames, fps=10, macro_block_size=1)
        print(f"ep{e:02d} {'success' if success else 'fail   '} T placed at start: {placed}", flush=True)

    s, p0 = np.array([r["success"] for r in res]), np.array([r["T_placed_at_start"] for r in res])
    print(f"success {s.mean():.0%}; T must move: {s[~p0].sum()} / {(~p0).sum()} solved")
    (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
