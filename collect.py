"""Record multi-agent Push-T rollouts of a scripted policy in the layout of LeWM's pusht_expert_train.h5.

Every scene (env episode) becomes N episodes, one per agent, each seen from that agent's own view, so
train.py reads the file unchanged:

    pixels (224, 224, 3) uint8, proprio (4,), state (7,), action (2,)   as in pusht_expert_train.h5
    episode_idx, step_idx, scene_idx, agent_idx                         one episode id per (scene, agent)
    global_state (4N + 3,), joint_action (2N,)                          Markov-game state / joint action
    block_contact, agent_contact                                        contacts of this agent

action[t] is applied after observation t; the last row of an episode holds NaN (train.py maps it to 0).

    python collect.py                                                    # 1000 coop scenes -> multipusht_2a_coop.h5
    python collect.py seed=1 num_episodes=100 output.name=multipusht_2a_coop_heldout.h5
"""

import os
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parent / "data"))

import hdf5plugin
import hydra
import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from policies import POLICIES
from world import MultiPushT


class BloscHDF5Writer(swm.data.formats.hdf5.HDF5Writer):
    """swm's HDF5Writer with the chunking and compression of the official LeWM datasets."""

    def _init_schema(self, sample_ep):
        for col, vals in sample_ep.items():
            sample = np.asarray(vals[0])
            image = col.startswith('pixels')
            self._f.create_dataset(
                col, shape=(0, *sample.shape), maxshape=(None, *sample.shape), dtype=sample.dtype,
                chunks=(100 if image else 1000, *sample.shape),
                **(hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE) if image else {}),
            )
        for col, dtype in (('ep_len', np.int32), ('ep_offset', np.int64)):
            self._f.create_dataset(col, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(1024,))


def rollout(env, policies, seed):
    """One env episode -> list of N per-agent episodes (dicts of per-step lists)."""
    obs, infos = env.reset(seed=seed)
    core = env.core
    for i, p in enumerate(policies):
        p.reset(core, i)
    eps = [{k: [] for k in ('pixels', 'proprio', 'state', 'action', 'global_state', 'joint_action',
                            'block_contact', 'agent_contact')} for _ in policies]

    def record(obs, infos, state):
        for ep, ag in zip(eps, env.possible_agents):
            ep['pixels'].append(obs[ag]['pixels'])
            ep['proprio'].append(obs[ag]['proprio'].astype(np.float32))
            ep['state'].append(obs[ag]['state'].astype(np.float32))
            ep['global_state'].append(state.astype(np.float32))
            ep['block_contact'].append(np.int8(infos[ag]['block_contact']))
            ep['agent_contact'].append(np.int8(infos[ag]['agent_contact']))

    record(obs, infos, env.state())
    while env.agents:
        actions = {ag: np.asarray(p(core, i), dtype=np.float32)
                   for i, (ag, p) in enumerate(zip(env.possible_agents, policies))}
        joint = np.concatenate([actions[ag] for ag in env.possible_agents])
        for ep, ag in zip(eps, env.possible_agents):
            ep['action'].append(actions[ag])
            ep['joint_action'].append(joint)
        obs, _, _, _, infos = env.step(actions)
        record(obs, infos, env.state())
    for ep in eps:
        ep['action'].append(np.full(2, np.nan, dtype=np.float32))
        ep['joint_action'].append(np.full(2 * len(eps), np.nan, dtype=np.float32))
    return eps


@hydra.main(version_base=None, config_path="./config", config_name="collect")
def run(cfg: DictConfig):
    env = MultiPushT(**OmegaConf.to_container(cfg.env, resolve=True))
    N = env.core.n_agents
    rng = np.random.default_rng(cfg.seed)
    kwargs = OmegaConf.to_container(cfg.policy_kwargs, resolve=True)
    policy_cls = POLICIES[cfg.policy]
    if getattr(policy_cls, 'joint', False):  # one team planner shared by all agents
        policies = [policy_cls(np.random.default_rng(cfg.seed), **kwargs)] * N
    else:
        policies = [policy_cls(np.random.default_rng([cfg.seed, i]), **kwargs) for i in range(N)]

    path = Path(os.environ['STABLEWM_HOME']) / 'datasets' / cfg.output.name
    path.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    with BloscHDF5Writer(path, mode=cfg.output.mode) as w:
        for scene in tqdm(range(cfg.num_episodes), desc='Recording'):
            eps = rollout(env, policies, seed=int(rng.integers(2**31)))
            T = len(eps[0]['action'])
            for i, ep in enumerate(eps):
                ep['episode_idx'] = [np.int64(scene * N + i)] * T
                ep['step_idx'] = list(np.arange(T, dtype=np.int64))
                ep['scene_idx'] = [np.int64(scene)] * T
                ep['agent_idx'] = [np.int64(i)] * T
                w.write_episode(ep)
            success += T - 1 < env.max_episode_steps
    print(f"wrote {path}: {cfg.num_episodes} scenes x {N} agents, success {success / cfg.num_episodes:.1%}")


if __name__ == "__main__":
    run()
