"""Adapters of the official TD-MPC2 and DreamerV3 code (`tdmpc2_*.py`, `dreamerv3_*.py`, `upstream.py`).

Each method's tests run in its own venv with its pinned checkout under data/external/ (README,
"TD-MPC2 and DreamerV3"); elsewhere they skip. The provenance tests run anywhere.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.gym_env import OBSERVATION_SIZE, ArenaGymEnv
from uav_arena.upstream import importable, provenance

EXTERNAL = Path(__file__).resolve().parents[1]/"data"/"external"


def test_provenance_requires_the_pinned_commit_and_unchanged_tracked_files(tmp_path):
    git = lambda *args: subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("init", "-q")
    (tmp_path/"code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "pinned")
    commit = git("rev-parse", "HEAD")
    (tmp_path/"outputs").mkdir()                       # untracked run outputs are fine
    assert provenance(tmp_path, "url", commit)["commit"] == commit
    with pytest.raises(ValueError, match="not the pinned commit"):
        provenance(tmp_path, "url", "0"*40)
    (tmp_path/"code.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="unchanged"):
        provenance(tmp_path, "url", commit)


def test_setup_script_installs_the_commits_the_adapters_check():
    import importlib.util
    from uav_arena import dreamerv3_train, tdmpc2_train
    path = Path(__file__).resolve().parents[1]/"scripts"/"setup_online_rl.py"
    spec = importlib.util.spec_from_file_location("setup_online_rl", path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    for name, module in (("tdmpc2", tdmpc2_train), ("dreamerv3", dreamerv3_train)):
        assert (setup.METHODS[name]["url"], setup.METHODS[name]["commit"]) == (module.UPSTREAM_URL, module.UPSTREAM_COMMIT)


def _actions(seed, n):
    return np.random.default_rng(seed).uniform(-1, 1, (n, 9)).astype(np.float32)


# ---------- TD-MPC2 ----------

@pytest.fixture(scope="module")
def tdmpc2():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    pytest.importorskip("hydra")
    if not (EXTERNAL/"tdmpc2"/".git").exists():
        pytest.skip("needs the pinned TD-MPC2 checkout in data/external/tdmpc2")
    from uav_arena import tdmpc2_train
    provenance(EXTERNAL/"tdmpc2", tdmpc2_train.UPSTREAM_URL, tdmpc2_train.UPSTREAM_COMMIT)
    with importable(EXTERNAL/"tdmpc2"/"tdmpc2"):
        yield tdmpc2_train, torch


class _Cfg:
    def __init__(self, **values):
        self.__dict__.update(values)

    def get(self, key, default=None):
        return getattr(self, key, default)


def test_tdmpc2_task_follows_upstreams_contract_and_steps_the_wrapper(tdmpc2):
    tdmpc2_train, torch = tdmpc2
    cfg = _Cfg(task="uav-arena", seed=3, obs="state")
    env = tdmpc2_train.make_env(cfg)
    assert (cfg.obs_shape, cfg.action_dim, cfg.episode_length, cfg.seed_steps, cfg.rho) == (
        {"state": (OBSERVATION_SIZE,)}, 9, 200, 1000, 0.7)
    with pytest.raises(ValueError):
        tdmpc2_train.make_env(_Cfg(task="walker-walk", seed=0, obs="state"))
    reference = ArenaGymEnv()
    observation, _ = reference.reset(seed=3)
    obs = env.reset()
    assert obs.dtype == torch.float32 and np.array_equal(obs.numpy(), observation)
    for action in _actions(0, 200):
        obs, reward, done, info = env.step(torch.from_numpy(action))
        observation, gym_reward, terminated, truncated, gym_info = reference.step(action)
        assert np.array_equal(obs.numpy(), observation) and reward.item() == np.float32(gym_reward)
        assert done == (terminated or truncated) and info["terminated"].item() == float(terminated)
        assert info["success"] == float(gym_info["success"]) and "payload_mass" not in info
        if done:
            break
    assert done and info["terminated"].item() == 1.0     # random actions end in an event, not the time limit


def test_tdmpc2_time_limit_is_done_but_not_terminated(tdmpc2):
    tdmpc2_train, torch = tdmpc2
    env = tdmpc2_train.make_env(_Cfg(task="uav-arena", seed=0, obs="state"))
    env.reset()
    env.unwrapped.reset(options=dict(seed=0, condition="static/0ms"))   # a hover that times out
    steps, done = 0, False
    while not done:
        _, _, done, info = env.step(torch.zeros(9))
        steps += 1
    assert (steps, info["outcome"], info["terminated"].item()) == (200, "timeout", 0.0)


def test_tdmpc2_trains_saves_and_replays_an_episode(tdmpc2, tmp_path):
    tdmpc2_train, torch = tdmpc2
    if not torch.cuda.is_available():
        pytest.skip("upstream TD-MPC2 runs on CUDA only")
    from uav_arena.gym_eval import run_episode
    from uav_arena.tdmpc2_policy import make
    report = tdmpc2_train.train(EXTERNAL/"tdmpc2", tmp_path/"run", seed=1, steps=1100,
                                overrides=["compile=false", "eval_episodes=1"])
    config = report["config"]
    assert (config["task"], config["episodic"], config["rho"], config["episode_length"]) == ("uav-arena", True, 0.7, 200)
    assert (tmp_path/"run.pt").exists() and (tmp_path/"run.log.jsonl").exists()
    policy, env = make(tmp_path/"run"), ArenaGymEnv()
    first, second = (run_episode(env, policy, 0, "static/0ms") for _ in range(2))
    assert {k: v for k, v in first.items() if not k.startswith("plan")} == \
        {k: v for k, v in second.items() if not k.startswith("plan")}


# ---------- DreamerV3 ----------

@pytest.fixture(scope="module")
def dreamerv3():
    pytest.importorskip("jax")
    elements = pytest.importorskip("elements")
    if not (EXTERNAL/"dreamerv3"/".git").exists():
        pytest.skip("needs the pinned DreamerV3 checkout in data/external/dreamerv3")
    from uav_arena import dreamerv3_train
    provenance(EXTERNAL/"dreamerv3", dreamerv3_train.UPSTREAM_URL, dreamerv3_train.UPSTREAM_COMMIT)
    with importable(EXTERNAL/"dreamerv3"):
        from dreamerv3 import main as upstream_main
        import ruamel.yaml
        configs = ruamel.yaml.YAML(typ="safe").load((EXTERNAL/"dreamerv3"/"dreamerv3"/"configs.yaml").read_text())
        config = elements.Config(configs["defaults"]).update(configs[dreamerv3_train.PRESET])
        dreamerv3_train.register(upstream_main)
        yield dreamerv3_train, upstream_main, config.update(task=dreamerv3_train.TASK, seed=0)


def test_dreamerv3_task_passes_upstreams_wrappers_and_steps_the_wrapper(dreamerv3):
    dreamerv3_train, upstream_main, config = dreamerv3
    env = upstream_main.make_env(config, 0)            # NormalizeAction, UnifyDtypes, CheckSpaces, ClipAction
    assert set(env.obs_space) == {"state", "reward", "is_first", "is_last", "is_terminal", "log/success"}
    assert env.obs_space["state"].shape == (OBSERVATION_SIZE,) and env.act_space["action"].shape == (9,)
    ours, reference = dreamerv3_train.ArenaTask("train", seed=5), ArenaGymEnv()
    obs = ours.step({"action": np.zeros(9, np.float32), "reset": True})
    observation, _ = reference.reset(seed=5)
    assert obs["is_first"] and not obs["is_last"] and np.array_equal(obs["state"], observation)
    for action in _actions(1, 200):
        obs = ours.step({"action": action, "reset": False})
        observation, reward, terminated, truncated, info = reference.step(action)
        assert np.array_equal(obs["state"], observation) and obs["reward"] == np.float32(reward)
        assert (obs["is_first"], obs["is_last"], obs["is_terminal"]) == (False, terminated or truncated, terminated)
        if obs["is_last"]:
            break
    assert obs["is_terminal"] and obs["log/success"] == 0.0   # random actions end in an event
    obs = ours.step({"action": np.zeros(9, np.float32), "reset": False})
    assert obs["is_first"] and not obs["is_last"]             # an ended episode restarts


def test_dreamerv3_time_limit_is_last_but_not_terminal(dreamerv3):
    dreamerv3_train, upstream_main, config = dreamerv3
    task = dreamerv3_train.ArenaTask("train", seed=0)
    task.step({"action": np.zeros(9, np.float32), "reset": True})
    task.env.reset(options=dict(seed=0, condition="static/0ms"))         # a hover that times out
    steps, obs = 0, {"is_last": False}
    while not obs["is_last"]:
        obs = task.step({"action": np.zeros(9, np.float32), "reset": False})
        steps += 1
    assert (steps, obs["is_terminal"], obs["reward"] < -9) == (200, False, True)


def test_dreamerv3_trains_saves_and_replays_an_episode(dreamerv3, tmp_path):
    dreamerv3_train, upstream_main, config = dreamerv3
    from uav_arena.dreamerv3_policy import make
    from uav_arena.gym_eval import run_episode
    report = dreamerv3_train.train(EXTERNAL/"dreamerv3", tmp_path/"run", seed=0, steps=300,
                                   configs=(dreamerv3_train.PRESET, "debug"))   # tiny networks on the CPU
    assert (report["config"]["task"], report["config"]["seed"]) == ("arena_train", 0)
    assert (tmp_path/"run"/"final"/"done").exists() and (tmp_path/"run.json").exists()
    policy, env = make(tmp_path/"run"), ArenaGymEnv()
    first, second = (run_episode(env, policy, 0, "static/0ms") for _ in range(2))
    assert {k: v for k, v in first.items() if not k.startswith("plan")} == \
        {k: v for k, v in second.items() if not k.startswith("plan")}
