import shutil

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_worldmodel")
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="the pinned C controller needs a compiler")

from uav_arena.collect import _write
from uav_arena.env import OUTCOMES
from uav_arena.openloop import evaluate, load_episode, observation
from uav_arena.participants.example import train
from uav_arena.planning import ArenaCost, ArenaPlanner
from uav_arena.run import make_predictor, run_episode
from uav_arena.scenarios import CONDITIONS, make_scenario

FACTORY = "uav_arena.participants.example.model:make"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("example")
    _write((200_003, root))
    checkpoint = root/"mlp.pt"
    train.main(["--data", str(root), "--episodes", "1", "--steps", "20", "--batch", "64", "--width", "64",
                "--workers", "0", "--device", "cpu", "--out", str(checkpoint)])
    return root, checkpoint


def test_example_answers_the_planners_call_and_scores_in_openloop(trained):
    root, checkpoint = trained
    predictor = make_predictor(FACTORY, options={"checkpoint": str(checkpoint)})
    episode = load_episode(root/"episode_200003.npz")
    predictor.reset()
    predictor.observe(observation(episode, 10, 50), episode["layout"])
    prediction = predictor(None, torch.zeros(1, 512, 15, 3, 3))
    assert prediction["uav"].shape == (1, 512, 15, 3, 3) and prediction["payload"].shape == (1, 512, 15, 3)
    assert prediction["payload_vel"].shape == (1, 512, 15, 3) and prediction["payload"].dtype == torch.float32
    result = evaluate(predictor, [episode], 100, horizon=15, stride=50)
    assert result["decisions"] == 4 and np.isfinite(result["payload"]).all()


def test_example_flies_a_closed_loop_episode(trained):
    _, checkpoint = trained
    predictor = make_predictor(FACTORY, options={"checkpoint": str(checkpoint)})
    record = run_episode(make_scenario(0, CONDITIONS[0]), ArenaPlanner(ArenaCost(predictor)))
    assert record["outcome"] in OUTCOMES and record["decisions"] >= 1
