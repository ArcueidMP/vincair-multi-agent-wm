"""JEPA training target alignment, latent gradients and public arena interface."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from uav_arena.collect import _write
from uav_arena.openloop import load_episode, observation
from uav_arena.windows import Windows, load, window
from uav_arena.participants.jepa.model import JEPAPredictor
from uav_arena.participants.jepa.objective import EntityJEPA, cost_features, window_states
from uav_arena.participants.jepa.train import main as train, statistics


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    import json
    root = tmp_path_factory.mktemp("jepa_data")
    _write((200000, root, "explore-v2"))
    (root/"manifest.json").write_text(json.dumps(dict(policy="explore-v2")))
    return root


def model(attn="full"):
    return EntityJEPA(dict(dim=16, latent=8, enc_layers=1, pred_layers=1, heads=4,
                           history=3, attn=attn), sigreg_projections=16)


@pytest.fixture(scope="module")
def dataset(data_dir):
    return Windows(load(data_dir), history=3)


@pytest.mark.parametrize("attn", ["full", "no_cross", "flat"])
def test_loss_and_fifteen_step_backward_are_finite_for_all_variants(attn, dataset):
    from torch.utils.data import default_collate
    torch.manual_seed(4)
    batch = default_collate([dataset[i] for i in range(4, 8)])
    net = model(attn)
    stats = statistics(dataset, count=16)
    batch["setpoints"].requires_grad_()
    loss, parts = net.loss(batch, stats, 15)
    assert torch.isfinite(loss) and set(parts) == {"latent", "sigreg", "head_absolute", "head_delta"}
    loss.backward()
    assert batch["setpoints"].grad[:, 0].abs().max() > 0
    assert torch.isfinite(batch["setpoints"].grad).all()
    for parameter in net.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_cost_features_use_only_the_fifteen_planner_outputs():
    state = torch.arange(51.)
    torch.testing.assert_close(cost_features(state), torch.cat((state[:9], state[45:51])))


def test_training_moments_are_positive_and_deterministic(dataset):
    a, b = statistics(dataset, count=16), statistics(dataset, count=16)
    assert set(a) == {"state_mean", "state_std", "delta_std"}
    for key in a:
        assert a[key].shape == (51,) and torch.isfinite(a[key]).all()
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    assert (a["state_std"] >= .001).all() and (a["delta_std"] >= .001).all()


@pytest.mark.parametrize("attn", ["full", "no_cross", "flat"])
def test_observation_padding_and_history_match_training_windows(attn, data_dir, dataset, tmp_path):
    net = model(attn)
    stats = statistics(dataset, count=16)
    checkpoint = tmp_path/"model.pt"
    torch.save(dict(config=net.config, model=net.state_dict(), stats=stats), checkpoint)
    predictor = JEPAPredictor(checkpoint, device="cpu")
    episode = load_episode(data_dir/"episode_200000.npz")

    class NoRead:
        def __getattribute__(self, name):
            raise AssertionError(f"unexpected layout access: {name}")

    for k in range(5):
        obs = observation(episode, k, 200)
        obs["payload_mass"] = object()
        predictor.observe(obs, NoRead())
        batch = {key: torch.as_tensor(value)[None] for key, value in window(episode, k, 200, history=3).items()}
        history, _, _ = window_states(batch)
        expected = predictor.model.backbone.encode((history-stats["state_mean"])/stats["state_std"])
        torch.testing.assert_close(predictor.context["z"], expected, atol=1e-6, rtol=1e-6)
    state_before = {key: value.clone() for key, value in predictor.context.items()}
    for size in (1, 512, 555):
        candidate = torch.as_tensor(episode["setpoints"][5:20], dtype=torch.float64)[None, None].expand(1, size, -1, -1, -1)
        a, b = predictor(None, candidate), predictor(None, candidate)
        assert a["uav"].shape == (1, size, 15, 3, 3) and a["payload"].shape == (1, size, 15, 3)
        assert a["payload_vel"].shape == (1, size, 15, 3)
        for key in a:
            assert a[key].dtype == torch.float64 and a[key].device.type == "cpu"
            assert torch.isfinite(a[key]).all()
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    for key in state_before:
        torch.testing.assert_close(predictor.context[key], state_before[key], atol=0, rtol=0)
    predictor.reset()
    assert not predictor.frames and predictor.context is None
    with pytest.raises(RuntimeError, match="observe"):
        predictor(None, candidate)


def test_cli_training_saves_a_loadable_checkpoint_and_refuses_overwrite(data_dir, tmp_path):
    import json
    checkpoint = tmp_path/"trained.pt"
    args = ["--data", str(data_dir), "--episodes", "1", "--steps", "8", "--batch", "4",
            "--rollout-steps", "2", "--history", "3", "--dim", "16", "--latent", "8",
            "--enc-layers", "1", "--pred-layers", "1", "--sigreg-projections", "16",
            "--stats-samples", "16", "--workers", "0", "--device", "cpu", "--out", str(checkpoint)]
    train(args)
    predictor = JEPAPredictor(checkpoint, device="cpu")
    assert predictor.history == 3
    info = json.loads(checkpoint.with_suffix(".json").read_text())
    assert info["selected_seeds"] == [200000, 200000] and info["training"]["steps"] == 8
    with pytest.raises(FileExistsError):
        train(args)


@pytest.mark.parametrize("attn", ["full", "no_cross", "flat"])
def test_candidate_future_does_not_leak_and_world_translation_is_consistent(attn, data_dir, dataset, tmp_path):
    net, stats = model(attn), statistics(dataset, count=16)
    checkpoint = tmp_path/"model.pt"
    torch.save(dict(config=net.config, model=net.state_dict(), stats=stats), checkpoint)
    a, b = JEPAPredictor(checkpoint, device="cpu"), JEPAPredictor(checkpoint, device="cpu")
    episode = load_episode(data_dir/"episode_200000.npz")
    offset = torch.tensor([1., -2., .5])
    for k in range(4):
        obs = observation(episode, k, 200)
        shifted = dict(obs)
        for key in ("uav_pos", "payload_pos", "recent_setpoints"):
            shifted[key] = obs[key]+offset.numpy()
        a.observe(obs, None)
        b.observe(shifted, None)
    candidates = torch.as_tensor(episode["setpoints"][4:19], dtype=torch.float32)[None, None]
    original = a(None, candidates)
    changed = candidates.clone()
    changed[..., -1, :, :] += 1.
    future = a(None, changed)
    translated = b(None, candidates+offset)
    for key in original:
        torch.testing.assert_close(future[key][..., :-1, :, :] if key == "uav" else future[key][..., :-1, :],
                                   original[key][..., :-1, :, :] if key == "uav" else original[key][..., :-1, :],
                                   atol=1e-6, rtol=1e-6)
        expected = original[key] if key == "payload_vel" else original[key]+offset
        torch.testing.assert_close(translated[key], expected, atol=1e-5, rtol=1e-5)
