"""Frozen readout diagnostics: parameter isolation, units, splits and outputs."""
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from uav_arena.collect import _write
from uav_arena.participants.jepa.objective import EntityJEPA, StateReadout
from uav_arena.participants.jepa.probe_readout import (
    evaluation_indices, load_frozen, main, physical_errors, state_digest,
)


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(7)
    model = EntityJEPA(dict(dim=16, latent=8, enc_layers=1, pred_layers=1,
                           heads=4, history=1), sigreg_projections=16)
    stats = dict(state_mean=torch.zeros(51), state_std=torch.ones(51), delta_std=torch.ones(51))
    path = tmp_path/"parent.pt"
    torch.save(dict(config=model.config, model=model.state_dict(), stats=stats), path)
    return path


def test_only_new_readout_receives_gradients(checkpoint):
    model, _, _ = load_frozen(checkpoint, "cpu")
    before = state_digest(model)
    head = StateReadout(model.backbone.token_dim, model.backbone.attn)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    with torch.no_grad():
        z = model.backbone.encode(torch.randn(4, 16, 51))
    loss = head(z).square().mean()
    loss.backward()
    optimizer.step()
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert before == state_digest(model)


def test_report_converts_normalized_errors_to_physical_units():
    prediction, target = torch.zeros(2, 16, 15), torch.zeros(2, 16, 15)
    prediction[..., 0:9:3] = .01
    prediction[..., 11] = .03
    prediction[..., 12] = .04
    errors = physical_errors(prediction, target, dict(state_std=torch.full((51,), 2.)))
    torch.testing.assert_close(errors["uav_cm"], torch.full((2, 16), 2.))
    torch.testing.assert_close(errors["payload_cm"], torch.full((2, 16), 6.))
    torch.testing.assert_close(errors["payload_velocity_m_s"], torch.full((2, 16), .08))


def test_validation_stride_restarts_for_each_episode():
    class Data:
        latencies = (0, 50, 100, 200)
        offsets = np.array([0, 100, 264])

    expected = [base+j for base in (0, 40, 80, 100, 140, 180, 220, 260) for j in range(4)]
    assert evaluation_indices(Data(), 10) == expected


def test_probe_cli_is_training_only_preserves_parent_and_saves_separate_head(checkpoint, tmp_path):
    train, val = tmp_path/"training", tmp_path/"validation"
    for directory, seed in ((train, 200000), (val, 299000)):
        directory.mkdir()
        _write((seed, directory, "explore-v2"))
        (directory/"manifest.json").write_text(json.dumps(dict(policy="explore-v2")))
    # A validation directory cannot accidentally supply the selected train seeds.
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    prefix = tmp_path/"diagnostic"
    args = ["--checkpoint", str(checkpoint), "--data", str(train), "--val-data", str(val),
            "--episodes", "1", "--val-episodes", "1", "--steps", "8", "--batch", "4",
            "--device", "cpu", "--out", str(prefix)]
    report = main(args)
    assert report["frozen_parameters_unchanged"]
    assert report["training"]["seeds_inclusive"] == [200000, 200000]
    assert report["validation"]["seeds_inclusive"] == [299000, 299000]
    assert report["parent_checkpoint_sha256"] == digest
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    weights = torch.load(str(prefix)+".probe.pt", weights_only=True)
    assert "model" not in weights and weights["kind"] == "frozen_encoder_readout_probe"
    for head in ("original", "probe"):
        metrics = report["measurements"]["validation"][head]
        assert len(metrics["overall"]["payload_cm"]["future_steps"]) == 15
        assert set(metrics["by_latency_ms"]) == {"0", "50", "100", "200"}
    with pytest.raises(FileExistsError):
        main(args)
    bad = args.copy()
    bad[bad.index("--data")+1] = str(val)
    bad[bad.index("--out")+1] = str(tmp_path/"bad")
    with pytest.raises(ValueError, match="missing selected"):
        main(bad)
