"""Fresh training keeps the original shuffle order and restores complete state."""
import json

import pytest
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from uav_arena.participants.jepa import train
from uav_arena.participants.jepa.training_checkpoint import CheckpointBatches


@pytest.mark.parametrize("workers", [0, 2])
def test_checkpoint_sampler_matches_original_loader_across_epochs(workers):
    data = torch.arange(17)
    a = DataLoader(data, batch_size=3, shuffle=True, drop_last=True, num_workers=workers,
                   persistent_workers=workers > 0, generator=torch.Generator().manual_seed(0))
    generator = torch.Generator().manual_seed(0)
    sampler = CheckpointBatches(BatchSampler(RandomSampler(data, generator=generator), 3, True))
    b = DataLoader(data, batch_sampler=sampler, num_workers=workers,
                   persistent_workers=workers > 0, generator=generator)
    for epoch in range(4):
        if epoch:
            sampler.next_epoch()
        expected = list(a)
        actual = []
        for batch in b:
            actual.append(batch)
            sampler.commit()
        assert len(expected) == len(actual)
        assert all(torch.equal(x, y) for x, y in zip(expected, actual))


class Tiny(nn.Module):
    def __init__(self, options, **config):
        super().__init__()
        self.config = dict(backbone_options=options, **config)
        self.linear = nn.Linear(2, 1)
        self.dropout = nn.Dropout(.2)

    def loss(self, batch, stats, steps):
        loss = (self.linear(self.dropout(batch["x"]))-batch["y"]).square().mean()
        return loss, dict(latent=loss.detach())


def assert_same(x, y):
    if isinstance(x, torch.Tensor):
        assert torch.equal(x, y)
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for k in x:
            assert_same(x[k], y[k])
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for xx, yy in zip(x, y):
            assert_same(xx, yy)
    else:
        assert x == y


@pytest.mark.parametrize("workers", [0, 2])
def test_fresh_split_resume_matches_continuous_and_rejects_changed_data(tmp_path, monkeypatch, workers):
    monkeypatch.setattr(train, "EntityJEPA", Tiny)
    monkeypatch.setattr(train, "load", lambda *a, **kw: [object()])
    dataset = [{"x": torch.tensor([float(i), .5]), "y": torch.tensor([i*.2])} for i in range(7)]
    monkeypatch.setattr(train, "Windows", lambda *a, **kw: dataset)
    monkeypatch.setattr(train, "statistics", lambda *a, **kw: {})
    data = tmp_path/"data"
    data.mkdir()
    manifest = data/"manifest.json"
    manifest.write_text(json.dumps(dict(policy="explore-v2")))
    common = ["--data", str(data), "--episodes", "1", "--steps", "8", "--history", "1",
              "--batch", "2", "--device", "cpu", "--workers", str(workers)]
    continuous, split = tmp_path/"continuous.pt", tmp_path/"split.pt"
    train.main(common+["--out", str(continuous)])
    train.main(common+["--out", str(split), "--stop-after", "4"])
    resume = split.with_suffix(".resume.pt")
    assert not split.exists()
    paused = torch.load(resume, weights_only=True)
    assert paused["step"] == 4 and paused["sampler"]["epoch"] == 1
    manifest.write_text(json.dumps(dict(policy="explore-v2", changed=True)))
    with pytest.raises(ValueError, match="source or data manifest"):
        train.main(["--resume", str(resume), "--out", str(split), "--device", "cpu", "--workers", str(workers)])
    manifest.write_text(json.dumps(dict(policy="explore-v2")))
    train.main(["--resume", str(resume), "--out", str(split), "--device", "cpu", "--workers", str(workers)])
    a = torch.load(continuous.with_suffix(".resume.pt"), weights_only=True)
    b = torch.load(resume, weights_only=True)
    for key in ("model", "optimizer", "scheduler", "sampler", "cpu_rng", "loader_rng", "step"):
        assert_same(a[key], b[key])
    assert b["step"] == 8
    with pytest.raises(FileExistsError):
        train.main(common+["--out", str(split)])
