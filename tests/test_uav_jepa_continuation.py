"""Continuation preserves optimizer, schedule, RNG and committed batch ordering."""
import hashlib
import json

import pytest
import torch
from torch import nn

from uav_arena.participants.jepa import continue_train as training


def test_prefetch_does_not_commit_and_epoch_resume():
    sampler = training.CommittedBatches(10, 3, 7)
    prefetched = list(sampler)
    assert sampler.cursor == 0
    sampler.commit()
    restored = training.CommittedBatches(10, 3, 7, **sampler.state_dict())
    assert list(restored) == prefetched[1:]
    restored.commit()
    restored.commit()
    restored.next_epoch()
    assert restored.epoch == 1 and restored.cursor == 0
    assert list(restored) != prefetched


class Tiny(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(2, 1)
        self.dropout = nn.Dropout(.2)

    def loss(self, batch, stats, steps):
        prediction = self.linear(self.dropout(batch["x"]))
        loss = (prediction-batch["y"]).square().mean()
        return loss, dict(head_absolute=loss.detach())


def test_split_run_exactly_matches_continuous_across_epochs(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "EntityJEPA", Tiny)
    monkeypatch.setattr(training, "load", lambda *a, **kw: [object()])
    dataset = [{"x": torch.tensor([float(i), .5]), "y": torch.tensor([i*.2])} for i in range(7)]
    monkeypatch.setattr(training, "Windows", lambda *a, **kw: dataset)
    data = tmp_path/"data"
    data.mkdir()
    manifest = data/"manifest.json"
    manifest.write_text(json.dumps(dict(policy="explore-v2")))
    config = dict(history=1)
    torch.manual_seed(3)
    parent = tmp_path/"parent.pt"
    torch.save(dict(config=config, model=Tiny(**config).state_dict(), stats={},
                    training_info=dict(step=10000)), parent)
    reference = tmp_path/"reference.json"
    reference.write_text(json.dumps(dict(config=config, training=dict(data=str(data), episodes=1,
        seed=0, lr=.001, history=1, batch=2, rollout_steps=1),
        dataset_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
    common = ["--checkpoint", str(parent), "--training-config", str(reference),
              "--steps", "8", "--device", "cpu", "--workers", "0"]
    continuous, split = tmp_path/"continuous.pt", tmp_path/"split.pt"
    training.main(common+["--out", str(continuous)])
    training.main(common+["--out", str(split), "--stop-after", "4"])
    assert not split.exists()
    resume = split.with_suffix(".resume.pt")
    paused = torch.load(resume, weights_only=True)
    assert paused["step"] == 4 and paused["sampler"] == dict(epoch=1, cursor=1)
    training.main(["--resume", str(resume), "--out", str(split), "--device", "cpu", "--workers", "0"])
    a = torch.load(continuous.with_suffix(".resume.pt"), weights_only=True)
    b = torch.load(resume, weights_only=True)
    def same(x, y):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                same(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for xx, yy in zip(x, y):
                same(xx, yy)
        else:
            assert x == y
    for key in ("model", "optimizer", "scheduler", "sampler", "cpu_rng", "step"):
        same(a[key], b[key])
    assert b["setup"]["parent_optimizer_restored"] is False
    assert b["setup"]["parent_step"] == 10000
    final = torch.load(split, weights_only=True)
    assert final["training_info"]["step"] == 8
    with pytest.raises(FileExistsError):
        training.main(common+["--out", str(split)])
