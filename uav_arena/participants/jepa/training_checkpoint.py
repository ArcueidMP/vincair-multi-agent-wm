"""Committed batch position without changing the original RandomSampler order."""
import torch


class CheckpointBatches:
    def __init__(self, batch_sampler, state=None):
        self.batch_sampler = batch_sampler
        self.batches = state["batches"] if state else None
        self.cursor = state["cursor"] if state else 0
        self.epoch = state["epoch"] if state else 0

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        if self.batches is None:
            self.batches = list(self.batch_sampler)
        yield from self.batches[self.cursor:]

    def commit(self):
        self.cursor += 1

    def next_epoch(self):
        if self.cursor != len(self):
            raise ValueError("cannot advance an unfinished epoch")
        self.batches = None
        self.cursor = 0
        self.epoch += 1

    def state_dict(self):
        return dict(batches=self.batches, cursor=self.cursor, epoch=self.epoch)


def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
