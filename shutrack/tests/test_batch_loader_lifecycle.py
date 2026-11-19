"""Batch prefetch thread lifecycle: stop, queue-full, exception and close order."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch

import shutrack.training.train as train_module
from shutrack.data.batch_loader import BatchIndexLoader
from shutrack.data.packed_dataset import PackedShutrackDataset
from shutrack.tests.conftest import write_tiny_packed_dataset


class _Sampler:
    def __init__(self, count: int) -> None:
        self.count = count
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.count)

    def __iter__(self):
        for index in range(self.count):
            yield [index]


class _Dataset:
    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after
        self.closed = False

    def batch(self, indices):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("producer boom")
        return {"x": torch.tensor([float(indices[0])])}


def test_early_exit_stops_producer_thread():
    loader = BatchIndexLoader(_Dataset(), _Sampler(1000), prefetch=2)
    iterator = iter(loader)
    next(iterator)
    thread = loader._thread
    assert thread is not None and thread.is_alive()
    iterator.close()
    assert not thread.is_alive()
    assert loader._thread is None


def test_queue_full_still_honours_stop():
    loader = BatchIndexLoader(_Dataset(), _Sampler(1000), prefetch=1)
    iterator = iter(loader)
    next(iterator)
    # Give the producer time to fill the single-slot queue and block.
    for _ in range(50):
        if loader.dataset.calls >= 3:
            break
        threading.Event().wait(0.01)
    iterator.close()
    assert loader._thread is None


def test_producer_exception_propagates_to_consumer():
    loader = BatchIndexLoader(_Dataset(fail_after=1), _Sampler(10), prefetch=1)
    iterator = iter(loader)
    next(iterator)
    with pytest.raises(RuntimeError, match="producer boom"):
        next(iterator)
    assert loader._thread is None


def _train_config(tmp_path: Path) -> dict:
    return {
        "dataset_dir": str(write_tiny_packed_dataset(tmp_path / "packed")),
        "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
        "device": "cpu",
        "epochs": 1,
        "batch_size": 2,
        "num_workers": 0,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 0,
        "warmup_steps": 1,
        "grad_clip": 1.0,
        "seed": 42,
        "max_samples": 4,
        "log_interval": 1,
        "competition_enabled": True,
        "d_model": 32,
        "n_heads": 4,
        "hce_layers": 2,
        "ff_dim": 64,
        "appearance_proj_dim": 16,
    }


def test_training_loop_exception_closes_loader_before_dataset(tmp_path, monkeypatch):
    config = _train_config(tmp_path)
    created: list[BatchIndexLoader] = []
    events: list[str] = []
    original_loader = train_module.BatchIndexLoader

    class RecordingLoader(original_loader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(train_module, "BatchIndexLoader", RecordingLoader)
    original_close = PackedShutrackDataset.close

    def close(self):
        for loader in created:
            thread = loader._thread
            assert thread is None or not thread.is_alive(), (
                "dataset closed while the prefetch producer was still running"
            )
        events.append("dataset_close")
        original_close(self)

    monkeypatch.setattr(PackedShutrackDataset, "close", close)

    def boom(outputs, batch, **kwargs):
        raise RuntimeError("training boom")

    monkeypatch.setattr(train_module, "gate_loss", boom)

    with pytest.raises(RuntimeError, match="training boom"):
        train_module.train_shutrack(config)

    assert created, "batch loader was not constructed"
    assert events == ["dataset_close"]
