"""Ordering, shutdown and gradient checks for the paired training pipeline."""

import threading

import numpy as np
import pytest
import torch

from shutrack.training.loss import appearance_column_loss
from shutrack.training.paired_pipeline import (
    DeferredMetrics,
    move_batch,
    npz_array_view,
    prepared_batches,
)


@pytest.mark.parametrize("fortran", [False, True])
def test_uncompressed_npz_rank_view_is_readonly_and_exact(tmp_path, fortran):
    path = tmp_path / "rank.npz"
    array = np.arange(60, dtype=np.float16).reshape(10, 6)
    if fortran:
        array = np.asfortranarray(array)
    np.savez(path, rank=array)
    with np.load(path, allow_pickle=False) as archive:
        view = npz_array_view(archive, "rank")
        assert isinstance(view, np.memmap)
        assert not view.flags.writeable
        assert np.array_equal(view[[8, 1, 5]], array[[8, 1, 5]])
    assert np.array_equal(view, array)  # NPZ close must not invalidate the mmap


def test_compressed_and_empty_npz_arrays_keep_correct_values(tmp_path):
    path = tmp_path / "compressed.npz"
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez_compressed(path, rank=array)
    with np.load(path, allow_pickle=False) as archive:
        assert np.array_equal(npz_array_view(archive, "rank"), array)
    np.savez(path, rank=np.empty((0, 8), dtype=np.float16))
    with np.load(path, allow_pickle=False) as archive:
        assert npz_array_view(archive, "rank").shape == (0, 8)


def test_npz_object_array_is_rejected(tmp_path):
    path = tmp_path / "object.npz"
    np.savez(path, rank=np.array([object()], dtype=object))
    with np.load(path, allow_pickle=False) as archive:
        with pytest.raises(ValueError, match="object"):
            npz_array_view(archive, "rank")


@pytest.mark.parametrize("prefetch", [0, 1])
def test_batch_order_values_and_tail(prefetch):
    rows = np.array([8, 2, 9, 0, 6, 1, 7], dtype=np.int64)
    source = np.arange(60, dtype=np.float32).reshape(10, 6)
    built = []

    def build(indices):
        built.append(indices.copy())
        return {"x": source[indices], "mask": indices % 2 == 0, "metadata": "kept"}

    actual = []
    with prepared_batches(rows, 3, build, {"x", "mask"}, prefetch=prefetch) as batches:
        for indices, batch in batches:
            batch = move_batch(batch, torch.device("cpu"), {"x", "mask"})
            assert batch["metadata"] == "kept"
            torch.testing.assert_close(batch["x"], torch.from_numpy(source[indices]), rtol=0, atol=0)
            assert batch["mask"].dtype == torch.bool
            actual.extend(indices.tolist())
    assert actual == rows.tolist()
    assert [len(x) for x in built] == [3, 3, 1]


def test_producer_is_joined_when_consumer_fails_with_full_queue():
    threads = []
    calls = []

    def build(indices):
        threads.append(threading.current_thread())
        calls.append(indices.copy())
        return {"x": indices.copy()}

    with pytest.raises(RuntimeError, match="consumer failed"):
        with prepared_batches(np.arange(100), 1, build, {"x"}, prefetch=1) as batches:
            next(batches)
            raise RuntimeError("consumer failed")
    assert threads
    assert all(not thread.is_alive() for thread in threads)
    assert len(calls) <= 3  # consumed + queued + producer's in-flight batch


def test_producer_error_reaches_consumer_and_thread_is_joined():
    threads = []

    def build(indices):
        threads.append(threading.current_thread())
        if indices[0] == 2:
            raise ValueError("producer failed")
        return {"x": indices.copy()}

    with pytest.raises(ValueError, match="producer failed"):
        with prepared_batches(np.arange(6), 2, build, {"x"}, prefetch=1) as batches:
            list(batches)
    assert all(not thread.is_alive() for thread in threads)


@pytest.mark.parametrize("prefetch", [0, 1])
def test_empty_input_does_not_build_batch(prefetch):
    def build(_):
        raise AssertionError("empty input must not be read")

    with prepared_batches(np.array([], dtype=np.int64), 4, build, {"x"},
                          prefetch=prefetch) as batches:
        assert list(batches) == []


def test_invalid_loader_parameters_are_rejected():
    with pytest.raises(ValueError, match="prefetch"):
        with prepared_batches(np.arange(2), 1, lambda x: {}, (), prefetch=-1):
            pass
    with pytest.raises(ValueError, match="batch_size"):
        with prepared_batches(np.arange(2), 0, lambda x: {}, ()):
            pass


def test_half_transport_restores_float32_without_rounding_other_inputs():
    half = np.array([[0.125, -0.0625], [0.0, 1.0]], dtype=np.float16)
    untouched = np.array([0.123456789], dtype=np.float32)
    batch = move_batch({"reid": half, "evidence": untouched}, torch.device("cpu"),
                       {"reid", "evidence"}, float32_keys={"reid"})
    assert batch["reid"].dtype == torch.float32
    torch.testing.assert_close(batch["reid"], torch.from_numpy(half.astype(np.float32)),
                               rtol=0, atol=0)
    torch.testing.assert_close(batch["evidence"], torch.from_numpy(untouched),
                               rtol=0, atol=0)


def test_deferred_metrics_match_weighted_python_sum_and_detach_graph():
    metrics = DeferredMetrics(torch.device("cpu"))
    assert metrics.summary() == {}
    expected = 0.0
    for value, count in [(0.125, 4), (0.375, 3), (0.75, 1)]:
        scalar = torch.tensor(value, requires_grad=True)
        metrics.add("loss", scalar, count)
        metrics.add("count", torch.tensor(count))
        expected += float(scalar.detach()) * count
    assert metrics.summary() == {"loss": expected, "count": 8.0}
    assert all(not value.requires_grad and value.grad_fn is None
               for value in metrics._sums.values())
    with pytest.raises(ValueError, match="scalar"):
        metrics.add("bad", torch.zeros(2))


def test_column_deferred_statistics_leave_loss_and_gradient_unchanged():
    generator = torch.Generator().manual_seed(42)
    batch = {
        "current_track": torch.randn(5, 8, generator=generator),
        "current_detection": torch.randn(5, 8, generator=generator),
        "column_positive": torch.nn.functional.normalize(
            torch.randn(5, 8, generator=generator), dim=-1),
        "column_score": torch.tensor([0.7, 0.8, 0.9, 0.6, 1.0]),
        "column_offset": torch.tensor([0.1, 0.2, 0.1, 0.2, 0.0]),
        "column_rival": torch.tensor([0.2, 0.4, 0.5, 0.1, 0.3]),
        "column_valid": torch.tensor([True, False, True, True, True]),
    }
    logits = torch.randn(5, 2, generator=generator, requires_grad=True)
    eager, deferred = {}, {}
    old_loss = appearance_column_loss({"gate_logits": logits}, batch, components=eager)
    new_loss = appearance_column_loss({"gate_logits": logits}, batch, components=deferred,
                                      tensor_statistics=True)
    old_gradient = torch.autograd.grad(old_loss, logits, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(new_loss, logits)[0]
    torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
    torch.testing.assert_close(old_gradient, new_gradient, rtol=0, atol=0)
    assert isinstance(eager["valid"], int)
    for key in ("valid", "active"):
        assert torch.is_tensor(deferred[key])
        assert int(deferred[key]) == eager[key]
        assert not deferred[key].requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_pinned_batches_are_transferred_without_value_changes():
    source = np.arange(24, dtype=np.float32).reshape(6, 4)
    with prepared_batches(np.arange(6), 2, lambda rows: {"x": source[rows]},
                          {"x"}, prefetch=1, pin_memory=True) as batches:
        for rows, batch in batches:
            assert batch["x"].is_pinned()
            moved = move_batch(batch, torch.device("cuda"), {"x"})
            torch.testing.assert_close(moved["x"].cpu(), torch.from_numpy(source[rows]),
                                       rtol=0, atol=0)
