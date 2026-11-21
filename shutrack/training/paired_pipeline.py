"""Ordered CPU batch prefetch and deferred metrics for paired supervision.

The sampler owns all ordering. A single producer shares the existing arrays;
only the bounded queue holds additional batches. No CUDA compute runs there.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import struct
import zipfile
from typing import Callable, Iterable, Iterator, Mapping

import numpy as np
import torch

from shutrack.data.batch_loader import BatchIndexLoader


def npz_array_view(archive, key: str) -> np.ndarray:
    """Map an uncompressed NPY member without copying the entire rank matrix.

    Compressed/in-memory archives retain NumPy's normal loading path. The
    mapped view is read-only and keeps its mapping alive independently of NPZ.
    """
    member = archive.zip.getinfo(key + ".npy")
    filename = archive.zip.filename
    if member.compress_type != zipfile.ZIP_STORED or not isinstance(filename, (str, Path)):
        return np.asarray(archive[key])
    with open(filename, "rb") as stream:
        stream.seek(member.header_offset)
        header = stream.read(30)
        if len(header) != 30:
            raise ValueError(f"{key}: truncated ZIP local header")
        fields = struct.unpack("<4sHHHHHIIIHH", header)
        if fields[0] != b"PK\x03\x04":
            raise ValueError(f"{key}: invalid ZIP local header")
        start = member.header_offset + 30 + fields[-2] + fields[-1]
        stream.seek(start)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            return np.asarray(archive[key])
        offset = stream.tell()
    if dtype.hasobject:
        raise ValueError(f"{key}: object arrays are not supported")
    size = math.prod(shape) * dtype.itemsize
    if offset + size != start + member.file_size:
        raise ValueError(f"{key}: NPY payload does not match the ZIP member size")
    if not size:
        return np.empty(shape, dtype=dtype)
    return np.memmap(filename, mode="r", offset=offset, dtype=dtype, shape=shape,
                     order="F" if fortran_order else "C")


def prepare_cpu_batch(
    batch: Mapping, tensor_keys: Iterable[str], *, pin_memory: bool = False
) -> dict:
    keys = frozenset(tensor_keys)
    result = {}
    for key, value in batch.items():
        if key in keys:
            value = value if torch.is_tensor(value) else torch.from_numpy(value)
            if value.device.type != "cpu":
                raise ValueError("The batch producer must return CPU tensors")
            if pin_memory and not value.is_pinned():
                value = value.pin_memory()
        result[key] = value
    return result


def move_batch(
    batch: Mapping, device: torch.device, tensor_keys: Iterable[str], *,
    float32_keys: Iterable[str] = (),
) -> dict:
    device = torch.device(device)
    keys = frozenset(tensor_keys)
    float32_keys = frozenset(float32_keys)
    result = {}
    for key, value in batch.items():
        if key in keys:
            value = value if torch.is_tensor(value) else torch.from_numpy(value)
            value = value.to(
                device, non_blocking=device.type == "cuda" and value.is_pinned()
            )
            if key in float32_keys and value.dtype == torch.float16:
                value = value.float()
        result[key] = value
    return result


class _RowBatches:
    def __init__(self, rows: np.ndarray, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.rows = rows
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.rows) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for start in range(0, len(self.rows), self.batch_size):
            yield self.rows[start : start + self.batch_size]


class _PreparedBatches:
    def __init__(self, build_batch, tensor_keys, pin_memory):
        self.build_batch = build_batch
        self.tensor_keys = tensor_keys
        self.pin_memory = pin_memory

    def batch(self, rows):
        return rows, prepare_cpu_batch(
            self.build_batch(rows), self.tensor_keys, pin_memory=self.pin_memory
        )


@contextmanager
def prepared_batches(
    rows: np.ndarray,
    batch_size: int,
    build_batch: Callable[[np.ndarray], Mapping],
    tensor_keys: Iterable[str],
    *,
    prefetch: int = 1,
    pin_memory: bool = False,
) -> Iterator:
    """Yield batches in exactly the input row order and always join the producer."""
    if prefetch < 0:
        raise ValueError("prefetch cannot be negative")
    loader = BatchIndexLoader(
        _PreparedBatches(build_batch, tuple(tensor_keys), pin_memory),
        _RowBatches(rows, batch_size),
        prefetch=prefetch,
    )
    iterator = iter(loader)
    try:
        yield iterator
    finally:
        iterator.close()


class DeferredMetrics:
    """Accumulate detached scalar metrics on-device; copy once at epoch end."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self._sums: dict[str, torch.Tensor] = {}

    def add(self, name: str, value, weight: float = 1.0) -> None:
        scalar = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
        if scalar.numel() != 1:
            raise ValueError(f"{name}: expected a scalar metric")
        scalar = scalar.to(device=self.device, dtype=torch.float64).reshape(())
        weighted = scalar * weight
        if name in self._sums:
            self._sums[name].add_(weighted)
        else:
            self._sums[name] = weighted

    def summary(self) -> dict[str, float]:
        if not self._sums:
            return {}
        names = list(self._sums)
        values = torch.stack([self._sums[name] for name in names]).cpu().tolist()
        return dict(zip(names, values))
