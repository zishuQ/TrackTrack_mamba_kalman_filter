"""Stage only one sequence's selected feature rows, preserving source precision.

Disk reads follow sorted unique source row IDs. Sampling and batch order remain
owned by the trainer. The context restores the mmap sources after joining the
batch producer, so private feature copies live for one sequence only.
"""
from contextlib import contextmanager
import mmap
import os
import time

import numpy as np


def aligned_stage_chunk_samples(chunk_samples, batch_size):
    """Allow a partial batch only at the end of the sequence."""
    if chunk_samples <= 0 or batch_size <= 0:
        raise ValueError("stage chunk size and batch size must be positive")
    return max(1, chunk_samples // batch_size) * batch_size


def feature_chunks(rows, aligned_chunk_samples, *, enabled):
    if not enabled or aligned_chunk_samples is None:
        yield rows
        return
    for start in range(0, len(rows), aligned_chunk_samples):
        yield rows[start : start + aligned_chunk_samples]


def _root_mapping(array):
    root = None
    seen = set()
    while array is not None and id(array) not in seen:
        seen.add(id(array))
        if isinstance(array, np.memmap):
            root = array
        array = getattr(array, "base", None)
    return root


def release_mapped_pages(array, *, discard_cache=False):
    """Release file pages only; all original file bytes stay unchanged."""
    mapped = _root_mapping(array)
    if mapped is None:
        return
    try:
        mapped._mmap.madvise(mmap.MADV_DONTNEED)
        if discard_cache and hasattr(os, "posix_fadvise"):
            fd = os.open(mapped.filename, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, int(mapped.offset), mapped.nbytes,
                                 os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
    except (BufferError, ValueError, OSError):
        pass


class SelectedFeatureRows:
    """A compact immutable copy addressed by the original global row indices."""

    def __init__(self, source, indices, block_bytes=16 * 1024 * 1024):
        indices = np.unique(np.asarray(indices, dtype=np.int64))
        self.shape = source.shape
        self.dtype = source.dtype
        self.lookup = np.full(source.shape[0], -1, dtype=np.int64)
        self.lookup[indices] = np.arange(indices.size)
        self.values = np.empty((indices.size,) + source.shape[1:], dtype=source.dtype)
        row_bytes = max(1, int(np.prod(source.shape[1:])) * source.dtype.itemsize)
        block_rows = max(1, block_bytes // row_bytes)
        mapped = _root_mapping(source)
        if mapped is not None and source.flags.c_contiguous:
            # Bulk file reads avoid a page fault for every randomly gathered
            # mmap row (especially expensive for WSL file-backed mappings).
            offset = int(mapped.offset) + (source.ctypes.data - mapped.ctypes.data)
            width = max(1, int(np.prod(source.shape[1:])))
            with open(mapped.filename, "rb", buffering=0) as stream:
                start = 0
                while start < len(indices):
                    first = int(indices[start])
                    end = int(np.searchsorted(indices, first + block_rows, side="left"))
                    last = int(indices[end - 1]) + 1
                    if last - first > 4 * (end - start):
                        self.values[start:end] = source[indices[start:end]]
                    else:
                        stream.seek(offset + first * row_bytes)
                        block = np.fromfile(stream, dtype=source.dtype,
                                            count=(last - first) * width)
                        block = block.reshape((last - first,) + source.shape[1:])
                        self.values[start:end] = block[indices[start:end] - first]
                    start = end
        else:
            for start in range(0, len(indices), block_rows):
                end = min(start + block_rows, len(indices))
                self.values[start:end] = source[indices[start:end]]
        self.values.flags.writeable = False
        # Drop this mapping from RSS while keeping reusable file pages cached.
        release_mapped_pages(source)

    @property
    def nbytes(self):
        return self.values.nbytes + self.lookup.nbytes

    def __getitem__(self, indices):
        local = self.lookup[indices]
        if np.any(local < 0):
            raise IndexError("Feature row was not included in the staged sample plan")
        return self.values[local]


def release_sequence_pages(sequence, *, discard_cache=False):
    release_mapped_pages(sequence.event_track, discard_cache=discard_cache)
    cache = getattr(sequence, "cache", None)
    if cache is not None:
        release_mapped_pages(cache.features, discard_cache=discard_cache)
    for array in getattr(sequence, "_rank_arrays", ()) or ():
        release_mapped_pages(array, discard_cache=discard_cache)


@contextmanager
def staged_sequence_rows(sequence, rows, group, *, enabled=True):
    """Serve exactly the existing sample plan from bounded private RAM."""
    if not enabled or len(rows) == 0:
        yield {"seconds": 0.0, "bytes": 0}
        return
    started = time.perf_counter()
    history = sequence.small["window_indices"][rows]
    current = sequence.small["current_event_index"][rows]
    events = np.concatenate((history.reshape(-1), current))
    # batch_tensors reads row zero before masking padded history entries.
    events = np.unique(np.maximum(events, 0))
    detection_ids = np.asarray(sequence.event_det[events], dtype=np.int64)
    if group == "appearance_row_column":
        detection_ids = np.concatenate((detection_ids, sequence.small["column_det"][rows]))
    detection_ids = np.unique(detection_ids[detection_ids >= 0])
    original_track = sequence.event_track
    original_detection = sequence.cache.features
    original_rank = None
    rank = None
    if group in ("appearance_rank", "appearance_row_column", "appearance_rank_speed4"):
        sequence.load_rank_arrays()
        original_rank = sequence._rank_arrays
        rank_ids = sequence.rank_pos[rows]
        rank_ids = np.unique(rank_ids[rank_ids >= 0])
        rank = tuple(SelectedFeatureRows(array, rank_ids) for array in original_rank)
    track = SelectedFeatureRows(original_track, events)
    detection = SelectedFeatureRows(original_detection, detection_ids)
    size = track.nbytes + detection.nbytes + sum(array.nbytes for array in rank or ())
    sequence.event_track = track
    sequence.cache.features = detection
    if rank is not None:
        sequence._rank_arrays = rank
    try:
        yield {"seconds": time.perf_counter() - started, "bytes": size,
               "event_rows": len(events), "detection_rows": len(detection_ids)}
    finally:
        sequence.event_track = original_track
        sequence.cache.features = original_detection
        if original_rank is not None:
            sequence._rank_arrays = original_rank
        release_sequence_pages(sequence)
