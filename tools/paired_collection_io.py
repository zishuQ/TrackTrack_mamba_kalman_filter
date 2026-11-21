"""Bounded writes and header-only completeness checks for paired collection."""

from __future__ import annotations

import json
import fcntl
from contextlib import contextmanager
import math
import mmap
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

BLOCK_BYTES = 16 * 1024 * 1024
SHARD_FILES = ("samples.npz", "event_evidence.npy", "event_track.npy",
               "event_detection_index.npy", "event_has_detection.npy")


@contextmanager
def collection_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".collection.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Collection is already active for {directory.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def drop_pages(array, *, flush=False):
    seen = set()
    while array is not None and id(array) not in seen:
        seen.add(id(array))
        mapping = getattr(array, "_mmap", None)
        if mapping is None and isinstance(array, mmap.mmap):
            mapping = array
        if mapping is not None:
            try:
                if flush:
                    mapping.flush()
                mapping.madvise(mmap.MADV_DONTNEED)
            except (BufferError, ValueError, OSError):
                pass
            return
        array = getattr(array, "base", None)


def close_mapping(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        try:
            array.flush()
        finally:
            mapping.close()


def block_rows(width, dtype, budget=BLOCK_BYTES):
    return max(1, int(budget) // max(1, 2 * int(width) * np.dtype(dtype).itemsize))


def atomic_npz(path, arrays, *, guard=None, budget=BLOCK_BYTES):
    """Write stored NPY members in bounded chunks; no full-array byte copies."""
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            for name, source in arrays.items():
                array = np.asanyarray(source)
                if array.dtype.hasobject:
                    raise ValueError(f"Object arrays cannot be collected: {name}")
                fortran = bool(array.flags.f_contiguous and not array.flags.c_contiguous)
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array_header_2_0(member, {
                        "descr": np.lib.format.dtype_to_descr(array.dtype),
                        "fortran_order": fortran, "shape": array.shape,
                    })
                    iterator = np.nditer(
                        array, flags=["external_loop", "buffered", "zerosize_ok"],
                        op_flags=["readonly"], order="F" if fortran else "C",
                        buffersize=max(1, budget // max(1, array.dtype.itemsize)),
                    )
                    for chunk in iterator:
                        member.write(chunk.tobytes(order="C"))
                        drop_pages(source)
                        if guard is not None:
                            guard()
                del array
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_track_copy(path, source, count, *, guard=None):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    destination = None
    try:
        destination = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=source.dtype,
            shape=(int(count), int(source.shape[1])),
        )
        step = block_rows(source.shape[1], source.dtype)
        for start in range(0, count, step):
            end = min(count, start + step)
            destination[start:end] = source[start:end]
            drop_pages(destination, flush=True)
            drop_pages(source, flush=True)
            if guard is not None:
                guard()
        close_mapping(destination)
        os.replace(temporary, path)
    finally:
        if destination is not None:
            close_mapping(destination)
        temporary.unlink(missing_ok=True)


def npy_header(handle):
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"Unsupported NPY header version: {version}")
    if dtype.hasobject:
        raise ValueError("Object arrays are not accepted")
    return tuple(shape), dtype, handle.tell()


def inspect_shard(directory, protocol=None, *, repair_counts=False):
    """Check headers and payload lengths only, never load large ReID arrays.

    This is a collection-completeness gate, not the full label validator.
    Count repair is permitted only for already complete, structurally sound
    shards. It does not change any numerical array.
    """
    directory = Path(directory)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    if not summary.get("complete"):
        raise ValueError(f"{directory.name}: incomplete summary")
    if protocol is not None and summary.get("protocol") != protocol:
        raise ValueError(f"{directory.name}: incompatible replay protocol")
    for name in SHARD_FILES:
        if not (directory / name).is_file():
            raise ValueError(f"{directory.name}: missing {name}")
    shapes = {}
    with zipfile.ZipFile(directory / "samples.npz") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            with archive.open(info) as member:
                shape, dtype, offset = npy_header(member)
            expected = offset + math.prod(shape) * dtype.itemsize
            if info.file_size != expected:
                raise ValueError(f"{directory.name}: truncated or mis-sized {info.filename}")
            shapes[info.filename.removesuffix(".npy")] = shape
    if not shapes.get("frame") or shapes["frame"][0] <= 0:
        raise ValueError(f"{directory.name}: zero/missing sample rows")
    samples = shapes["frame"][0]
    required = ("track_id", "detection_index", "gt_id", "window_indices",
                "current_event_index", "current_competition", "rank_valid",
                "motion_gt_valid", "pred_box_g0", "pred_box_g1", "gt_box_t1")
    for name in required:
        if name not in shapes or not shapes[name] or shapes[name][0] != samples:
            raise ValueError(f"{directory.name}: sample row mismatch for {name}")
    replay = summary.get("protocol") or {}
    if replay.get("rank_arrays"):
        rank_count = shapes.get("rank_rows", (-1,))[0]
        for name in ("rank_template0", "rank_detection", "rank_positive",
                     "rank_competitor", "rank_score"):
            if name not in shapes or shapes[name][0] != rank_count:
                raise ValueError(f"{directory.name}: rank row mismatch for {name}")
    if replay.get("column"):
        for name in ("column_det", "column_offset", "column_rival", "column_score"):
            if name not in shapes or shapes[name][0] != samples:
                raise ValueError(f"{directory.name}: column row mismatch for {name}")
    events = None
    for name in SHARD_FILES[1:]:
        with (directory / name).open("rb") as handle:
            shape, dtype, offset = npy_header(handle)
        expected = offset + math.prod(shape) * dtype.itemsize
        if (directory / name).stat().st_size != expected or not shape:
            raise ValueError(f"{directory.name}: mis-sized {name}")
        if events is None:
            events = shape[0]
        if shape[0] != events:
            raise ValueError(f"{directory.name}: event row mismatch for {name}")
    changed = summary.get("samples") != samples or summary.get("events") != events
    if changed and not repair_counts:
        raise ValueError(f"{directory.name}: summary counts disagree with array headers")
    if changed:
        backup = summary_path.with_name(
            f"summary.before-count-fix.{time.time_ns()}.json"
        )
        shutil.copy2(summary_path, backup)
        summary.update(samples=int(samples), events=int(events))
        atomic_json(summary_path, summary)
    return {"samples": int(samples), "events": int(events), "repaired": changed}



def stream_nonfinite_rows(path, name, mask=None, *, budget=BLOCK_BYTES):
    """Count rows with NaN/Inf without loading a full NPZ member.

    Both C and Fortran NPY layouts are handled. The working buffer is byte
    bounded; the Fortran path additionally keeps one boolean per row.
    """
    with zipfile.ZipFile(path) as archive, archive.open(name + ".npy") as member:
        shape, dtype, offset = npy_header(member)
        if not shape:
            raise ValueError(f"{name}: expected a row array")
        rows = int(shape[0])
        width = math.prod(shape[1:])
        selected = None if mask is None else np.asarray(mask, dtype=bool)
        if selected is not None and selected.shape != (rows,):
            raise ValueError(f"{name}: mask length mismatch")
        # Re-read the header's layout flag (npy_header intentionally returns
        # only dimensions and dtype for the structural-completeness checker).
        member.seek(0)
        version = np.lib.format.read_magic(member)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        _, fortran, _ = reader(member)
        remaining = rows * width
        if not remaining:
            return 0
        if fortran and width > 1:
            bad_rows = np.zeros(rows, dtype=bool)
            step = max(1, budget // dtype.itemsize)
            position = 0
            while remaining:
                size = min(remaining, step)
                raw = member.read(size * dtype.itemsize)
                if len(raw) != size * dtype.itemsize:
                    raise ValueError(f"{name}: truncated NPY payload")
                values = np.frombuffer(raw, dtype=dtype)
                bad = np.flatnonzero(~np.isfinite(values))
                bad_rows[(position + bad) % rows] = True
                position += size
                remaining -= size
            return int(np.count_nonzero(bad_rows if selected is None else bad_rows & selected))
        row_bytes = max(1, width * dtype.itemsize)
        step = max(1, budget // row_bytes)
        bad_count = 0
        for start in range(0, rows, step):
            end = min(rows, start + step)
            size = (end - start) * row_bytes
            raw = member.read(size)
            if len(raw) != size:
                raise ValueError(f"{name}: truncated NPY payload")
            values = np.frombuffer(raw, dtype=dtype).reshape(end - start, width)
            bad = (~np.isfinite(values)).any(axis=1)
            if selected is not None:
                bad &= selected[start:end]
            bad_count += int(np.count_nonzero(bad))
        return bad_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inspect completed collection headers / repair counts")
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--repair-counts", action="store_true")
    args = parser.parse_args()
    results = {}
    failed = False
    for directory in args.directories:
        try:
            results[directory] = inspect_shard(directory, repair_counts=args.repair_counts)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            failed = True
            results[directory] = {"error": str(exc)}
    print(json.dumps(results, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
