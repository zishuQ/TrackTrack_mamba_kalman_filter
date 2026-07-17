from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np


MAMBA_SHADOW_SCHEMA_VERSION = 1
MAMBA_SHADOW_SCHEMA_DESCRIPTOR = {
    "name": "agentguard_mamba_shadow_teacher",
    "version": MAMBA_SHADOW_SCHEMA_VERSION,
    "driver": "nsa_track_lifecycle_and_accepted_associations",
    "state": "cxcywh_vxvyvwvh",
    "box": "pixel_xyxy",
    "gmc": "nsa_effective_warp_before_predict",
    "association": "nsa_only",
}
MAMBA_SHADOW_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        MAMBA_SHADOW_SCHEMA_DESCRIPTOR,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

TEACHER_ARRAY_DTYPES = {
    "event_id": np.dtype("U96"),
    "teacher_config_sha256": np.dtype("U64"),
    "frame_id": np.dtype(np.int32),
    "track_id": np.dtype(np.int64),
    "matched": np.dtype(np.bool_),
    "gap": np.dtype(np.int32),
    "accepted_detection_index": np.dtype(np.int64),
    "mamba_prior_mean": np.dtype(np.float32),
    "mamba_prior_covariance": np.dtype(np.float32),
    "mamba_prior_box": np.dtype(np.float32),
    "mamba_posterior_mean": np.dtype(np.float32),
    "mamba_posterior_covariance": np.dtype(np.float32),
    "mamba_posterior_box": np.dtype(np.float32),
    "nsa_prior_box": np.dtype(np.float32),
    "nsa_full_update_box": np.dtype(np.float32),
    "accepted_detection_box": np.dtype(np.float32),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def mean_to_xyxy(mean: np.ndarray) -> np.ndarray:
    value = np.asarray(mean, dtype=np.float64).reshape(8)
    cx, cy, width, height = value[:4]
    return np.asarray(
        [
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ],
        dtype=np.float64,
    )


def xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float64).reshape(4)
    return np.asarray(
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1],
        dtype=np.float64,
    )


def apply_warp_to_state(
    mean: np.ndarray,
    covariance: np.ndarray,
    warp_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    warp = np.asarray(warp_matrix, dtype=np.float64).reshape(2, 3)
    rotation = warp[:, :2]
    rotation_8 = np.kron(np.eye(4, dtype=np.float64), rotation)
    warped_mean = rotation_8 @ np.asarray(mean, dtype=np.float64).reshape(8)
    warped_mean[:2] += warp[:, 2]
    covariance = np.asarray(covariance, dtype=np.float64).reshape(8, 8)
    warped_covariance = rotation_8 @ covariance @ rotation_8.T
    return warped_mean, warped_covariance


def _finite(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass
class _ShadowState:
    mean: np.ndarray
    covariance: np.ndarray
    last_observation: np.ndarray
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    prior_box: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    posterior_box: np.ndarray
    missing_age: int = 0
    event_gap: int = 0


class MambaShadowWriter:
    """Bounded-memory NPZ shard writer for a single sequence."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        sequence: str,
        teacher_config: dict[str, Any],
        event_cache_manifest_sha256: str,
        detection_cache_manifest_sha256: str,
        shard_size: int = 2048,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.sequence = str(sequence)
        self.teacher_config = dict(teacher_config)
        self.teacher_config_sha256 = canonical_sha256(self.teacher_config)
        self.event_cache_manifest_sha256 = str(event_cache_manifest_sha256)
        self.detection_cache_manifest_sha256 = str(
            detection_cache_manifest_sha256
        )
        self.shard_size = max(int(shard_size), 1)
        self._records: list[dict[str, Any]] = []
        self._shard_id = 0
        self._num_records = 0
        self._num_matched = 0
        self._num_unmatched = 0
        self._file_hashes: dict[str, str] = {}
        self._closed = False
        self._temporary = self.output_dir.with_name(
            f"{self.output_dir.name}.incomplete.{os.getpid()}"
        )
        if self.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite shadow teacher: {self.output_dir}")
        if self._temporary.exists():
            raise FileExistsError(f"stale shadow teacher directory: {self._temporary}")
        self._temporary.mkdir(parents=True)
        self._write_manifest(complete=False)

    def append(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed Mamba shadow writer")
        normalized = self._normalize_record(record)
        self._records.append(normalized)
        self._num_records += 1
        self._num_matched += int(normalized["matched"])
        self._num_unmatched += int(not normalized["matched"])
        if len(self._records) >= self.shard_size:
            self._flush()

    @property
    def num_records(self) -> int:
        return self._num_records

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        matched = bool(record["matched"])
        normalized = {
            "event_id": str(record["event_id"]),
            "teacher_config_sha256": self.teacher_config_sha256,
            "frame_id": int(record["frame_id"]),
            "track_id": int(record["track_id"]),
            "matched": matched,
            "gap": int(record["gap"]),
            "accepted_detection_index": int(
                record.get("accepted_detection_index", -1)
            ),
        }
        shapes = {
            "mamba_prior_mean": (8,),
            "mamba_prior_covariance": (8, 8),
            "mamba_prior_box": (4,),
            "mamba_posterior_mean": (8,),
            "mamba_posterior_covariance": (8, 8),
            "mamba_posterior_box": (4,),
            "nsa_prior_box": (4,),
            "nsa_full_update_box": (4,),
            "accepted_detection_box": (4,),
        }
        for name, shape in shapes.items():
            normalized[name] = _finite(name, record[name], shape).astype(
                np.float32, copy=False
            )
        if normalized["gap"] < 0:
            raise ValueError("shadow gap must be non-negative")
        if matched != (normalized["accepted_detection_index"] >= 0):
            raise ValueError("matched shadow record has inconsistent detection index")
        return normalized

    def _flush(self) -> None:
        if not self._records:
            return
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in TEACHER_ARRAY_DTYPES.items():
            values = [record[name] for record in self._records]
            arrays[name] = np.asarray(values, dtype=dtype)
        path = self._temporary / f"teacher_{self._shard_id:05d}.npz"
        np.savez(path, **arrays)
        self._file_hashes[path.name] = sha256_file(path)
        self._records.clear()
        self._shard_id += 1

    def _write_manifest(self, *, complete: bool) -> None:
        manifest = {
            "complete": bool(complete),
            "schema_version": MAMBA_SHADOW_SCHEMA_VERSION,
            "schema_sha256": MAMBA_SHADOW_SCHEMA_SHA256,
            "sequence": self.sequence,
            "num_records": self._num_records,
            "num_matched": self._num_matched,
            "num_unmatched": self._num_unmatched,
            "num_shards": self._shard_id,
            "shard_size": self.shard_size,
            "teacher_config": self.teacher_config,
            "teacher_config_sha256": self.teacher_config_sha256,
            "event_cache_manifest_sha256": self.event_cache_manifest_sha256,
            "detection_cache_manifest_sha256": (
                self.detection_cache_manifest_sha256
            ),
            "array_dtypes": {
                key: str(value) for key, value in TEACHER_ARRAY_DTYPES.items()
            },
            "shard_sha256": self._file_hashes,
        }
        (self._temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    def close(self) -> dict[str, Any]:
        if self._closed:
            return json.loads((self.output_dir / "manifest.json").read_text())
        self._flush()
        self._write_manifest(complete=True)
        self._temporary.rename(self.output_dir)
        self._closed = True
        return json.loads((self.output_dir / "manifest.json").read_text())

    def abort(self) -> None:
        if not self._closed and self._temporary.exists():
            shutil.rmtree(self._temporary)
        self._closed = True


class MambaShadowReader:
    def __init__(self, sequence_dir: str | Path) -> None:
        self.sequence_dir = Path(sequence_dir).resolve()
        self.manifest = json.loads(
            (self.sequence_dir / "manifest.json").read_text()
        )
        required = {
            "complete": True,
            "schema_version": MAMBA_SHADOW_SCHEMA_VERSION,
            "schema_sha256": MAMBA_SHADOW_SCHEMA_SHA256,
        }
        mismatches = {
            key: (self.manifest.get(key), expected)
            for key, expected in required.items()
            if self.manifest.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Mamba shadow manifest mismatch: {mismatches}")
        self.paths = sorted(self.sequence_dir.glob("teacher_*.npz"))
        if len(self.paths) != int(self.manifest["num_shards"]):
            raise ValueError("Mamba shadow shard count does not match manifest")

    def iter_shards(self, *, verify_hashes: bool = True) -> Iterator[dict[str, np.ndarray]]:
        hashes = self.manifest.get("shard_sha256", {})
        for path in self.paths:
            if verify_hashes and sha256_file(path) != hashes.get(path.name):
                raise ValueError(f"Mamba shadow shard hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as payload:
                arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
            lengths = {len(value) for value in arrays.values()}
            if len(lengths) != 1:
                raise ValueError(f"Mamba shadow shard arrays have unequal lengths: {path}")
            yield arrays

    def iter_records(self, *, verify_hashes: bool = True) -> Iterator[dict[str, Any]]:
        for arrays in self.iter_shards(verify_hashes=verify_hashes):
            for index in range(len(arrays["event_id"])):
                yield {name: value[index] for name, value in arrays.items()}


class MambaShadow:
    """Mamba KF driven exclusively by NSA track IDs and accepted matches."""

    def __init__(
        self,
        *,
        sequence: str,
        image_width: int,
        image_height: int,
        checkpoint_path: str | Path,
        writer: MambaShadowWriter,
        device: str = "cuda",
        filter_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.sequence = str(sequence)
        self.writer = writer
        self.states: dict[int, _ShadowState] = {}
        self._closed = False
        if filter_factory is None:
            from mamba_kalman_filter.config import Config

            Config.apply_mot20_exp31_1_profile(log_name="exp31_1_shadow")
            from trackers.mamba_kalman_filter_wrapper import (
                MambaKalmanFilterWrapper,
            )

            filter_factory = lambda path, selected_device: MambaKalmanFilterWrapper(
                model_path=path, device=selected_device
            )
        self.filter = filter_factory(str(checkpoint_path), str(device))
        self.filter.set_image_size(int(image_width), int(image_height))

    def initiate_track(self, track_id: int, measurement: np.ndarray) -> None:
        track_id = int(track_id)
        if track_id in self.states:
            raise ValueError(f"Mamba shadow track already exists: {track_id}")
        measurement = _finite("initial measurement", measurement, (4,))
        mean, covariance = self.filter.initiate(measurement.astype(np.float32))
        mean = _finite("initial Mamba mean", np.asarray(mean).reshape(8), (8,))
        covariance = _finite(
            "initial Mamba covariance", np.asarray(covariance).reshape(8, 8), (8, 8)
        )
        box = mean_to_xyxy(mean)
        self.states[track_id] = _ShadowState(
            mean=mean,
            covariance=covariance,
            last_observation=measurement.copy(),
            prior_mean=mean.copy(),
            prior_covariance=covariance.copy(),
            prior_box=box.copy(),
            posterior_mean=mean.copy(),
            posterior_covariance=covariance.copy(),
            posterior_box=box.copy(),
        )

    def begin_frame(
        self,
        active_track_ids: Iterable[int],
        warp_matrix: np.ndarray,
    ) -> None:
        track_ids = [int(value) for value in active_track_ids]
        missing = [track_id for track_id in track_ids if track_id not in self.states]
        if missing:
            raise KeyError(f"Mamba shadow missing active NSA tracks: {missing[:8]}")
        if not track_ids:
            return
        for track_id in track_ids:
            state = self.states[track_id]
            state.mean, state.covariance = apply_warp_to_state(
                state.mean, state.covariance, warp_matrix
            )
        means = np.stack([self.states[track_id].mean for track_id in track_ids])
        covariances = np.stack(
            [self.states[track_id].covariance for track_id in track_ids]
        )
        measurements = np.stack(
            [self.states[track_id].last_observation for track_id in track_ids]
        )
        predicted_mean, predicted_covariance = self.filter.batch_predict(
            means.astype(np.float32),
            covariances.astype(np.float32),
            measurements.astype(np.float32),
            track_ids,
        )
        predicted_mean = np.asarray(predicted_mean, dtype=np.float64)
        predicted_covariance = np.asarray(predicted_covariance, dtype=np.float64)
        _finite("batched Mamba predicted mean", predicted_mean, (len(track_ids), 8))
        _finite(
            "batched Mamba predicted covariance",
            predicted_covariance,
            (len(track_ids), 8, 8),
        )
        for index, track_id in enumerate(track_ids):
            state = self.states[track_id]
            state.mean = predicted_mean[index].copy()
            state.covariance = predicted_covariance[index].copy()
            state.prior_mean = state.mean.copy()
            state.prior_covariance = state.covariance.copy()
            state.prior_box = mean_to_xyxy(state.mean)
            state.posterior_mean = state.mean.copy()
            state.posterior_covariance = state.covariance.copy()
            state.posterior_box = state.prior_box.copy()
            state.missing_age += 1
            state.event_gap = state.missing_age

    def update_matches(
        self,
        matches: Iterable[tuple[int, np.ndarray]],
    ) -> None:
        items = [(int(track_id), np.asarray(measurement)) for track_id, measurement in matches]
        if not items:
            return
        track_ids = [item[0] for item in items]
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("duplicate NSA track in Mamba shadow update batch")
        missing = [track_id for track_id in track_ids if track_id not in self.states]
        if missing:
            raise KeyError(f"Mamba shadow update for unknown NSA tracks: {missing}")
        means = np.stack([self.states[track_id].mean for track_id in track_ids])
        covariances = np.stack(
            [self.states[track_id].covariance for track_id in track_ids]
        )
        measurements = np.stack(
            [_finite("matched measurement", value, (4,)) for _, value in items]
        )
        posterior_mean, posterior_covariance = self.filter.batch_update(
            means.astype(np.float32),
            covariances.astype(np.float32),
            measurements.astype(np.float32),
            track_ids,
        )
        posterior_mean = np.asarray(posterior_mean, dtype=np.float64)
        posterior_covariance = np.asarray(posterior_covariance, dtype=np.float64)
        _finite("batched Mamba posterior mean", posterior_mean, (len(items), 8))
        _finite(
            "batched Mamba posterior covariance",
            posterior_covariance,
            (len(items), 8, 8),
        )
        for index, (track_id, measurement) in enumerate(items):
            state = self.states[track_id]
            state.event_gap = state.missing_age
            state.mean = posterior_mean[index].copy()
            state.covariance = posterior_covariance[index].copy()
            state.posterior_mean = state.mean.copy()
            state.posterior_covariance = state.covariance.copy()
            state.posterior_box = mean_to_xyxy(state.mean)
            state.last_observation = measurement.astype(np.float64, copy=True)
            state.missing_age = 0

    def record_event(self, event: Any, nsa_full_update_box: np.ndarray) -> None:
        track_id = int(event.track_id)
        if track_id not in self.states:
            raise KeyError(f"Mamba shadow record for unknown NSA track: {track_id}")
        state = self.states[track_id]
        matched = bool(event.has_detection)
        if matched:
            detection_index = int(event.detection.detection_index)
            detection_box = np.asarray(event.detection.box, dtype=np.float64)
        else:
            detection_index = -1
            detection_box = np.zeros(4, dtype=np.float64)
        nsa_prior = np.asarray(event.pre_update_state.mean, dtype=np.float64)
        self.writer.append(
            {
                "event_id": event.event_id,
                "frame_id": int(event.frame_id),
                "track_id": track_id,
                "matched": matched,
                "gap": int(state.event_gap),
                "accepted_detection_index": detection_index,
                "mamba_prior_mean": state.prior_mean,
                "mamba_prior_covariance": state.prior_covariance,
                "mamba_prior_box": state.prior_box,
                "mamba_posterior_mean": state.posterior_mean,
                "mamba_posterior_covariance": state.posterior_covariance,
                "mamba_posterior_box": state.posterior_box,
                "nsa_prior_box": mean_to_xyxy(nsa_prior),
                "nsa_full_update_box": nsa_full_update_box,
                "accepted_detection_box": detection_box,
            }
        )

    def remove_track(self, track_id: int) -> None:
        track_id = int(track_id)
        if track_id in self.states:
            self.filter.delete_track(track_id)
            del self.states[track_id]

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.writer.close()
        for track_id in list(self.states):
            self.remove_track(track_id)
        self._closed = True
        return self.writer.close()

    def abort(self) -> None:
        for track_id in list(self.states):
            self.remove_track(track_id)
        self._closed = True
        self.writer.abort()
