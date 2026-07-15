#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import json
import shutil
import zipfile
from pathlib import Path


BASE_SEQUENCES = ["MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07", "MOT17-08", "MOT17-12", "MOT17-14"]
DETECTORS = ["DPM", "FRCNN", "SDP"]


def _sequence_length(data_root: Path, sequence: str) -> int:
    config = configparser.ConfigParser()
    path = data_root / sequence / "seqinfo.ini"
    if not path.is_file():
        raise FileNotFoundError(f"missing seqinfo.ini: {path}")
    config.read(path)
    return int(config["Sequence"]["seqLength"])


def _validate_result(path: Path, sequence_length: int) -> dict[str, int]:
    rows = 0
    minimum_frame = sequence_length + 1
    maximum_frame = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split(",")
            if len(fields) != 10:
                raise ValueError(f"{path}:{line_number} has {len(fields)} columns, expected 10")
            try:
                values = [float(field) for field in fields]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} contains a non-numeric value") from exc
            frame = int(values[0])
            if values[0] != frame or not 1 <= frame <= sequence_length:
                raise ValueError(
                    f"{path}:{line_number} frame {values[0]} is outside [1, {sequence_length}]"
                )
            rows += 1
            minimum_frame = min(minimum_frame, frame)
            maximum_frame = max(maximum_frame, frame)
    return {
        "rows": rows,
        "minimum_frame": 0 if rows == 0 else minimum_frame,
        "maximum_frame": maximum_frame,
        "sequence_length": sequence_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand MOT17 FRCNN test results and create a validated 21-file ZIP."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--data-root", default="/home/shang/datasets/MOT17/test")
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    source_dir = Path(args.source_dir).resolve()
    data_root = Path(args.data_root).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"test result folder does not exist: {source_dir}")

    for base in BASE_SEQUENCES:
        source = source_dir / f"{base}-FRCNN.txt"
        if not source.is_file():
            raise FileNotFoundError(f"missing FRCNN test result: {source}")
        for detector in ("DPM", "SDP"):
            shutil.copy2(source, source_dir / f"{base}-{detector}.txt")

    expected = {
        f"{base}-{detector}.txt"
        for base in BASE_SEQUENCES
        for detector in DETECTORS
    }
    actual = {path.name for path in source_dir.glob("MOT17-*.txt")}
    if actual != expected:
        raise ValueError(
            f"submission folder must contain exactly 21 MOT17 files; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    validation = {}
    for filename in sorted(expected):
        sequence = filename[: -len(".txt")]
        validation[filename] = _validate_result(
            source_dir / filename, _sequence_length(data_root, sequence)
        )

    output_zip = Path(args.output_zip).resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in sorted(expected):
            archive.write(source_dir / filename, arcname=filename)
    with zipfile.ZipFile(output_zip) as archive:
        archived = set(archive.namelist())
    if archived != expected or any("/" in name for name in archived):
        raise ValueError("submission ZIP must contain exactly 21 files at its root")
    manifest = {
        "schema_version": 1,
        "source_dir": str(source_dir),
        "output_zip": str(output_zip),
        "file_count": len(expected),
        "zip_root_only": True,
        "validation": validation,
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
