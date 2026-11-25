#!/usr/bin/env python3
"""Package DanceTrack test results in the official layout (tracker/*.txt).

Validates the expected sequence set, 10 columns, finite values and frame
ranges (via seqinfo.ini when available), then writes a ZIP with a top-level
``tracker/`` folder, matching the Codabench DanceTrack submission format.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_lengths(root: Path | None) -> dict[str, int]:
    lengths = {}
    if root is None:
        return lengths
    for ini in root.glob("*/seqinfo.ini"):
        parser = configparser.ConfigParser()
        parser.read(ini)
        try:
            lengths[ini.parent.name] = int(parser["Sequence"]["seqLength"])
        except (KeyError, ValueError):
            continue
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--expected-names-file", required=True)
    parser.add_argument("--seqinfo-root", default="")
    args = parser.parse_args()

    source = Path(args.source)
    expected = [line.strip() for line in Path(args.expected_names_file).read_text().splitlines()
                if line.strip()]
    expected = [name[:-4] if name.endswith(".txt") else name for name in expected]
    files = sorted(source.glob("*.txt"))
    names = {path.stem for path in files}
    missing = sorted(set(expected) - names)
    extra = sorted(names - set(expected))
    if missing or extra:
        raise SystemExit(f"file set mismatch: missing={missing[:5]} extra={extra[:5]}")
    lengths = sequence_lengths(Path(args.seqinfo_root) if args.seqinfo_root else None)

    rows = 0
    for path in files:
        count = 0
        with path.open() as handle:
            for line_number, raw in enumerate(handle, start=1):
                fields = raw.rstrip("\r\n").split(",")
                if len(fields) != 10:
                    raise SystemExit(f"{path.name}:{line_number} has {len(fields)} columns")
                try:
                    values = [float(field) for field in fields]
                except ValueError:
                    raise SystemExit(f"{path.name}:{line_number} has a non-numeric value")
                if not all(math.isfinite(value) for value in values):
                    raise SystemExit(f"{path.name}:{line_number} has NaN/Inf")
                frame = int(values[0])
                if frame != values[0] or frame < 1:
                    raise SystemExit(f"{path.name}:{line_number} has invalid frame {values[0]}")
                maximum = lengths.get(path.stem)
                if maximum is not None and frame > maximum:
                    raise SystemExit(f"{path.name}:{line_number} frame {frame} > {maximum}")
                count += 1
        if count == 0:
            raise SystemExit(f"empty result file: {path}")
        rows += count

    zip_path = Path(args.zip)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("tracker/", "")
        for path in files:
            archive.write(path, arcname=f"tracker/{path.name}")
    digest = sha256(zip_path)
    (zip_path.parent / (zip_path.name + ".sha256")).write_text(digest + "\n")
    print(json.dumps({"zip": str(zip_path), "sha256": digest, "files": len(files),
                      "rows": rows, "size_bytes": zip_path.stat().st_size,
                      "status": "ok"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
