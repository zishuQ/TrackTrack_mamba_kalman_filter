"""Zip a flat directory of MOT result TXTs at the archive root.

Checks the file set and non-emptiness before writing, then reports the row
count and SHA-256 of the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-names-file", default="")
    args = parser.parse_args()

    source = Path(args.source)
    files = sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix == ".txt"
    )
    if len(files) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} txt files in {source}, found {len(files)}"
        )
    names = {path.name for path in files}
    if args.expected_names_file:
        expected = set()
        for line in Path(args.expected_names_file).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            expected.add(line if line.endswith(".txt") else line + ".txt")
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        if missing or extra:
            raise SystemExit(
                f"file set mismatch: missing={missing[:5]} extra={extra[:5]}"
            )

    rows = {}
    for path in files:
        count = sum(
            1 for line in path.read_text().splitlines() if line.strip()
        )
        if count == 0:
            raise SystemExit(f"empty result file: {path}")
        rows[path.name] = count

    zip_path = Path(args.zip)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)

    with zipfile.ZipFile(zip_path) as archive:
        archived = sorted(
            name for name in archive.namelist() if not name.endswith("/")
        )
        if archived != sorted(names):
            raise SystemExit("archive contents do not match the source file set")

    digest = sha256(zip_path)
    (zip_path.parent / (zip_path.name + ".sha256")).write_text(digest + "\n")
    print(
        json.dumps(
            {
                "zip": str(zip_path),
                "files": len(files),
                "rows": sum(rows.values()),
                "size_bytes": zip_path.stat().st_size,
                "sha256": digest,
                "per_file_rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
