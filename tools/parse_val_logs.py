#!/usr/bin/env python3
"""Parse run.py TrackEval logs into COMBINED and per-sequence tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METRICS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA")
ROW = re.compile(
    r"^(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
)


def parse_log(path: Path) -> dict:
    lines = path.read_text(errors="ignore").replace("\r", "\n").splitlines()
    combined = None
    per_sequence: dict[str, dict[str, float]] = {}
    in_per_sequence = False
    for index, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("Per-sequence metrics"):
            in_per_sequence = True
            continue
        if line.startswith("HOTA") and "AssA" in line and "MOTA" in line:
            values = lines[index + 1].strip() if index + 1 < len(lines) else ""
            numbers = re.findall(r"[0-9.]+", values)
            if len(numbers) >= 5 and not in_per_sequence:
                combined = {
                    name: float(value) for name, value in zip(METRICS, numbers[:5])
                }
            continue
        match = ROW.match(line)
        if in_per_sequence and match:
            sequence = match.group(1)
            per_sequence[sequence] = {
                name: float(value)
                for name, value in zip(METRICS, match.groups()[1:])
            }
    return {"combined": combined, "per_sequence": per_sequence}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True, help="name=path")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = {}
    for item in args.log:
        name, path = item.split("=", 1)
        result[name] = parse_log(Path(path))
    payload = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
