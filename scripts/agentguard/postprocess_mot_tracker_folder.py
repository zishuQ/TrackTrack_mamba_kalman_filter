#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply TrackTrack's production Gaussian interpolation to MOT files."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--tau", type=int, default=12)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "3. Tracker"))
    from utils.gbi import gb_interpolation

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(source_dir.glob("*.txt"))
    if not sources:
        raise FileNotFoundError(f"no MOT result files in {source_dir}")
    for source in sources:
        destination = output_dir / source.name
        gb_interpolation(
            str(source),
            str(destination),
            interval=int(args.interval),
            tau=int(args.tau),
        )
        print(f"{source.name} -> {destination}", flush=True)


if __name__ == "__main__":
    main()
