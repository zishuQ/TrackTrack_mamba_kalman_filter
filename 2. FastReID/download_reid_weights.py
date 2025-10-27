#!/usr/bin/env python3
"""Download FastReID ReID weights listed in README.

Usage:
  python download_reid_weights.py              # download all weights
  python download_reid_weights.py --only mot17  # only MOT17 related (half + full)
  python download_reid_weights.py --only mot20  # only MOT20 related (half + full)
  python download_reid_weights.py --only dance  # only DanceTrack

Requires: gdown (pip install gdown)
Existing files are skipped.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

try:
    import gdown  # type: ignore
except ImportError:  # pragma: no cover
    print("[download_reid_weights] Please install gdown: pip install gdown", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"

WEIGHTS = {
    # filename: google drive file id
    "mot17_half_sbs_S50.pth": "1kTG7mVNhYGicR0IXZ0Y1rebVoBRfOMGY",
    "mot17_sbs_S50.pth": "1rUYqWIj0nsQ23rDSv8NVx0Rrp3Lco1KP",
    "mot20_half_sbs_S50.pth": "1xMI_PpfeY02yfkHzRHZfA4KZtRqHak1o",
    "mot20_sbs_S50.pth": "1RhMnTt9JCuZUWk-jPhDPX2NQCZ5g_O3m",
    "dance_sbs_S50.pth": "1c9Vn4PADNKFrCuS0HxhPz3PcTvvLWVhc",
}

GROUPS = {
    "mot17": {"mot17_half_sbs_S50.pth", "mot17_sbs_S50.pth"},
    "mot20": {"mot20_half_sbs_S50.pth", "mot20_sbs_S50.pth"},
    "dance": {"dance_sbs_S50.pth"},
}


def download(file: str, fid: str):
    out = WEIGHTS_DIR / file
    if out.exists():
        print(f"[skip] {file}")
        return
    url = f"https://drive.google.com/uc?id={fid}"
    print(f"[down] {file}")
    gdown.download(url, str(out), quiet=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(GROUPS.keys()), help="Download only specific group")
    args = ap.parse_args()

    WEIGHTS_DIR.mkdir(exist_ok=True)

    if args.only:
        target = GROUPS[args.only]
    else:
        target = set(WEIGHTS.keys())

    for fname in sorted(target):
        download(fname, WEIGHTS[fname])

    print("[done]")


if __name__ == "__main__":
    main()
