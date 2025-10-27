#!/usr/bin/env python3
"""Utility to download YOLOX json metas, weights and optional detection result pack.

Requires: gdown (pip install gdown)

Examples
  python download_assets.py                 # download json + weights
  python download_assets.py --with-dets     # also download detection results folder
  python download_assets.py --only json     # only json files
  python download_assets.py --only weights  # only weight files
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

try:
    import gdown  # type: ignore
except ImportError:
    print("[download_assets] gdown not installed. Run: pip install gdown", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "jsons"
WEIGHTS_DIR = ROOT / "weights"
DET_PARENT = ROOT.parent / "outputs" / "1. det"

JSON_FILES = {
    "mot17_val.json": "1hqcoFTtdzd5xMrC_xgz6mniI_sKg_0G9",
    "mot17_test.json": "1CQ91C7Hl4B2rDfy_IU2orusD9vaiD5cs",
    "mot20_val.json": "16IrR-TWc-K6c6NHwjM3OV74NBXPrxR-_",
    "mot20_test.json": "1h3EkjOpcn058g2tgGGEcD7r5sAGgskwg",
    "dance_val.json": "1O__fCM3gPbzHtav3XrlzHjjs96Dl45m8",
    "dance_test.json": "12rBCIYLCXqT8bYmNrEwNdp6MmJ7whEg-",
}

WEIGHT_FILES = {
    "mot17_half.pth.tar": "1R-eMf5SgwmizMkOjqJq3ZiurWBNGYf1j",
    "mot17.pth.tar": "1MAb-Bhikx-fWe0VlJON_VMrYIyyyrt-F",
    "mot20_half.pth.tar": "1H1BxOfinONCSdQKnjGq0XlRxVUo_4M8o",
    "mot20.pth.tar": "1FunATdHrWfK95RiiEIw2GJ-gXB-tXMPB",
    "dance.pth.tar": "1ZKpYmFYCsRdXuOL60NRuc7VXAFYRskXB",
}

DET_FOLDER_ID = "1Ef-O0DCZAS8ObqJ9cv751ils-KehSgA7"


def download_file(file_id: str, output_path: Path):
    if output_path.exists():
        print(f"[skip] {output_path.name} already exists")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[down] {output_path.name}")
    gdown.download(url, str(output_path), quiet=False)


def download_folder(folder_id: str, output_dir: Path):
    # use gdown's folder download via subprocess to keep simple
    if any(output_dir.glob("*.pickle")):
        print(f"[skip] detection results seem present in {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[down] detection results folder (may be large)")
    cmd = [sys.executable, "-m", "gdown", "--folder", f"https://drive.google.com/drive/folders/{folder_id}", "-O", str(output_dir)]
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-dets", action="store_true", help="Also download detection result pickle files")
    ap.add_argument("--only", choices=["json", "weights"], help="Download only a subset")
    args = ap.parse_args()

    JSON_DIR.mkdir(exist_ok=True)
    WEIGHTS_DIR.mkdir(exist_ok=True)

    if args.only in (None, "json"):
        for name, fid in JSON_FILES.items():
            download_file(fid, JSON_DIR / name)

    if args.only in (None, "weights"):
        for name, fid in WEIGHT_FILES.items():
            download_file(fid, WEIGHTS_DIR / name)

    if args.with_dets and args.only not in ("json",):
        download_folder(DET_FOLDER_ID, DET_PARENT)

    print("[done]")


if __name__ == "__main__":
    main()
