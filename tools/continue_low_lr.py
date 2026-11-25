#!/usr/bin/env python3
"""Continue training from a checkpoint with preserved normalization and a fresh schedule.

Pass the normal train_supervision_groups arguments (including --init and a new
--max-epochs). This wrapper:
  - keeps the source checkpoint's normalization_mean/std instead of refitting;
  - records source checkpoint sha256, source epoch, stage epochs and cumulative
    epochs in the checkpoint payload and a provenance JSON;
  - leaves optimizer and scheduler fresh (no --resume), so the new cosine
    schedule is computed from the new budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as trainer


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--stage-epochs", type=int, required=True)
    parser.add_argument("--cumulative-epochs", type=int, required=True)
    parser.add_argument("--provenance", default="")
    known, remaining = parser.parse_known_args()

    source = Path(known.source_checkpoint)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    norm_mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
    norm_std = np.asarray(payload["normalization_std"], dtype=np.float32)
    meta = {
        "source_checkpoint": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_epoch": int(payload["epoch"]),
        "source_global_step": payload.get("global_step"),
        "stage_epochs": int(known.stage_epochs),
        "cumulative_epochs": int(known.cumulative_epochs),
        "preserve_normalization": True,
        "fresh_optimizer": True,
        "fresh_scheduler": True,
    }
    if known.provenance:
        path = Path(known.provenance)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    original_fit = trainer.fit_input_statistics
    original_build = trainer.build_checkpoint_payload

    def fit(datasets):
        return norm_mean.copy(), norm_std.copy()

    def build(**kwargs):
        result = original_build(**kwargs)
        result["continuation"] = dict(meta)
        return result

    trainer.fit_input_statistics = fit
    trainer.build_checkpoint_payload = build
    sys.argv = [sys.argv[0], *remaining]
    try:
        trainer.main()
    finally:
        trainer.fit_input_statistics = original_fit
        trainer.build_checkpoint_payload = original_build


if __name__ == "__main__":
    main()
