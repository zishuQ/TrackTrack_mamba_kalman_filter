#!/usr/bin/env python3
"""Run train_supervision_groups.py with a stop after a snapshot epoch.

The LR schedule is still computed from the original --max-epochs budget; this
wrapper only stops after the snapshot checkpoint of the requested epoch has
been written (last.pt and epoch_NNN.pt both exist). Usage:

  python tools/train_stop_after.py <trainer args...> --stop-after-epoch 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_supervision_groups as trainer


class StopAfterEpoch(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stop-after-epoch", type=int, required=True)
    known, remaining = parser.parse_known_args()
    stop = int(known.stop_after_epoch)
    original = trainer.save_checkpoint_atomic
    snapshot = f"epoch_{stop:03d}.pt"

    def save(path, payload):
        original(path, payload)
        if Path(path).name == snapshot and int(payload.get("epoch", 0)) >= stop:
            raise StopAfterEpoch()

    trainer.save_checkpoint_atomic = save
    sys.argv = [sys.argv[0], *remaining]
    try:
        trainer.main()
    except StopAfterEpoch:
        print(json.dumps({"stop_after_epoch": stop, "snapshot": snapshot}), flush=True)
    finally:
        trainer.save_checkpoint_atomic = original


if __name__ == "__main__":
    main()
