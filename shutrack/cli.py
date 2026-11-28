"""Inspect a current SHUTrack checkpoint."""

from __future__ import annotations

import argparse
import json

from shutrack.training.checkpoint import checkpoint_summary, load_shutrack_checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("shutrack")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-checkpoint", help="Strict-load a SHUTrack checkpoint")
    validate.add_argument("--checkpoint", required=True)
    args = parser.parse_args(argv)

    model, checkpoint = load_shutrack_checkpoint(args.checkpoint, map_location="cpu")
    result = {
        "checkpoint": str(args.checkpoint),
        "summary": checkpoint_summary(checkpoint),
        "parameter_counts": model.parameter_report(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
