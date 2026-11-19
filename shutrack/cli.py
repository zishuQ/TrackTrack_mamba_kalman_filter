"""Command line entry points for SHUTrack v2.

Examples
--------
Train (new checkpoint directory):

    ./.venv/bin/python -m shutrack.cli train \
        --dataset-dir outputs/agentguard/train_data/MOT17 \
        --checkpoint-dir outputs/agentguard/experiments/shutrack_v2_mot17/checkpoints \
        --device cuda --epochs 100 --batch-size 1024 --num-workers 0

Inspect the read-only packed source:

    ./.venv/bin/python -m shutrack.cli inspect-data \
        --dataset-dir outputs/agentguard/train_data/MOT17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from shutrack.training.checkpoint import checkpoint_summary, load_shutrack_checkpoint
from shutrack.training.train import (
    TRAIN_DEFAULTS,
    evaluate_checkpoint,
    train_shutrack,
)


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS["batch_size"])
    parser.add_argument("--num-workers", type=int, default=TRAIN_DEFAULTS["num_workers"])
    parser.add_argument("--lr", type=float, default=TRAIN_DEFAULTS["lr"])
    parser.add_argument(
        "--weight-decay", type=float, default=TRAIN_DEFAULTS["weight_decay"]
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=TRAIN_DEFAULTS["warmup_epochs"]
    )
    parser.add_argument("--warmup-steps", type=int, default=TRAIN_DEFAULTS["warmup_steps"])
    parser.add_argument("--grad-clip", type=float, default=TRAIN_DEFAULTS["grad_clip"])
    parser.add_argument("--seed", type=int, default=TRAIN_DEFAULTS["seed"])
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=TRAIN_DEFAULTS["log_interval"])
    parser.add_argument(
        "--competition",
        choices=["on", "off"],
        default="on" if TRAIN_DEFAULTS["competition_enabled"] else "off",
        help="Train with the association-competition branch or the lean dual-evidence variant",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="Initialize model weights from an A checkpoint with a fresh optimizer",
    )
    parser.add_argument("--memory-shards", type=int, default=TRAIN_DEFAULTS["memory_shards"])
    parser.add_argument("--shard-cycles", type=int, default=TRAIN_DEFAULTS["shard_cycles"])
    parser.add_argument("--epochs-per-shard", type=int, default=TRAIN_DEFAULTS["epochs_per_shard"])
    parser.add_argument(
        "--sequence-sampling",
        choices=["sample-proportional", "sqrt-size"],
        default=TRAIN_DEFAULTS["sequence_sampling"],
    )
    parser.add_argument("--d-model", type=int, default=TRAIN_DEFAULTS["d_model"])
    parser.add_argument("--n-heads", type=int, default=TRAIN_DEFAULTS["n_heads"])
    parser.add_argument("--hce-layers", type=int, default=TRAIN_DEFAULTS["hce_layers"])
    parser.add_argument("--ff-dim", type=int, default=TRAIN_DEFAULTS["ff_dim"])
    parser.add_argument("--dropout", type=float, default=TRAIN_DEFAULTS["dropout"])
    parser.add_argument(
        "--appearance-proj-dim",
        type=int,
        default=TRAIN_DEFAULTS["appearance_proj_dim"],
    )
    parser.add_argument(
        "--sequence-cache-size",
        type=int,
        default=TRAIN_DEFAULTS["sequence_cache_size"],
    )


def _train_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        key: getattr(args, key)
        for key in (
            "dataset_dir",
            "checkpoint_dir",
            "device",
            "epochs",
            "batch_size",
            "num_workers",
            "lr",
            "weight_decay",
            "warmup_epochs",
            "warmup_steps",
            "grad_clip",
            "seed",
            "max_steps",
            "max_samples",
            "log_interval",
            "resume",
            "init_checkpoint",
            "memory_shards",
            "shard_cycles",
            "epochs_per_shard",
            "sequence_sampling",
            "d_model",
            "n_heads",
            "hce_layers",
            "ff_dim",
            "dropout",
            "appearance_proj_dim",
            "sequence_cache_size",
        )
    }
    config["competition_enabled"] = str(args.competition).lower() == "on"
    return config


def _cmd_train(args: argparse.Namespace) -> None:
    summary = train_shutrack(_train_config_from_args(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


def _cmd_smoke(args: argparse.Namespace) -> None:
    config = _train_config_from_args(args)
    if int(config["max_steps"]) <= 0:
        config["max_steps"] = 20
    if int(config["batch_size"]) > 64:
        config["batch_size"] = 32
    print(json.dumps(train_shutrack(config), indent=2, sort_keys=True))


def _cmd_validate_checkpoint(args: argparse.Namespace) -> None:
    if args.dataset_dir:
        result = evaluate_checkpoint(
            args.checkpoint,
            args.dataset_dir,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            device=args.device,
        )
    else:
        model, checkpoint = load_shutrack_checkpoint(
            args.checkpoint, map_location="cpu"
        )
        result = {
            "checkpoint": str(args.checkpoint),
            "summary": checkpoint_summary(checkpoint),
            "parameter_counts": model.parameter_report(),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


def _cmd_inspect_data(args: argparse.Namespace) -> None:
    from shutrack.data.packed_dataset import PackedShutrackDataset

    dataset = PackedShutrackDataset(
        args.dataset_dir,
        max_samples=args.max_samples,
        competition_enabled=str(args.competition).lower() == "on",
    )
    try:
        sample = dataset[0]
        result = {
            "descriptor": dataset.descriptor(),
            "total_samples": dataset.total_samples,
            "sequence_counts": dict(zip(dataset.sequences, dataset._sequence_counts)),
            "first_sample": {
                key: (list(value.shape) if hasattr(value, "shape") else value)
                for key, value in sample.items()
                if key
                in {
                    "history_evidence",
                    "history_track",
                    "history_detection",
                    "history_mask",
                    "history_has_detection",
                    "history_reset",
                    "current_evidence",
                    "current_competition",
                    "current_track",
                    "current_detection",
                    "current_has_detection",
                    "safe_gate_target",
                    "valid_channels",
                    "sample_weight",
                    "sequence",
                    "track_id",
                    "segment_id",
                }
            },
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        dataset.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("shutrack")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train HCE + RSC")
    _add_train_arguments(train_parser)
    train_parser.set_defaults(func=_cmd_train)

    smoke_parser = subparsers.add_parser(
        "smoke", help="Short real-data training smoke (defaults to 20 steps)"
    )
    _add_train_arguments(smoke_parser)
    smoke_parser.set_defaults(func=_cmd_smoke)

    validate_parser = subparsers.add_parser(
        "validate-checkpoint", help="Strict-load a SHUTrack checkpoint"
    )
    validate_parser.add_argument("--checkpoint", required=True)
    validate_parser.add_argument("--dataset-dir", default="")
    validate_parser.add_argument("--batch-size", type=int, default=64)
    validate_parser.add_argument("--max-batches", type=int, default=8)
    validate_parser.add_argument("--device", default="cpu")
    validate_parser.set_defaults(func=_cmd_validate_checkpoint)

    inspect_parser = subparsers.add_parser(
        "inspect-data", help="Inspect the packed read-only training source"
    )
    inspect_parser.add_argument("--dataset-dir", required=True)
    inspect_parser.add_argument("--max-samples", type=int, default=0)
    inspect_parser.add_argument(
        "--competition",
        choices=["on", "off"],
        default="on",
        help="Inspect the full branch or the lean dual-evidence variant",
    )
    inspect_parser.set_defaults(func=_cmd_inspect_data)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
