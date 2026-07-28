#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from agentguard.datasets.iwg_rg_cma_dataset import StreamingIWGRGCMADataset
from agentguard.training.train_iwg_rg_cma import (
    _forward,
    _move_batch,
    load_iwg_rg_cma_checkpoint,
    validate_iwg_rg_cma_checkpoint_contract,
)


SUPPORTED_ARCHITECTURES = {
    "legacy-clean-cross-modal",
    "legacy-clean-bidirectional-cma",
}
CHANNELS = ("motion", "appearance")
CONDITIONS = ("paired", "appearance_shuffled", "motion_shuffled")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether clean RG-CMA corrections depend on paired motion "
            "and appearance histories without perturbing Base-IWG."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--samples-per-sequence", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sequence_counts(dataset: StreamingIWGRGCMADataset) -> dict[str, int]:
    counts = dataset.metadata.get("timeline_counts", {})
    return {
        sequence: int(counts[sequence]["labeled_endpoints"])
        for sequence in dataset.metadata["train_sequences"]
    }


def _balanced_indices(
    dataset: StreamingIWGRGCMADataset,
    *,
    samples_per_sequence: int,
    seed: int,
) -> tuple[list[int], dict[str, int]]:
    counts = _sequence_counts(dataset)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    selected_counts: dict[str, int] = {}
    offset = 0
    for sequence in dataset.metadata["train_sequences"]:
        count = counts[sequence]
        take = count if samples_per_sequence <= 0 else min(count, samples_per_sequence)
        local = rng.choice(count, size=take, replace=False)
        selected.extend((local + offset).tolist())
        selected_counts[sequence] = int(take)
        offset += count
    rng.shuffle(selected)
    return selected, selected_counts


def _matched_derangement(
    *,
    sequences: list[str],
    padding_mask: torch.Tensor,
) -> torch.LongTensor:
    if padding_mask.ndim != 2 or padding_mask.shape[0] != len(sequences):
        raise ValueError("padding mask and sequence list must share batch size")
    permutation = torch.arange(padding_mask.shape[0], device=padding_mask.device)
    groups: dict[tuple[str, tuple[bool, ...]], list[int]] = {}
    padding_rows = padding_mask.detach().cpu().bool().tolist()
    for index, (sequence, row) in enumerate(zip(sequences, padding_rows)):
        groups.setdefault((sequence, tuple(row)), []).append(index)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        source = torch.tensor(indices, device=padding_mask.device)
        shift = max(1, len(indices) // 2)
        permutation[source] = torch.roll(source, shifts=shift)
    return permutation


@contextlib.contextmanager
def _shuffle_projection_input(
    module: torch.nn.Module,
    permutation: torch.LongTensor,
) -> Iterator[None]:
    def hook(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        value = inputs[0]
        if value.shape[0] != permutation.shape[0]:
            raise RuntimeError("counterfactual permutation has the wrong batch size")
        return (value.index_select(0, permutation), *inputs[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@dataclass
class GateAccumulator:
    weight: float = 0.0
    absolute_error: float = 0.0
    squared_error: float = 0.0
    base_absolute_error: float = 0.0
    correction_absolute: float = 0.0
    help_weight: float = 0.0
    harm_weight: float = 0.0

    def add(
        self,
        *,
        prediction: torch.Tensor,
        base: torch.Tensor,
        correction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        prediction = prediction.float()
        base = base.float()
        target = target.float()
        weight = weight.float()
        absolute_error = (prediction - target).abs()
        base_error = (base - target).abs()
        improvement = base_error - absolute_error
        self.weight += float(weight.sum())
        self.absolute_error += float((absolute_error * weight).sum())
        self.squared_error += float(((prediction - target).square() * weight).sum())
        self.base_absolute_error += float((base_error * weight).sum())
        self.correction_absolute += float((correction.float().abs() * weight).sum())
        self.help_weight += float(((improvement > 1e-6).float() * weight).sum())
        self.harm_weight += float(((improvement < -1e-6).float() * weight).sum())

    def result(self) -> dict[str, float]:
        if self.weight <= 0:
            raise RuntimeError("gate accumulator received no valid weighted samples")
        return {
            "weighted_count": self.weight,
            "mae": self.absolute_error / self.weight,
            "mse": self.squared_error / self.weight,
            "base_mae": self.base_absolute_error / self.weight,
            "correction_improvement_mean": (
                self.base_absolute_error - self.absolute_error
            )
            / self.weight,
            "correction_abs_mean": self.correction_absolute / self.weight,
            "correction_help_rate": self.help_weight / self.weight,
            "correction_harm_rate": self.harm_weight / self.weight,
        }


@dataclass
class DependencyAccumulator:
    weight: float = 0.0
    paired_error: float = 0.0
    shuffled_error: float = 0.0
    correction_change: float = 0.0
    paired_correction_abs: float = 0.0
    shuffled_worse_weight: float = 0.0
    shuffled_better_weight: float = 0.0
    paired_correction_sum: float = 0.0
    shuffled_correction_sum: float = 0.0
    paired_correction_sq_sum: float = 0.0
    shuffled_correction_sq_sum: float = 0.0
    correction_product_sum: float = 0.0
    batch_effects: list[tuple[float, float]] = field(default_factory=list)

    def add(
        self,
        *,
        paired_prediction: torch.Tensor,
        shuffled_prediction: torch.Tensor,
        paired_correction: torch.Tensor,
        shuffled_correction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        weight = weight.float()
        paired_error = (paired_prediction.float() - target.float()).abs()
        shuffled_error = (shuffled_prediction.float() - target.float()).abs()
        error_delta = shuffled_error - paired_error
        paired_correction = paired_correction.float()
        shuffled_correction = shuffled_correction.float()
        batch_weight = float(weight.sum())
        batch_numerator = float((error_delta * weight).sum())
        self.weight += batch_weight
        self.paired_error += float((paired_error * weight).sum())
        self.shuffled_error += float((shuffled_error * weight).sum())
        self.correction_change += float(
            ((shuffled_correction - paired_correction).abs() * weight).sum()
        )
        self.paired_correction_abs += float(
            (paired_correction.abs() * weight).sum()
        )
        self.shuffled_worse_weight += float(
            ((error_delta > 1e-6).float() * weight).sum()
        )
        self.shuffled_better_weight += float(
            ((error_delta < -1e-6).float() * weight).sum()
        )
        self.paired_correction_sum += float((paired_correction * weight).sum())
        self.shuffled_correction_sum += float((shuffled_correction * weight).sum())
        self.paired_correction_sq_sum += float(
            (paired_correction.square() * weight).sum()
        )
        self.shuffled_correction_sq_sum += float(
            (shuffled_correction.square() * weight).sum()
        )
        self.correction_product_sum += float(
            (paired_correction * shuffled_correction * weight).sum()
        )
        if batch_weight > 0:
            self.batch_effects.append((batch_numerator, batch_weight))

    def result(
        self,
        *,
        seed: int,
        bootstrap_resamples: int,
    ) -> dict[str, Any]:
        if self.weight <= 0:
            raise RuntimeError("dependency accumulator received no valid samples")
        paired_mean = self.paired_correction_sum / self.weight
        shuffled_mean = self.shuffled_correction_sum / self.weight
        covariance = (
            self.correction_product_sum / self.weight
            - paired_mean * shuffled_mean
        )
        paired_variance = max(
            self.paired_correction_sq_sum / self.weight - paired_mean**2, 0.0
        )
        shuffled_variance = max(
            self.shuffled_correction_sq_sum / self.weight - shuffled_mean**2, 0.0
        )
        denominator = math.sqrt(paired_variance * shuffled_variance)
        correlation = covariance / denominator if denominator > 0 else 0.0
        effect = (self.shuffled_error - self.paired_error) / self.weight
        ci = _bootstrap_ratio_ci(
            self.batch_effects,
            seed=seed,
            resamples=bootstrap_resamples,
        )
        paired_mae = self.paired_error / self.weight
        correction_abs = self.paired_correction_abs / self.weight
        return {
            "weighted_count": self.weight,
            "paired_mae": paired_mae,
            "shuffled_mae": self.shuffled_error / self.weight,
            "shuffled_minus_paired_mae": effect,
            "shuffled_minus_paired_mae_ci95": ci,
            "relative_mae_change_percent": (
                100.0 * effect / paired_mae if paired_mae > 0 else 0.0
            ),
            "shuffled_worse_rate": self.shuffled_worse_weight / self.weight,
            "shuffled_better_rate": self.shuffled_better_weight / self.weight,
            "correction_change_mae": self.correction_change / self.weight,
            "relative_correction_change": (
                (self.correction_change / self.weight) / correction_abs
                if correction_abs > 0
                else 0.0
            ),
            "paired_vs_shuffled_correction_correlation": correlation,
            "bootstrap_units": len(self.batch_effects),
        }


def _bootstrap_ratio_ci(
    values: list[tuple[float, float]],
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    if not values or resamples <= 0:
        return []
    numerators = np.asarray([value[0] for value in values], dtype=np.float64)
    denominators = np.asarray([value[1] for value in values], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.integers(0, len(values), size=len(values))
        estimates[index] = numerators[sample].sum() / denominators[sample].sum()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _group_masks(
    sequences: list[str],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result = {
        "overall": torch.ones(len(sequences), dtype=torch.bool, device=device)
    }
    for sequence in sorted(set(sequences)):
        result[sequence] = torch.tensor(
            [value == sequence for value in sequences],
            dtype=torch.bool,
            device=device,
        )
    return result


def _condition_accumulators(
    sequences: list[str],
) -> dict[str, dict[str, dict[str, GateAccumulator]]]:
    groups = ["overall", *sequences]
    return {
        condition: {
            channel: {group: GateAccumulator() for group in groups}
            for channel in CHANNELS
        }
        for condition in CONDITIONS
    }


def _dependency_accumulators(
    sequences: list[str],
) -> dict[str, dict[str, DependencyAccumulator]]:
    groups = ["overall", *sequences]
    return {
        direction: {group: DependencyAccumulator() for group in groups}
        for direction in ("appearance_to_motion", "motion_to_appearance")
    }


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2")
    if args.samples_per_sequence == 1:
        raise ValueError("samples-per-sequence must be 0 or at least 2")
    checkpoint_path = Path(args.checkpoint).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_everything(args.seed)

    dataset = StreamingIWGRGCMADataset(dataset_dir)
    model, checkpoint = load_iwg_rg_cma_checkpoint(
        checkpoint_path, map_location=device
    )
    validate_iwg_rg_cma_checkpoint_contract(
        checkpoint,
        expected_dataset_sha256=dataset.metadata["dataset_sha256"],
    )
    if model.architecture_variant not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "counterfactual requires one of "
            f"{sorted(SUPPORTED_ARCHITECTURES)!r}, got "
            f"{model.architecture_variant!r}"
        )
    model.to(device).eval()
    selected_indices, selected_counts = _balanced_indices(
        dataset,
        samples_per_sequence=args.samples_per_sequence,
        seed=args.seed,
    )
    selected_sequences = list(dataset.metadata["train_sequences"])
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=StreamingIWGRGCMADataset.collate_fn,
    )
    condition_totals = _condition_accumulators(selected_sequences)
    dependency_totals = _dependency_accumulators(selected_sequences)
    attention_numerator = {
        "motion_query_to_appearance": 0.0,
        "appearance_query_to_motion": 0.0,
    }
    attention_count = 0
    max_base_difference = 0.0
    processed = 0
    shuffled_samples = 0

    try:
        with torch.inference_mode():
            for batch_index, raw_batch in enumerate(loader, start=1):
                batch = _move_batch(raw_batch, device)
                permutation = _matched_derangement(
                    sequences=raw_batch["sequence"],
                    padding_mask=batch["padding_mask"],
                )
                changed = permutation != torch.arange(
                    permutation.shape[0], device=permutation.device
                )
                shuffled_samples += int(changed.sum())
                paired = _forward(model, batch)
                with _shuffle_projection_input(
                    model.appearance_projection, permutation
                ):
                    appearance_shuffled = _forward(model, batch)
                with _shuffle_projection_input(model.motion_projection, permutation):
                    motion_shuffled = _forward(model, batch)
                outputs = {
                    "paired": paired,
                    "appearance_shuffled": appearance_shuffled,
                    "motion_shuffled": motion_shuffled,
                }
                for counterfactual in (appearance_shuffled, motion_shuffled):
                    max_base_difference = max(
                        max_base_difference,
                        float(
                            (
                                counterfactual["base_gate"]
                                - paired["base_gate"]
                            )
                            .abs()
                            .max()
                        ),
                    )
                if max_base_difference != 0.0:
                    raise AssertionError(
                        "counterfactual projection hook changed Base-IWG output"
                    )

                masks = _group_masks(raw_batch["sequence"], device=device)
                valid_channels = torch.stack(
                    [batch["valid_motion"], batch["valid_appearance"]], dim=-1
                ).float()
                gate_weights = (
                    valid_channels * batch["sample_weight"].float().unsqueeze(-1)
                )
                target = batch["safe_gate_target"].float()
                for condition, condition_output in outputs.items():
                    for channel_index, channel in enumerate(CHANNELS):
                        for group, group_mask in masks.items():
                            weight = gate_weights[:, channel_index] * group_mask.float()
                            if float(weight.sum()) <= 0:
                                continue
                            condition_totals[condition][channel][group].add(
                                prediction=condition_output["refined_gate"][:, channel_index],
                                base=condition_output["base_gate"][:, channel_index],
                                correction=condition_output["gate_correction"][:, channel_index],
                                target=target[:, channel_index],
                                weight=weight,
                            )

                directions = (
                    (
                        "appearance_to_motion",
                        0,
                        appearance_shuffled,
                    ),
                    (
                        "motion_to_appearance",
                        1,
                        motion_shuffled,
                    ),
                )
                for direction, channel_index, shuffled in directions:
                    for group, group_mask in masks.items():
                        weight = (
                            gate_weights[:, channel_index]
                            * group_mask.float()
                            * changed.float()
                        )
                        if float(weight.sum()) <= 0:
                            continue
                        dependency_totals[direction][group].add(
                            paired_prediction=paired["refined_gate"][:, channel_index],
                            shuffled_prediction=shuffled["refined_gate"][:, channel_index],
                            paired_correction=paired["gate_correction"][:, channel_index],
                            shuffled_correction=shuffled["gate_correction"][:, channel_index],
                            target=target[:, channel_index],
                            weight=weight,
                        )

                cross_weights = paired["cross_modal_attention_weights"].float()
                attention_numerator["motion_query_to_appearance"] += float(
                    cross_weights[:, :, 0, 1].sum()
                )
                attention_numerator["appearance_query_to_motion"] += float(
                    cross_weights[:, :, 1, 0].sum()
                )
                attention_count += int(cross_weights.shape[0] * cross_weights.shape[1])
                processed += int(batch["track_feats"].shape[0])
                if args.progress_every > 0 and batch_index % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "batches": batch_index,
                                "processed": processed,
                                "total": len(selected_indices),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        dataset.close()

    condition_results = {
        condition: {
            channel: {
                group: accumulator.result()
                for group, accumulator in group_totals.items()
                if accumulator.weight > 0
            }
            for channel, group_totals in channel_totals.items()
        }
        for condition, channel_totals in condition_totals.items()
    }
    dependency_results = {
        direction: {
            group: accumulator.result(
                seed=args.seed + direction_index * 100 + group_index,
                bootstrap_resamples=args.bootstrap_resamples,
            )
            for group_index, (group, accumulator) in enumerate(group_totals.items())
            if accumulator.weight > 0
        }
        for direction_index, (direction, group_totals) in enumerate(
            dependency_totals.items()
        )
    }
    report = {
        "schema_version": 1,
        "status": "completed",
        "analysis": "paired_cross_modal_counterfactual",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "architecture_variant": model.architecture_variant,
        "dataset_dir": str(dataset_dir),
        "dataset_sha256": dataset.metadata["dataset_sha256"],
        "split": dataset.metadata["split"],
        "device": str(device),
        "seed": args.seed,
        "selection": {
            "policy": "equal_random_without_replacement_per_sequence",
            "samples_per_sequence_limit": args.samples_per_sequence,
            "selected_counts": selected_counts,
            "total_samples": len(selected_indices),
        },
        "intervention": {
            "appearance_shuffled": (
                "permute complete appearance history windows immediately before "
                "RG-CMA appearance_projection; Base-IWG is unchanged"
            ),
            "motion_shuffled": (
                "permute complete motion history windows immediately before "
                "RG-CMA motion_projection; Base-IWG is unchanged"
            ),
            "cross_channel_evidence": {
                "appearance_to_motion": (
                    "effect of shuffled appearance on the motion correction only"
                ),
                "motion_to_appearance": (
                    "effect of shuffled motion on the appearance correction only"
                ),
            },
            "matching_policy": "same_sequence_and_identical_padding_mask",
            "shuffled_samples": shuffled_samples,
            "shuffled_fraction": shuffled_samples / len(selected_indices),
            "base_gate_max_abs_difference": max_base_difference,
        },
        "paired_cross_attention_mean": {
            key: value / attention_count
            for key, value in attention_numerator.items()
        },
        "dependency": dependency_results,
        "condition_metrics": condition_results,
        "limitations": [
            "This is an in-sample supervised gate diagnostic, not tracker HOTA.",
            "Endpoint observations from the same track are correlated.",
            "Attention weights alone are not treated as evidence of causal use.",
            "A positive shuffled-minus-paired MAE indicates useful paired source information.",
        ],
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["dependency"], indent=2, sort_keys=True))
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
