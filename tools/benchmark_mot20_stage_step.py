#!/usr/bin/env python3
"""Compare staging and mmap on identical MOT20 batches, including I/O time."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT, ROOT / "3. Tracker", ROOT / "agentguard/src"):
    sys.path.insert(0, str(path))

import train_supervision_groups as base  # noqa: E402
from shutrack.models.shutrack import ShutrackV2Config, ShutrackV2Model  # noqa: E402
from shutrack.training.defaults import MAINLINE_STAGE_CHUNK_SAMPLES  # noqa: E402
from shutrack.training.feature_staging import (  # noqa: E402
    aligned_stage_chunk_samples, feature_chunks, release_sequence_pages, staged_sequence_rows,
)
from shutrack.training.loss import (  # noqa: E402
    appearance_column_loss, appearance_rank_loss, motion_box_diou_loss,
)

GROUP = "appearance_row_column"


def train_step(model, optimizer, batch):
    outputs = model(**base.model_inputs(batch))
    motion_loss, _ = motion_box_diou_loss(outputs, batch, harm_weight=1.0)
    appearance_loss, _ = appearance_rank_loss(
        outputs, batch, alpha=0.95, tau=0.1, rank_weight=18.0
    )
    column = appearance_column_loss(outputs, batch, alpha=0.95, tau=0.1, tensor_statistics=True)
    loss = motion_loss + appearance_loss + 50.0 * column
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence", default="MOT20-05")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--chunk-samples", type=int, default=MAINLINE_STAGE_CHUNK_SAMPLES)
    parser.add_argument("--mode", choices=("stage", "mmap"), default="stage")
    parser.add_argument("--cold-cache", action="store_true",
                        help="Request eviction of this sequence's feature pages before timing.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    torch.set_num_threads(4)
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    pin_memory = use_cuda
    batch_size = args.batch_size
    chunk_samples = aligned_stage_chunk_samples(args.chunk_samples, batch_size)
    enabled = args.mode == "stage"
    sequences = sorted(base.official_sequences("train", "MOT20"))
    datasets = {}
    for sequence in sequences:
        with np.load(args.data_root / sequence / "samples.npz", allow_pickle=False) as samples:
            datasets[sequence] = SimpleNamespace(n=len(samples["frame"]))
    _, rows_by_sequence, quotas, _ = base.plan_phase(datasets, 0, 1, "sqrt-size", 0.5, 42, 1)
    rows = rows_by_sequence[args.sequence][:args.steps * batch_size]
    if len(rows) != args.steps * batch_size:
        parser.error("sample plan does not contain the requested number of full batches")

    data = base.SequenceData(args.sequence, "all", args.data_root, "MOT20")
    data.open_cache()
    try:
        data.load_rank_arrays()
        taus = base.load_global_taus(sequences, args.data_root)
        targets = base.derive_targets(data.samples, taus)

        def build_batch(selected):
            return base.batch_tensors(data, selected, targets, GROUP, compact_reid=True)

        defaults = base.build_parser().parse_args(["--group", GROUP])
        variant, width = base.resolve_model_options(defaults)
        base._seed_everything(42)
        model = ShutrackV2Model(ShutrackV2Config(
            reid_dim=data.event_track.shape[1], appearance_proj_dim=width,
            architecture_variant=variant, dropout=0.0,
        )).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        # Initialize kernels and AdamW identically, outside the measured interval.
        with base.prepared_batches(
            rows[:batch_size], batch_size, build_batch, base.PAIRED_TENSOR_KEYS,
            prefetch=1, pin_memory=pin_memory,
        ) as batches:
            for _, batch in batches:
                train_step(model, optimizer, base.to_device(batch, device))
        del batch
        if use_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        if args.cold_cache:
            # Target only feature files used by this benchmark. Advisory, not global.
            release_sequence_pages(data, discard_cache=True)

        compute_events = []
        compute_seconds = 0.0
        losses = []
        staging = []
        started = time.perf_counter()
        for chunk in feature_chunks(rows, chunk_samples, enabled=enabled):
            with staged_sequence_rows(data, chunk, GROUP, enabled=enabled) as info:
                if enabled:
                    staging.append(info)
                with base.prepared_batches(
                    chunk, batch_size, build_batch, base.PAIRED_TENSOR_KEYS,
                    prefetch=1, pin_memory=pin_memory,
                ) as batches:
                    for _, cpu_batch in batches:
                        batch = base.to_device(cpu_batch, device)
                        if use_cuda:
                            start_event = torch.cuda.Event(enable_timing=True)
                            end_event = torch.cuda.Event(enable_timing=True)
                            start_event.record()
                        else:
                            step_started = time.perf_counter()
                        losses.append(train_step(model, optimizer, batch))
                        if use_cuda:
                            end_event.record()
                            compute_events.append((start_event, end_event))
                        else:
                            compute_seconds += time.perf_counter() - step_started
        if use_cuda:
            torch.cuda.synchronize(device)
        total_seconds = time.perf_counter() - started
        if use_cuda:
            compute_seconds = sum(first.elapsed_time(last) for first, last in compute_events) / 1000
        steps = len(losses)
        if steps != args.steps:
            raise RuntimeError(f"expected {args.steps} measured steps, got {steps}")
        result = {
            "sequence": args.sequence, "mode": args.mode, "steps": steps,
            "cold_cache_requested": args.cold_cache, "quota": quotas[args.sequence],
            "rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            "architecture_variant": variant, "appearance_proj_dim": width,
            "chunk_samples": chunk_samples, "batch_size": batch_size,
            "staging_chunks": len(staging),
            "staging_seconds": sum(info["seconds"] for info in staging),
            "peak_stage_chunk_bytes": max((info["bytes"] for info in staging), default=0),
            "elapsed_seconds": total_seconds, "seconds_per_step": total_seconds / steps,
            "compute_seconds_per_step": compute_seconds / steps,
            "mean_loss": float(torch.stack(losses).mean()),
            "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
        }
        if use_cuda:
            result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    finally:
        data.close_cache()

    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
