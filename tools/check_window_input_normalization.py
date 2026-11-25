#!/usr/bin/env python3
"""CPU-only check of reconstructed window evidence versus checkpoint normalization."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT, ROOT / "3. Tracker", ROOT / "agentguard/src"):
    sys.path.insert(0, str(path))
import train_supervision_groups as base
from shutrack.schema import EVIDENCE_SCALAR_INDICES
from shutrack.training.recurrent_motion import evidence_from_state

def main():
    torch.set_num_threads(1)
    exp = ROOT / "outputs/agentguard/experiments/sportsmot_mechanisms_20261124"
    checkpoint = exp / "runs/G1/groups/appearance_row_column/checkpoints/last.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stats_mean = np.asarray(payload["normalization_mean"], dtype=np.float64)
    stats_std = np.asarray(payload["normalization_std"], dtype=np.float64)
    config = payload["training_config"]
    data_root = Path(config["data_root"])
    sequence = sorted(path.name for path in data_root.iterdir()
                      if (path / "samples.npz").is_file())[0]
    data = base.SequenceData(sequence, "train", data_root, "SportsMOT")
    data.input_statistics = (stats_mean, stats_std)
    rows = np.arange(min(32, data.n), dtype=np.int64)
    events = data.small["current_event_index"][rows]
    summary = json.loads((data.dir / "summary.json").read_text())
    # CPU tensors only. No model, forward pass, or tracker is started.
    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(np.asarray(value).copy(), dtype=dtype)
    rebuilt = evidence_from_state(
        tensor(data.event_state_mean[events]),
        tensor(data.event_state_cov[events]).reshape(-1, 8, 8),
        tensor(data.event_det_measurement[events]),
        tensor(data.event_det_score[events]),
        tensor(data.event_gap[events], torch.int64),
        tensor(data.event_has[events], torch.bool),
        float(summary["img_width"]), float(summary["img_height"]),
    ).numpy()
    stored = np.asarray(data.event_evidence[events], dtype=np.float32)
    expected = stored.copy()
    cols = list(EVIDENCE_SCALAR_INDICES)
    expected[:, :-1] = (expected[:, :-1] - stats_mean[cols]) / stats_std[cols]
    normalized_rebuilt = rebuilt.copy()
    normalized_rebuilt[:, :-1] = (normalized_rebuilt[:, :-1] - stats_mean[cols]) / stats_std[cols]
    report = {
        "checkpoint_epoch": int(payload["epoch"]),
        "sequence": sequence, "rows": len(rows),
        "raw_rebuild_vs_stored_max_abs": float(np.max(np.abs(rebuilt-stored))),
        "old_training_raw_vs_expected_normalized_max_abs":
            float(np.max(np.abs(rebuilt-expected))),
        "after_normalizing_rebuild_vs_expected_max_abs":
            float(np.max(np.abs(normalized_rebuilt-expected))),
        "detection_score_example": {
            "raw": float(rebuilt[0,10]),
            "expected_model_input": float(expected[0,10]),
        },
        "raw_probe_evidence": rebuilt[0].tolist(),
        "expected_probe_evidence": expected[0].tolist(),
        "note": "Archived pre-review trainer used raw current_evidence; packed_replay_v2 normalizes it. This script diagnoses that archived protocol."
    }
    output = ROOT / "outputs/agentguard/experiments/window_training_speed_audit_20261124"
    output.mkdir(parents=True, exist_ok=True)
    (output / "normalization_check.json").write_text(json.dumps(report, indent=2) + "\n")
    data.samples.close()
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
