#!/usr/bin/env python3
"""Launch TrackTrack run.py in iwg-rg-cma mode while recording per-event gates.

Diagnostic-only wrapper: it forwards every CLI argument to run.py unchanged and
only adds an instrumentation hook on RuntimeStatistics.record_rg_cma. The
result JSON contains per-sequence distributions of the applied gate (the value
consumed by Track.update_with_gates), plus base/final RG-CMA gates, for matched
events only.

Usage:
  IWG_GATE_RECORD=/path/gates.json python tools/run_iwg_gate_eval.py <run.py args...>
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

TRACKER = Path("/home/shang/workspace/TrackTrack/3. Tracker")
OUT = Path(os.environ["IWG_GATE_RECORD"])


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    quantiles = np.quantile(array, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).tolist()
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "quantiles": [float(v) for v in quantiles],
        "hist_20": np.histogram(array, bins=20, range=(0.0, 1.0))[0].tolist(),
        "fraction_gt095": float((array > 0.95).mean()),
        "fraction_lt005": float((array < 0.05).mean()),
        "fraction_lt05": float((array < 0.5).mean()),
        "all_exactly_one": bool(np.all(array == 1.0)),
    }


def main() -> int:
    run_args = sys.argv[1:]
    os.chdir(TRACKER)
    sys.path.insert(0, str(TRACKER))
    from agentguard.runtime.statistics import RuntimeStatistics
    from integrations.agentguard.adapter import AgentGuardTrackerAdapter

    current = {"sequence": "unknown"}
    data = defaultdict(lambda: {
        "motion": [], "appearance": [],
        "base_motion": [], "base_appearance": [],
        "final_motion": [], "final_appearance": [],
        "matched": 0, "unmatched": 0,
    })

    original_init = AgentGuardTrackerAdapter.__init__

    def init_patched(self, args, vid_name, *rest, **kwargs):
        current["sequence"] = str(vid_name)
        return original_init(self, args, vid_name, *rest, **kwargs)

    AgentGuardTrackerAdapter.__init__ = init_patched

    original_record = RuntimeStatistics.record_rg_cma

    def record_patched(self, gate, base_gate, final_gate, correction, *, matched, **kwargs):
        bucket = data[current["sequence"]]
        applied = np.asarray(gate, dtype=np.float64).reshape(2)
        base = np.asarray(base_gate, dtype=np.float64).reshape(2)
        final = np.asarray(final_gate, dtype=np.float64).reshape(2)
        if matched:
            bucket["motion"].append(float(applied[0]))
            bucket["appearance"].append(float(applied[1]))
            bucket["base_motion"].append(float(base[0]))
            bucket["base_appearance"].append(float(base[1]))
            bucket["final_motion"].append(float(final[0]))
            bucket["final_appearance"].append(float(final[1]))
            bucket["matched"] += 1
        else:
            bucket["unmatched"] += 1
        return original_record(self, gate, base_gate, final_gate, correction,
                               matched=matched, **kwargs)

    RuntimeStatistics.record_rg_cma = record_patched

    code = 0
    try:
        sys.argv = ["run.py", *run_args]
        runpy.run_path(str(TRACKER / "run.py"), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        payload = {"sequences": {}}
        for sequence, bucket in data.items():
            payload["sequences"][sequence] = {
                "matched": bucket["matched"],
                "unmatched": bucket["unmatched"],
                "motion": distribution(bucket["motion"]),
                "appearance": distribution(bucket["appearance"]),
                "base_motion": distribution(bucket["base_motion"]),
                "final_motion": distribution(bucket["final_motion"]),
                "base_appearance": distribution(bucket["base_appearance"]),
                "final_appearance": distribution(bucket["final_appearance"]),
            }
        payload["global"] = {
            "motion": distribution([v for b in data.values() for v in b["motion"]]),
            "appearance": distribution([v for b in data.values() for v in b["appearance"]]),
            "base_motion": distribution([v for b in data.values() for v in b["base_motion"]]),
            "final_motion": distribution([v for b in data.values() for v in b["final_motion"]]),
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"gate_record": str(OUT)}, indent=1), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
