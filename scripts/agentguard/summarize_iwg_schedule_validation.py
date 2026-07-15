#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRIC_NAMES = ("HOTA", "MOTA", "IDF1", "DetA", "AssA")
ROW = re.compile(
    r"^\s*"
    + r"\s+".join([r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))"] * 5)
    + r"\s*$"
)
FILENAME = re.compile(
    r"eval_e(?P<epoch>\d{3})_(?P<gate>base|final)_(?P<stage>raw|post)\.log"
)


def _metrics(path: Path) -> dict[str, float]:
    matches = []
    for line in path.read_text(errors="replace").splitlines():
        match = ROW.match(line)
        if match:
            matches.append([float(value) for value in match.groups()])
    if not matches:
        raise ValueError(f"no combined metric row found in {path}")
    return dict(zip(METRIC_NAMES, matches[-1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize raw/post validation logs for an IWG schedule run."
    )
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    logs = run_root / "logs"
    cases: dict[str, dict] = {}
    for path in sorted(logs.glob("eval_e*_*.log")):
        match = FILENAME.fullmatch(path.name)
        if not match:
            continue
        epoch = str(int(match.group("epoch")))
        gate = match.group("gate")
        stage = match.group("stage")
        cases.setdefault(epoch, {}).setdefault(gate, {})[stage] = _metrics(path)
    if not cases:
        raise RuntimeError(f"no validation cases found under {logs}")
    summary = {
        "run_root": str(run_root),
        "run_name": run_root.name,
        "cases": cases,
    }
    output = run_root / "evaluation_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
