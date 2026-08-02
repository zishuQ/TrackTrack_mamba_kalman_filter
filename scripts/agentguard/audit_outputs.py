#!/usr/bin/env python3
"""Read-only catalog and cleanup audit for outputs/agentguard.

The command never removes or moves files. It only reads output metadata and
can write a Markdown report outside the output tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "agentguard"
DEFAULT_MANIFEST = REPO_ROOT / "agentguard" / "configs" / "official_baselines.json"


@dataclass
class Stat:
    files: int = 0
    bytes: int = 0

    def add(self, size: int) -> None:
        self.files += 1
        self.bytes += int(size)


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{value} B"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _prefixes(relative: str) -> Iterable[str]:
    parts = Path(relative).parts
    for index in range(1, len(parts) + 1):
        yield Path(*parts[:index]).as_posix()


def _classify(relative: str) -> str:
    parts = Path(relative).parts
    if not parts:
        return "unknown"
    top = parts[0]
    lower = relative.lower()
    name = parts[-1].lower()

    if top == "detection_cache":
        return "input detection cache"
    if top == "event_cache_v3_iwg_v2":
        return "input event cache"
    if top == "datasets":
        return "input dataset"
    if top == "labels":
        if len(parts) >= 2 and parts[1] in {"default", "integration_test", "test_ds"}:
            return "test fixture"
        return "input labels"
    if top == "reports":
        return "evaluation artifact"

    if "checkpoints" in parts and name.endswith((".pt", ".pth", ".ckpt")):
        return "checkpoint"
    if name.endswith((".pt", ".pth", ".ckpt")):
        return "checkpoint"
    if "provenance" in parts:
        return "training logs/resource/provenance"
    if "logs" in parts or name.endswith(".log"):
        if any(token in lower for token in ("eval", "evaluation", "test", "post", "track")):
            return "evaluation artifact"
        return "training logs/resource/provenance"
    if name.endswith(".jsonl") and any(
        token in name for token in ("metric", "resource", "training")
    ):
        return "training logs/resource/provenance"
    if any(token in name for token in ("resource", "training", "metric")):
        return "training logs/resource/provenance"
    if any(token in name for token in ("eval", "evaluation", "tracking", "summary")):
        return "evaluation artifact"
    if top in {"experiments", "overnight", "sweeps"}:
        return "experiment metadata"
    return "unknown"


def _scan(root: Path) -> tuple[dict[str, Stat], dict[str, Stat], dict[str, Stat]]:
    categories: dict[str, Stat] = defaultdict(Stat)
    prefixes: dict[str, Stat] = defaultdict(Stat)
    top_levels: dict[str, Stat] = defaultdict(Stat)

    if not root.is_dir():
        return categories, prefixes, top_levels

    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not (Path(directory) / name).is_symlink()]
        for filename in filenames:
            path = Path(directory) / filename
            try:
                size = path.stat().st_size
                relative = path.relative_to(root).as_posix()
            except OSError:
                continue
            category = _classify(relative)
            categories[category].add(size)
            top_levels[Path(relative).parts[0]].add(size)
            for prefix in _prefixes(relative):
                prefixes[prefix].add(size)
    return categories, prefixes, top_levels


def _relative_path(repo_root: Path, value: str | os.PathLike[str]) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _output_relative(value: str) -> str:
    prefix = "outputs/agentguard/"
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


def _path_status(root: Path, relative: str, prefixes: dict[str, Stat]) -> str:
    path = root / relative
    if not path.exists():
        return "missing"
    stat = prefixes.get(relative, Stat())
    return f"{stat.files} files / {_human_bytes(stat.bytes)}"


def _metadata_for_dataset_dirs(
    output_root: Path,
    prefixes: dict[str, Stat],
) -> list[dict[str, Any]]:
    base = output_root / "datasets" / "iwg_rg_cma"
    rows: list[dict[str, Any]] = []
    if not base.is_dir():
        return rows
    for dataset_dir in sorted(path for path in base.glob("*/*") if path.is_dir()):
        metadata_path = dataset_dir / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
        relative = dataset_dir.relative_to(output_root).as_posix()
        rows.append(
            {
                "path": relative,
                "dataset": metadata.get("dataset", dataset_dir.parts[-2]),
                "dataset_sha256": metadata.get("dataset_sha256", ""),
                "metadata_sha256": _sha256_file(metadata_path)
                if metadata_path.is_file()
                else "",
                "context_size": metadata.get("context_size", ""),
                "split": metadata.get("split", ""),
                "status": _path_status(output_root, relative, prefixes),
            }
        )
    return rows


def _metadata_for_label_dirs(output_root: Path, prefixes: dict[str, Stat]) -> list[str]:
    rows: set[str] = set()
    base = output_root / "labels"
    if not base.is_dir():
        return []
    for summary in base.rglob("summary.json"):
        rows.add(summary.parent.relative_to(output_root).as_posix())
    for name in ("default", "integration_test", "test_ds"):
        if (base / name).is_dir():
            rows.add((base / name).relative_to(output_root).as_posix())
    return sorted(rows)


def _official_rows(
    repo_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    prefixes: dict[str, Stat],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, entry in sorted((manifest.get("baselines") or {}).items()):
        checkpoint_rel = str(entry.get("checkpoint", ""))
        checkpoint = repo_root / checkpoint_rel
        actual_hash = _sha256_file(checkpoint) if checkpoint.is_file() else ""
        metadata_rel = str(entry.get("dataset_metadata", ""))
        metadata = repo_root / metadata_rel
        metadata_hash = _sha256_file(metadata) if metadata.is_file() else ""
        expected_dataset_hash = str(entry.get("dataset_sha256", ""))
        metadata_json = _read_json(metadata) if metadata.is_file() else {}
        dataset_hash_ok = bool(metadata_json) and metadata_json.get("dataset_sha256") == expected_dataset_hash
        label_dir = str(metadata_json.get("label_dir", "")) if metadata_json else ""
        label_rel = _relative_path(repo_root, label_dir) if label_dir else ""
        rows.append(
            {
                "dataset": dataset,
                "checkpoint": checkpoint_rel,
                "checkpoint_status": (
                    "OK" if actual_hash == entry.get("checkpoint_sha256") else "MISMATCH/MISSING"
                ),
                "dataset_dir": str(entry.get("dataset_dir", "")),
                "dataset_status": (
                    "OK"
                    if (repo_root / str(entry.get("dataset_dir", ""))).is_dir()
                    and dataset_hash_ok
                    else "MISMATCH/MISSING"
                ),
                "metadata_status": (
                    "OK" if metadata_hash == entry.get("dataset_metadata_sha256") else "MISMATCH/MISSING"
                ),
                "label_dir": label_rel,
                "checkpoint_size": _human_bytes(checkpoint.stat().st_size)
                if checkpoint.is_file()
                else "-",
            }
        )
    return rows


def _candidate_rows(
    repo_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    prefixes: dict[str, Stat],
    dataset_rows: list[dict[str, Any]],
    label_rows: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official_dataset_dirs = {
        _output_relative(str(entry.get("dataset_dir", "")))
        for entry in (manifest.get("baselines") or {}).values()
    }
    official_label_dirs = {
        _output_relative(row["label_dir"])
        for row in _official_rows(repo_root, output_root, manifest, prefixes)
        if row["label_dir"]
    }
    official_hashes = {
        str(entry.get("dataset_sha256", ""))
        for entry in (manifest.get("baselines") or {}).values()
    }

    high_confidence: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    explicit_high_confidence = {
        "outputs/agentguard/datasets/iwg_rg_cma/MOT17/nsa_all_v3_jsonl_native_log":
            "same dataset_sha256 as the project MOT17 baseline; duplicate native-log output",
        "outputs/agentguard/datasets/iwg_rg_cma/MOT17/nsa_all_v3_jsonl_rerun_native_log":
            "same dataset_sha256 as the project MOT17 baseline; rerun duplicate",
        "outputs/agentguard/labels/default": "test fixture labels",
        "outputs/agentguard/labels/integration_test": "integration-test labels",
        "outputs/agentguard/labels/test_ds": "test fixture labels",
    }
    explicit_review = {
        "outputs/agentguard/datasets/iwg_rg_cma/SportsMOT/nsa_train_v3_compact":
            "train-only variant; referenced by the schedule-matrix workflow and historical training outputs",
        "outputs/agentguard/datasets/iwg_rg_cma/SportsMOT/nsa_trainval_h8_v3_compact":
            "horizon h8 / context 6 variant; referenced by h8 training workflows and historical outputs",
        "outputs/agentguard/datasets/iwg_rg_cma/SportsMOT/nsa_trainval_future8_context8_v3_compact":
            "future8 / context 8 variant; referenced by context8 training workflows and historical outputs",
    }

    for row in dataset_rows:
        full = f"outputs/agentguard/{row['path']}"
        if row["path"] in official_dataset_dirs:
            continue
        reason = explicit_high_confidence.get(full)
        if reason is None and row.get("dataset_sha256") in official_hashes:
            reason = "metadata has the same dataset_sha256 as a project baseline"
        item = {"path": full, "reason": reason or explicit_review.get(full) or "non-canonical dataset; compare experiment provenance before deleting", "status": row["status"]}
        if reason:
            high_confidence.append(item)
        else:
            review.append(item)

    for label_dir in label_rows:
        full = f"outputs/agentguard/{label_dir}"
        if label_dir in official_label_dirs:
            continue
        reason = explicit_high_confidence.get(full)
        item = {"path": full, "reason": reason or "non-canonical label root; compare dataset metadata and provenance before deleting", "status": _path_status(output_root, label_dir, prefixes)}
        if reason:
            high_confidence.append(item)
        else:
            review.append(item)

    return high_confidence, review


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(repo_root: Path, output_root: Path, manifest_path: Path) -> str:
    manifest = _read_json(manifest_path) or {}
    categories, prefixes, top_levels = _scan(output_root)
    dataset_rows = _metadata_for_dataset_dirs(output_root, prefixes)
    label_rows = _metadata_for_label_dirs(output_root, prefixes)
    official_rows = _official_rows(repo_root, output_root, manifest, prefixes)
    high_confidence, review = _candidate_rows(
        repo_root, output_root, manifest, prefixes, dataset_rows, label_rows
    )

    category_rows = [
        (name, stat.files, _human_bytes(stat.bytes))
        for name, stat in sorted(categories.items())
    ]
    top_rows = [
        (name, stat.files, _human_bytes(stat.bytes))
        for name, stat in sorted(top_levels.items())
    ]
    dataset_table = [
        (
            row["dataset"],
            row["path"],
            row["split"],
            row["context_size"],
            row["dataset_sha256"][:12] or "-",
            row["status"],
        )
        for row in dataset_rows
    ]
    candidate_table = [
        (item["path"], item["status"], item["reason"]) for item in high_confidence
    ]
    review_table = [(item["path"], item["status"], item["reason"]) for item in review]

    lines = [
        "# AgentGuard Outputs Catalog",
        "",
        "This report is generated by `scripts/agentguard/audit_outputs.py`.",
        "The scan is read-only: it does not move, delete, rewrite, or rename anything under `outputs/agentguard`.",
        "",
        f"- Output root: `{output_root}`",
        f"- Baseline manifest: `{manifest_path}`",
        "- Policy: keep every directory under `outputs/agentguard/experiments`; manual deletion candidates below require human confirmation.",
        "",
        "## Classification",
        "",
        _table(["Category", "Files", "Size"], category_rows),
        "",
        "The top-level roots are summarized separately so the same output can be located quickly:",
        "",
        _table(["Root", "Files", "Size"], top_rows),
        "",
        "## Project Canonical Baselines (User-Selected)",
        "",
        "Here, \"official\" means the project baselines selected by the project owner; it does not mean an external dataset-owner or benchmark designation.",
        "",
        _table(
            ["Dataset", "Checkpoint", "Checkpoint", "Dataset", "Metadata", "Label root"],
            [
                (
                    row["dataset"],
                    row["checkpoint"],
                    f"{row['checkpoint_status']} ({row['checkpoint_size']})",
                    f"{row['dataset_status']}\n{row['dataset_dir']}",
                    row["metadata_status"],
                    row["label_dir"] or "-",
                )
                for row in official_rows
            ],
        ),
        "",
        "The baseline manifest is the authority for which checkpoint and dataset are current. Historical experiment folders remain preserved by policy.",
        "",
        "## Dataset And Label Inputs",
        "",
        _table(
            ["Dataset", "Path", "Split", "Context", "dataset_sha256", "Inventory"],
            dataset_table,
        )
        if dataset_table
        else "No compact dataset metadata was found.",
        "",
        "Label roots discovered from `summary.json` and known test fixture directories:",
        "",
        "\n".join(f"- `{path}`" for path in label_rows) if label_rows else "- None",
        "",
        "## Manual Deletion Candidates",
        "",
        "These entries are recommendations only. Delete them manually only after checking that no experiment provenance or external workflow still needs them.",
        "",
        "### High Confidence",
        "",
        _table(["Path", "Inventory", "Reason"], candidate_table)
        if candidate_table
        else "No high-confidence candidates were found.",
        "",
        "### Review Before Deleting",
        "",
        _table(["Path", "Inventory", "Reason"], review_table)
        if review_table
        else "No additional non-canonical dataset or label roots were found.",
        "",
        "## Keep",
        "",
        "- All `outputs/agentguard/experiments/**` historical outputs, including old checkpoints, logs, resource records, provenance, and evaluation artifacts.",
        "- The three project baseline checkpoint paths and their referenced dataset/label inputs when their manifest checks are `OK`.",
        "- Shared `detection_cache` and `event_cache_v3_iwg_v2` inputs unless a future manifest explicitly marks a replacement.",
        "",
        "## Category Meaning",
        "",
        "- `input detection cache` / `input event cache`: model-input event and detection data.",
        "- `input labels` / `input dataset`: generated training inputs.",
        "- `checkpoint`: model weights under experiment checkpoint directories.",
        "- `training logs/resource/provenance`: training logs, metrics, resource traces, commands, hashes, and environment records.",
        "- `evaluation artifact`: evaluation logs, summaries, reports, and post-processing outputs.",
        "- `test fixture`: synthetic or integration-test data.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the Markdown report here. Without this flag, print it to stdout.",
    )
    args = parser.parse_args()
    report = build_report(REPO_ROOT, args.output_root.resolve(), args.manifest.resolve())
    if args.report is None:
        print(report, end="")
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
