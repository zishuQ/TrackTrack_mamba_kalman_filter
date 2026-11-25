#!/usr/bin/env python3
"""Codabench submission client for the MOT test servers.

The request shapes mirror the flow recorded in this repository on 2026-09-14
(``outputs/agentguard/experiments/mot20_loo_only_v2_.../eval_logs``):
multipart ``curl -F`` for the metadata calls, a presigned ``PUT`` for the zip,
and a token-authenticated poll of ``/api/submissions/<id>/``.

Commands:
    login    exchange username/password for a DRF token (endpoint to confirm
             when the site is back from maintenance)
    validate local zip checks (file set, 10 columns, frame range)
    submit   reserve -> PUT -> create -> poll (``--dry-run`` prints the exact
             curl commands without sending anything)

Tokens live outside the repository (``~/.config/codabench/token``, mode 600).
"""

from __future__ import annotations

import argparse
import configparser
import getpass
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path

BASE_URL = "https://www.codabench.org"
TOKEN_PATH = Path.home() / ".config" / "codabench" / "token"
MAINTENANCE_MARKER = "maintenance.html"

MOT17_SEQUENCES = (
    "MOT17-01", "MOT17-03", "MOT17-06", "MOT17-07",
    "MOT17-08", "MOT17-12", "MOT17-14",
)
MOT17_DETECTORS = ("DPM", "FRCNN", "SDP")
MOT20_SEQUENCES = ("MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08")


# ----------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------
def _token_from_args(args) -> str:
    token = getattr(args, "token", "") or os.environ.get("CODABENCH_TOKEN", "")
    if token:
        return token
    path = Path(getattr(args, "token_file", "") or TOKEN_PATH)
    if path.is_file():
        return path.read_text().strip()
    raise SystemExit(
        f"no Codabench token: pass --token, set CODABENCH_TOKEN, or run "
        f"`login` (token file: {path})"
    )


def _curl(argv: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["curl", "-sS", *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def cmd_login(args) -> None:
    password = args.password or getpass.getpass("Codabench password: ")
    result = _curl(
        [
            "-X", "POST", f"{args.base_url}/api/api-token-auth/",
            "--data-urlencode", f"username={args.username}",
            "--data-urlencode", f"password={password}",
        ],
        timeout=30,
    )
    text = result.stdout
    if MAINTENANCE_MARKER in text:
        raise SystemExit("login failed: Codabench is in maintenance; retry later")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise SystemExit(f"login failed: unexpected response: {text[:300]}") from None
    token = payload.get("token")
    if not token:
        raise SystemExit(f"login response has no token: {text[:300]}")
    path = Path(args.token_file or TOKEN_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n")
    path.chmod(0o600)
    print(f"token saved to {path}")


# ----------------------------------------------------------------------
# Local validation
# ----------------------------------------------------------------------
def _seq_lengths(seqinfo_root: Path | None) -> dict[str, int]:
    lengths: dict[str, int] = {}
    if seqinfo_root is None:
        return lengths
    for seqinfo in seqinfo_root.glob("*/seqinfo.ini"):
        parser = configparser.ConfigParser()
        parser.read(seqinfo)
        length = int(parser["Sequence"]["seqLength"])
        name = seqinfo.parent.name
        lengths[name] = length
        base = name
        for detector in MOT17_DETECTORS:
            suffix = f"-{detector}"
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        lengths.setdefault(base, length)
    return lengths


def _expected_names(args) -> set[str]:
    dataset = args.dataset.lower()
    if dataset == "mot20":
        return {f"{sequence}.txt" for sequence in MOT20_SEQUENCES}
    if dataset == "mot17":
        return {
            f"{sequence}-{detector}.txt"
            for sequence in MOT17_SEQUENCES
            for detector in MOT17_DETECTORS
        }
    sequences = []
    if args.sequences_file:
        sequences = [
            line.strip()
            for line in Path(args.sequences_file).read_text().splitlines()
            if line.strip()
        ]
    if not sequences:
        raise SystemExit(
            "for this dataset pass --sequences-file with the expected "
            "sequence names (one per line)"
        )
    return {f"{sequence}.txt" for sequence in sequences}


def cmd_validate(args) -> None:
    zip_path = Path(args.zip)
    if not zip_path.is_file():
        raise SystemExit(f"zip not found: {zip_path}")
    expected = _expected_names(args)
    lengths = _seq_lengths(Path(args.seqinfo_root) if args.seqinfo_root else None)
    with zipfile.ZipFile(zip_path) as archive:
        raw_names = {name for name in archive.namelist() if not name.endswith("/")}
        prefix = ("tracker/" if raw_names
                  and all(name.startswith("tracker/") for name in raw_names) else "")
        names = {name[len(prefix):] for name in raw_names}
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        if missing:
            raise SystemExit(
                f"{len(missing)} expected files missing, e.g. {missing[:5]}"
            )
        if extra:
            raise SystemExit(f"unexpected files in zip, e.g. {extra[:5]}")
        total_rows = 0
        for name in sorted(raw_names):
            sequence = Path(name).stem.split("-DPM")[0].split("-FRCNN")[0].split("-SDP")[0]
            maximum = lengths.get(sequence)
            rows = 0
            with archive.open(name) as handle:
                for line_number, raw in enumerate(handle, start=1):
                    fields = raw.decode().rstrip("\r\n").split(",")
                    if len(fields) != 10:
                        raise SystemExit(
                            f"{name}:{line_number} has {len(fields)} columns, expected 10"
                        )
                    try:
                        values = [float(field) for field in fields]
                    except ValueError:
                        raise SystemExit(
                            f"{name}:{line_number} contains a non-numeric value"
                        ) from None
                    if not all(math.isfinite(value) for value in values):
                        raise SystemExit(
                            f"{name}:{line_number} contains NaN/Inf"
                        )
                    frame = int(values[0])
                    if frame != values[0] or frame < 1:
                        raise SystemExit(
                            f"{name}:{line_number} has invalid frame {values[0]}"
                        )
                    if maximum is not None and frame > maximum:
                        raise SystemExit(
                            f"{name}:{line_number} frame {frame} > seqLength {maximum}"
                        )
                    rows += 1
            if rows == 0:
                raise SystemExit(f"{name} is empty")
            total_rows += rows
    print(
        json.dumps(
            {
                "zip": str(zip_path),
                "files": len(names),
                "rows": total_rows,
                "size_bytes": zip_path.stat().st_size,
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )


# ----------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------
def _reserve_command(args, token: str) -> list[str]:
    zip_path = Path(args.zip)
    return [
        "-X", "POST", f"{args.base_url}/api/datasets/",
        "-H", f"Authorization: Token {token}",
        "-F", f"file_size={zip_path.stat().st_size}",
        "-F", f"name={args.name}",
        "-F", "type=submission",
        "-F", f"request_sassy_file_name={zip_path.name}",
    ]


def _upload_command(sassy_url: str, args) -> list[str]:
    return [
        "-o", "/dev/null", "-w", "%{http_code}",
        "-X", "PUT", sassy_url,
        "-H", "Content-Type: application/zip",
        "--data-binary", f"@{Path(args.zip)}",
    ]


def _create_command(args, token: str, key: str) -> list[str]:
    return [
        "-X", "POST", f"{args.base_url}/api/submissions/",
        "-H", f"Authorization: Token {token}",
        "-F", f"data={key}",
        "-F", f"phase={args.phase}",
        "-F", f"tasks={args.task}",
    ]


def _format_curl(argv: list[str]) -> str:
    import shlex

    return "curl " + " ".join(shlex.quote(part) for part in ["-sS", *argv])


def _print_dry_run(args) -> None:
    print("dry-run: no network calls. Planned requests:")
    print("1) reserve:", _format_curl(_reserve_command(args, "<TOKEN>")))
    print("2) upload: ", _format_curl(_upload_command("<SASSY_URL>", args)))
    print(
        "3) create: ",
        _format_curl(_create_command(args, "<TOKEN>", "<KEY>")),
    )
    print(
        "4) poll:   ",
        _format_curl(
            [
                "-H", "Authorization: Token <TOKEN>",
                f"{args.base_url}/api/submissions/<ID>/",
            ]
        ),
    )


def cmd_submit(args) -> None:
    validate_args = argparse.Namespace(
        zip=args.zip,
        dataset=args.dataset,
        sequences_file=args.sequences_file,
        seqinfo_root=args.seqinfo_root,
    )
    cmd_validate(validate_args)
    if args.dry_run:
        _print_dry_run(args)
        return
    token = _token_from_args(args)
    zip_path = Path(args.zip)

    reserve = _curl(_reserve_command(args, token), timeout=60)
    if MAINTENANCE_MARKER in reserve.stdout:
        raise SystemExit("reserve failed: Codabench is in maintenance")
    try:
        payload = json.loads(reserve.stdout)
    except json.JSONDecodeError:
        raise SystemExit(f"reserve failed: {reserve.stdout[:400]}") from None
    key = payload.get("key")
    sassy_url = payload.get("sassy_url")
    if not key or not sassy_url:
        raise SystemExit(
            f"reserve failed (name may already be reserved): {reserve.stdout[:400]}"
        )
    print(f"reserved data key={key}")

    upload = _curl(_upload_command(sassy_url, args), timeout=600)
    if upload.stdout.strip() != "200":
        raise SystemExit(
            f"upload failed http={upload.stdout.strip()} stderr={upload.stderr[:300]}"
        )
    print("upload ok")

    create = _curl(_create_command(args, token, key), timeout=60)
    submission = json.loads(create.stdout)
    submission_id = submission.get("id")
    if submission_id is None:
        raise SystemExit(f"create submission failed: {create.stdout[:400]}")
    print(
        f"submission id={submission_id} status={submission.get('status')} "
        f"filename={submission.get('filename')}"
    )
    if args.save_record:
        Path(args.save_record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_record).write_text(
            json.dumps({"create": submission}, indent=2, sort_keys=True) + "\n"
        )

    if not args.poll:
        return
    deadline = args.poll_timeout
    waited = 0.0
    while waited < deadline:
        result = _curl(
            [
                "-H", f"Authorization: Token {token}",
                f"{args.base_url}/api/submissions/{submission_id}/",
            ],
            timeout=60,
        )
        final = json.loads(result.stdout)
        status = final.get("status")
        print(f"poll status={status}")
        if status not in ("Submitting", "Running"):
            scores = {
                item.get("column_key"): item.get("score")
                for item in final.get("scores", [])
            }
            summary = {
                "submission_id": submission_id,
                "status": status,
                "phase_name": final.get("phase_name"),
                "is_public": final.get("is_public"),
                "scores": scores,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            if args.save_record:
                Path(args.save_record).parent.mkdir(parents=True, exist_ok=True)
                Path(args.save_record).write_text(
                    json.dumps(final, indent=2, sort_keys=True) + "\n"
                )
            return
        waited += args.poll_interval
        subprocess.run(["sleep", str(args.poll_interval)], check=False)
    raise SystemExit(f"submission {submission_id} still running after {deadline}s")


def cmd_poll(args) -> None:
    token = _token_from_args(args)
    result = _curl(
        [
            "-H", f"Authorization: Token {token}",
            f"{args.base_url}/api/submissions/{args.submission_id}/",
        ],
        timeout=60,
    )
    final = json.loads(result.stdout)
    scores = {
        item.get("column_key"): item.get("score")
        for item in final.get("scores", [])
    }
    print(
        json.dumps(
            {
                "status": final.get("status"),
                "phase_name": final.get("phase_name"),
                "is_public": final.get("is_public"),
                "scores": scores,
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("codabench")
    parser.add_argument("--base-url", default=BASE_URL)
    sub = parser.add_subparsers(dest="command", required=True)

    login = sub.add_parser("login")
    login.add_argument("--username", required=True)
    login.add_argument("--password", default="")
    login.add_argument("--token-file", default="")
    login.set_defaults(func=cmd_login)

    validate = sub.add_parser("validate")
    validate.add_argument("--zip", required=True)
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--sequences-file", default="")
    validate.add_argument("--seqinfo-root", default="")
    validate.set_defaults(func=cmd_validate)

    submit = sub.add_parser("submit")
    submit.add_argument("--zip", required=True)
    submit.add_argument("--competition", required=True)
    submit.add_argument("--phase", required=True)
    submit.add_argument(
        "--task",
        required=True,
        help="phase task-instance id (integer, e.g. 20804); the phase JSON "
        "returns it in phases[].tasks[].id -- the UUID alone is rejected",
    )
    submit.add_argument("--name", required=True)
    submit.add_argument("--dataset", default="mot20")
    submit.add_argument("--sequences-file", default="")
    submit.add_argument("--seqinfo-root", default="")
    submit.add_argument("--token", default="")
    submit.add_argument("--token-file", default="")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--poll", action="store_true")
    submit.add_argument("--poll-interval", type=float, default=30.0)
    submit.add_argument("--poll-timeout", type=float, default=3600.0)
    submit.add_argument("--save-record", default="")
    submit.set_defaults(func=cmd_submit)

    poll = sub.add_parser("poll")
    poll.add_argument("--submission-id", required=True)
    poll.add_argument("--token", default="")
    poll.add_argument("--token-file", default="")
    poll.set_defaults(func=cmd_poll)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
