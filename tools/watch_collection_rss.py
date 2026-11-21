#!/usr/bin/env python3
"""Sample RSS of running paired-data generators (read-only /proc scraper).

Writes one CSV row per live generate_trainval_paired.py process per interval,
plus host MemAvailable. Exits after no generator has been seen for
--idle-exit-min minutes, reporting the observed peak in the final row.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

NEEDLE = "generate_trainval_paired.py"


def rss_fields(pid: int) -> tuple[float, float] | None:
    try:
        with open(f"/proc/{pid}/status") as handle:
            rss = anon = None
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith("RssAnon:"):
                    anon = int(line.split()[1]) / (1024 * 1024)
            if rss is None:
                return None
            return rss, (anon or 0.0)
    except OSError:
        return None


def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


def generators() -> list[int]:
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as handle:
                cmd = handle.read().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if NEEDLE in cmd:
            pids.append(int(entry))
    return pids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--idle-exit-min", type=float, default=30.0)
    parser.add_argument("--max-hours", type=float, default=12.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.is_file() or out.stat().st_size == 0
    peak = 0.0
    peak_pid = 0
    last_seen = time.time()
    started = time.time()
    with out.open("a", buffering=1) as handle:
        if write_header:
            handle.write("time,pid,vmrss_gb,rss_anon_gb,mem_available_gb\n")
        while True:
            now = time.time()
            if now - started > args.max_hours * 3600:
                break
            pids = generators()
            if pids:
                last_seen = now
            for pid in pids:
                fields = rss_fields(pid)
                if fields is None:
                    continue
                rss, anon = fields
                if rss > peak:
                    peak, peak_pid = rss, pid
                handle.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')},{pid},"
                    f"{rss:.3f},{anon:.3f},{mem_available_gb():.2f}\n"
                )
            if not pids and (now - last_seen) > args.idle_exit_min * 60:
                break
            time.sleep(args.interval)
        handle.write(
            f"# peak_vmrss_gb={peak:.3f},pid={peak_pid},"
            f"finished_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )


if __name__ == "__main__":
    main()
