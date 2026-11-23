#!/usr/bin/env python3
"""Autonomous MOT17/MOT20 tuning loop.

One invocation advances one decision step:
  analyze the last scored candidate -> choose the next candidate from the
  dataset space -> propose it -> optionally run train/test/package/submit and
  poll the official score -> write a daily report -> update the search state.

Dataset-specific axes live in recipes/*_space.json; the policy in this file is
generic. One official submission per dataset per calendar day is enforced.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

SKILL = Path(__file__).resolve().parent
ROOT = SKILL.parents[2]
sys.path.insert(0, str(SKILL))
import mot_pipeline as mp

DAILY = ROOT / "outputs/agentguard/experiments/mot_pipeline_runs/daily"
IMPROVE = 0.02
DEGRADE = -0.02


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def today():
    return time.strftime("%Y-%m-%d")


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def hota(entry):
    return number((entry.get("official_scores") or {}).get("HOTA"))


def experiments(ledger, dataset):
    return ledger["datasets"][dataset]["experiments"]


def load_space(ledger, dataset):
    relative = ledger.get("spaces", {}).get(dataset)
    if not relative:
        raise SystemExit(f"no space recipe for {dataset}")
    path = ROOT / relative
    if not path.is_file():
        path = SKILL / relative
    return json.loads(path.read_text())


def best_experiment(ledger, dataset):
    scored = [(hota(entry), name) for name, entry in experiments(ledger, dataset).items()
              if hota(entry) is not None]
    return max(scored) if scored else None


def best_training_experiment(ledger, dataset):
    scored = [(hota(entry), name) for name, entry in experiments(ledger, dataset).items()
              if hota(entry) is not None and entry.get("kind") in mp.TRAINING_MODES]
    return max(scored) if scored else None


def submissions_today(ledger, dataset):
    count = 0
    for entry in experiments(ledger, dataset).values():
        for submission in (entry.get("submissions") or {}).values():
            if str(submission.get("submitted_at", "")).startswith(today()):
                count += 1
    return count


def pending_experiments(ledger, dataset):
    return [name for name, entry in experiments(ledger, dataset).items()
            if any(sub.get("status") in mp.NON_TERMINAL
                   for sub in (entry.get("submissions") or {}).values())]


def poll_pending(ledger, dataset, name):
    entry = experiments(ledger, dataset)[name]
    token_files = dict(mp.TOKENS)
    for account, submission in (entry.get("submissions") or {}).items():
        if submission.get("status") not in mp.NON_TERMINAL:
            continue
        token_file = token_files.get(account)
        if not token_file or not token_file.is_file():
            continue
        record = mp.experiment_dir(name) / "submissions" / f"{dataset}_submission.json"
        payload = mp._poll(token_file.read_text().strip(), submission["id"], record, 2700)
        mp.self_record(ledger, dataset, name, account, payload, record)


def quota(ledger, dataset):
    protocol = ledger["datasets"][dataset]["protocol"]
    result = {}
    for account, token_file in mp.TOKENS:
        if not token_file.is_file():
            result[account] = {"can": False, "reason": "missing token"}
            continue
        try:
            payload = mp._curl_json(token_file.read_text().strip(),
                                    f"/api/can_make_submission/{protocol['phase']}/")
            result[account] = {"can": bool(payload.get("can")),
                               "reason": payload.get("reason")}
        except Exception as exc:  # noqa: BLE001
            result[account] = {"can": None, "reason": str(exc)[:150]}
    return result


def axis_magnitudes(axis):
    values = []
    for step in axis.get("steps", []):
        magnitude = abs(float(step))
        if magnitude not in values:
            values.append(magnitude)
    return sorted(values)


def axis_slots(axis):
    if axis.get("type") == "choice":
        return axis.get("values", [])
    if axis["kind"] == "continue":
        return axis.get("lr_values", [])
    if axis["kind"] == "reproduce":
        return [axis.get("seed", 43)]
    return axis_magnitudes(axis)


def advance_axis(search):
    search["axis_index"] = int(search.get("axis_index", 0)) + 1
    search["step_index"] = 0
    search["direction"] = 1
    search["tried_both"] = False


def advance_after_result(axis, search, verdict):
    """Refine an effective direction, reverse a failing one, switch when done."""
    kind = axis["kind"]
    if kind in ("choice", "continue", "reproduce"):
        step_index = int(search.get("step_index", 0)) + 1
        search["step_index"] = step_index
        search["tried_both"] = False
        search["direction"] = 1
        if step_index >= len(axis_slots(axis)):
            advance_axis(search)
    elif verdict == "improve":
        step_index = int(search.get("step_index", 0)) + 1
        search["step_index"] = step_index
        search["tried_both"] = False
        if step_index >= len(axis_slots(axis)):
            advance_axis(search)
    elif verdict == "degrade" and not search.get("tried_both"):
        search["direction"] = -(search.get("direction") or 1)
        search["tried_both"] = True
    else:
        step_index = int(search.get("step_index", 0)) + 1
        search["step_index"] = step_index
        search["tried_both"] = False
        search["direction"] = 1
        if step_index >= len(axis_slots(axis)):
            advance_axis(search)
    search["updated_at"] = now()


def analyze_last(ledger, dataset, search, space):
    last = search.get("last_experiment")
    if not last or search.get("last_analyzed") == last:
        return None
    entry = experiments(ledger, dataset).get(last)
    if not entry:
        return None
    score = hota(entry)
    if score is None:
        return None
    baseline = number(search.get("baseline_hota"))
    delta = None if baseline is None else score - baseline
    verdict = ("improve" if delta is not None and delta > IMPROVE
               else "degrade" if delta is not None and delta < DEGRADE
               else "neutral")
    secondary = {}
    base_entry = experiments(ledger, dataset).get(search.get("baseline_experiment") or "")
    if base_entry:
        for key in ("AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag"):
            try:
                secondary[key] = (float(entry["official_scores"][key])
                                  - float(base_entry["official_scores"][key]))
            except (KeyError, TypeError, ValueError):
                continue
    result = {"experiment": last, "hota": score, "baseline_hota": baseline,
              "delta": delta, "verdict": verdict, "secondary": secondary,
              "axis": search.get("pending_axis"), "change": search.get("pending_change"),
              "analyzed_at": now()}
    search["last_result"] = result
    search["last_analyzed"] = last
    search.setdefault("history", []).append(result)
    axes = space["axes"]
    if int(search.get("axis_index", 0)) < len(axes):
        advance_after_result(axes[int(search.get("axis_index", 0))], search, verdict)
    search["updated_at"] = now()
    return result


def _inference_value(axis, base_value, step_index, direction):
    slots = axis_slots(axis)
    if step_index >= len(slots):
        return None
    if axis.get("type") == "choice":
        return slots[step_index]
    return round(float(base_value) + (direction or 1) * float(slots[step_index]), 4)


def candidate_name(dataset, axis, step_index, direction):
    suffix = f"p1" if direction > 0 else "m1"
    return f"auto_{dataset.lower()}_{axis['name']}_{today().replace('-', '')}_{suffix}_s{step_index}"


def build_candidate(ledger, dataset, space, search):
    """Pick the next untested candidate; mutates search's axis/step state."""
    best = best_experiment(ledger, dataset)
    if not best:
        raise SystemExit("no official score in the ledger yet")
    best_hota, best_name = best
    best_entry = experiments(ledger, dataset)[best_name]
    training = best_training_experiment(ledger, dataset)
    training_name = training[1] if training else best_name
    axes = space["axes"]
    tried = {entry.get("config_hash") for entry in experiments(ledger, dataset).values()
             if entry.get("config_hash") and entry.get("status") != "proposed"}
    attempts = 0
    while int(search.get("axis_index", 0)) < len(axes) and attempts < 100:
        attempts += 1
        axis = axes[int(search.get("axis_index", 0))]
        kind = axis["kind"]
        step_index = int(search.get("step_index", 0))
        direction = search.get("direction", 1) or 1
        candidate = None
        if kind == "inference":
            base_inference = dict(best_entry.get("inference") or space["baseline_inference"])
            param = axis["param"]
            value = _inference_value(axis, base_inference.get(param), step_index, direction)
            if value is None:
                advance_axis(search)
                continue
            if value == base_inference.get(param):
                search["step_index"] = step_index + 1
                continue
            if param in ("det_thr", "init_thr", "match_thr", "tai_thr") and not (0.05 <= float(value) <= 0.95):
                advance_axis(search)
                continue
            inference = dict(base_inference)
            inference[param] = value
            config = {"dataset": dataset, "kind": "inference",
                      "checkpoint_sha": mp.sha256(ROOT / best_entry["checkpoint"])[:32],
                      "inference": dict(sorted(inference.items()))}
            digest = mp.config_hash(config)
            if digest in tried:
                search["step_index"] = step_index + 1
                continue
            candidate = {
                "kind": "inference", "base": best_name, "digest": digest,
                "set": [f"{param}={value}"],
                "change": f"{param} {base_inference.get(param)} -> {value}",
                "checkpoint": best_entry["checkpoint"], "config": config}
        elif kind == "continue":
            slots = axis_slots(axis)
            if training is None or step_index >= len(slots):
                advance_axis(search)
                continue
            lr = float(slots[step_index])
            params = {"mode": "continue", "epochs": axis.get("epochs", 50),
                      "shards": space["training_defaults"]["shards"], "lr": lr,
                      "seed": 42, "batch_size": 1024,
                      "source": best_entry["checkpoint"],
                      "allow_partial_cycles": dataset == "MOT20"}
            config = mp.canonical_config(dataset, params)
            digest = mp.config_hash(config)
            if digest in tried:
                search["step_index"] = step_index + 1
                continue
            candidate = {
                "kind": "continue", "base": training_name, "digest": digest,
                "set": ["mode=continue", f"source={best_entry['checkpoint']}",
                        f"lr={lr}", f"epochs={params['epochs']}",
                        f"shards={params['shards']}"],
                "change": f"continue from {best_entry['checkpoint']} lr {lr} for {params['epochs']} epochs",
                "checkpoint": best_entry["checkpoint"], "config": config}
        elif kind == "reproduce":
            if training is None or step_index > 0:
                advance_axis(search)
                continue
            params = mp.experiment_params(experiments(ledger, dataset)[training_name])
            params["seed"] = int(axis.get("seed", 43))
            config = mp.canonical_config(dataset, params)
            digest = mp.config_hash(config)
            if digest in tried:
                search["step_index"] = step_index + 1
                continue
            candidate = {
                "kind": "reproduce", "base": training_name, "digest": digest,
                "set": [f"seed={params['seed']}"],
                "change": f"reproduce best training recipe with seed {params['seed']}",
                "checkpoint": None, "config": config}
        if candidate is None:
            advance_axis(search)
            continue
        candidate.update({"axis_index": int(search.get("axis_index", 0)),
                          "step_index": step_index, "direction": direction,
                          "axis": axis, "best_name": best_name, "best_hota": best_hota})
        return candidate
    return None


def rationale(candidate, search):
    axis = candidate["axis"]
    last = search.get("last_result") or {}
    last_text = ""
    if last:
        delta = last.get("delta")
        if delta is not None:
            last_text = (f" Last candidate {last['experiment']} scored {last['hota']:.3f} "
                         f"({delta:+.3f} vs its baseline, {last['verdict']}).")
    return (f"Best {candidate['best_name']} HOTA {candidate['best_hota']:.3f}.{last_text} "
            f"Axis {axis['name']}: {axis.get('hypothesis', '')} "
            f"Change: {candidate['change']}. "
            f"Expected: {axis.get('expected', 'official HOTA decides.')}")


def next_direction(ledger, dataset, space, search):
    probe = copy.deepcopy(search)
    probe.pop("pending_state", None)
    try:
        candidate = build_candidate(ledger, dataset, space, probe)
    except SystemExit:
        return "no scored experiment yet; submit a baseline before the loop can move"
    if candidate:
        return f"next candidate would change {candidate['change']} (axis {candidate['axis']['name']})"
    return "space exhausted; add a hypothesis or extend the dataset space"


def result_analysis(result):
    verdict = result.get("verdict")
    delta = result.get("delta")
    secondary = result.get("secondary") or {}
    parts = []
    if delta is not None:
        parts.append(f"HOTA {'rose' if delta > 0 else 'fell'} {abs(delta):.3f} against the "
                     f"candidate's baseline, so the change is [{verdict}] under the "
                     f"±{abs(IMPROVE):.2f} decision band.")
    assa, deta = secondary.get("AssA"), secondary.get("DetA")
    if assa is not None and deta is not None:
        if assa > 0 and abs(assa) >= abs(deta):
            parts.append("The gain is association-driven (AssA up at least as much as DetA).")
        elif deta > 0 and abs(deta) > abs(assa):
            parts.append("The gain is detection-driven (DetA moved more than AssA).")
        elif assa < 0 and deta < 0:
            parts.append("Both AssA and DetA fell, so the setting costs matches, not just IDs.")
    idsw = secondary.get("IDSW")
    if idsw is not None:
        parts.append(f"ID switches {'decreased' if idsw < 0 else 'increased'} by {abs(idsw):.0f}.")
    frag = secondary.get("Frag")
    if frag is not None and abs(frag) > 0:
        parts.append(f"Fragmentation {'improved' if frag < 0 else 'worsened'} by {abs(frag):.0f}.")
    return " ".join(parts) or "No secondary metrics recorded."


def write_daily(dataset, space, search, ledger, candidate=None, result=None, note=""):
    DAILY.mkdir(parents=True, exist_ok=True)
    path = DAILY / f"{today()}_{dataset}.md"
    best = best_experiment(ledger, dataset)
    lines = [f"# Daily tuning report — {dataset} — {today()}", ""]
    if best:
        lines.append(f"- official best: {best[1]} HOTA {best[0]:.3f}")
    lines.append(f"- daily submissions used today: {submissions_today(ledger, dataset)}")
    if note:
        lines += ["", f"> {note}"]
    history = search.get("history") or []
    if history:
        lines += ["", "## Cumulative evidence", "",
                  "| experiment | axis | change | HOTA | delta | verdict |",
                  "|---|---|---|---:|---:|---|"]
        for item in history[-10:]:
            delta = item.get("delta")
            lines.append(f"| {item['experiment']} | {item.get('axis') or '-'} | "
                         f"{item.get('change') or '-'} | {item['hota']:.3f} | "
                         f"{'-' if delta is None else f'{delta:+.3f}'} | {item['verdict']} |")
    if candidate:
        lines += ["", "## Candidate and hypothesis", "",
                  f"- experiment: `{candidate['name']}` (kind {candidate['kind']})",
                  f"- rationale: {candidate.get('rationale', '')}",
                  f"- expected effect: {candidate['axis'].get('expected', '-')}",
                  f"- config: `{json.dumps(candidate.get('config'), ensure_ascii=False)}`"]
    if result:
        delta = result.get("delta")
        lines += ["", "## Official result", "",
                  f"- {result['experiment']}: HOTA {result['hota']:.3f}",
                  f"- delta vs its baseline: {'-' if delta is None else f'{delta:+.3f}'}",
                  f"- verdict: {result['verdict']}"]
        if result.get("secondary"):
            lines.append("- secondary deltas: " +
                         ", ".join(f"{key} {value:+.3f}" for key, value in result["secondary"].items()))
        lines += ["", "## Result analysis", "", f"- {result_analysis(result)}"]
    if candidate and not result:
        direction = (f"execute `{candidate['name']}` (next daily slot if today's budget is used), "
                     "then analyze its official HOTA")
    else:
        direction = next_direction(ledger, dataset, space, search)
    lines += ["", "## Next direction", "", f"- {direction}"]
    path.write_text("\n".join(lines) + "\n")
    return path


def propose(dataset, candidate, reason):
    command = [str(mp.PY), str(SKILL / "mot_pipeline.py"), "propose",
               "--dataset", dataset, "--experiment", candidate["name"],
               "--base", candidate["base"], "--reason", reason]
    for item in candidate["set"]:
        command += ["--set", item]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"propose failed: {result.stderr[:400]}")
    return json.loads(mp.LEDGER.read_text())["datasets"][dataset]["experiments"][candidate["name"]]


def run_experiment(dataset, name):
    command = [str(mp.PY), str(SKILL / "mot_pipeline.py"), "run",
               "--dataset", dataset, "--experiment", name,
               "--stages", "train,test,package,submit", "--execute"]
    if subprocess.call(command):
        raise SystemExit(f"run failed for {name}")


def pending_candidate(ledger, dataset, search):
    name = search.get("pending")
    if not name:
        return None
    entry = experiments(ledger, dataset).get(name)
    if not entry or entry.get("status") != "proposed":
        return None
    if search.get("pending_state") != {
            key: search.get(key) for key in ("axis_index", "step_index", "direction", "baseline_hota")}:
        return None
    axis_name = search.get("pending_axis") or "pending"
    return {"name": name, "kind": entry.get("kind"), "base": entry.get("base_experiment"),
            "set": [], "change": search.get("pending_change") or entry.get("change_reason", ""),
            "rationale": entry.get("change_reason", ""), "config": entry.get("config"),
            "axis": {"name": axis_name, "expected": ""},
            "best_name": search.get("baseline_experiment"), "best_hota": search.get("baseline_hota")}


def tuner(args):
    ledger = mp.load_ledger()
    dataset = args.dataset
    if dataset not in ledger["datasets"] or dataset not in ledger.get("spaces", {}):
        raise SystemExit(f"{dataset}: no protocol/space in the ledger; "
                         f"available: {sorted(ledger.get('spaces', {}))}")
    space = load_space(ledger, dataset)
    search = ledger["datasets"][dataset].setdefault("search", {})

    pending = pending_experiments(ledger, dataset)
    if pending:
        if not args.execute:
            print(json.dumps({"pending": pending, "action": "poll with --execute"}, indent=2))
            return
        for name in pending:
            poll_pending(ledger, dataset, name)
        result = analyze_last(ledger, dataset, search, space)
        mp.save_ledger(ledger)
        path = write_daily(dataset, space, search, ledger, result=result,
                           note="polled pending submission")
        print(json.dumps({"pending_polled": pending, "result": result,
                          "report": str(path)}, indent=2, ensure_ascii=False))
        return

    result = analyze_last(ledger, dataset, search, space)
    mp.save_ledger(ledger)
    if result:
        print(json.dumps({"analysis": result}, indent=2, ensure_ascii=False))

    budget_used = submissions_today(ledger, dataset) >= 1
    if args.execute and budget_used:
        path = write_daily(dataset, space, search, ledger, result=result,
                           note="daily submission budget already used; no action")
        print(json.dumps({"status": "daily budget used", "report": str(path)}, indent=2))
        return

    accounts = quota(ledger, dataset) if args.execute else {}
    if args.execute and not any(value.get("can") for value in accounts.values()):
        path = write_daily(dataset, space, search, ledger, result=result,
                           note=f"no submission quota: {accounts}")
        print(json.dumps({"status": "no quota", "quota": accounts, "report": str(path)},
                         indent=2, ensure_ascii=False))
        return

    candidate = pending_candidate(ledger, dataset, search) or build_candidate(ledger, dataset, space, search)
    if not candidate:
        path = write_daily(dataset, space, search, ledger, result=result,
                           note="search space exhausted")
        print(json.dumps({"status": "exhausted", "report": str(path)}, indent=2))
        return

    if "name" not in candidate:
        axis = candidate["axis"]
        name = candidate_name(dataset, axis, candidate["step_index"], candidate["direction"])
        suffix = 0
        while name in experiments(ledger, dataset):
            suffix += 1
            name = candidate_name(dataset, axis, candidate["step_index"], candidate["direction"]) + f"_{suffix}"
        candidate["name"] = name
        candidate["rationale"] = rationale(candidate, search)

    if not args.execute:
        note = "dry run; pass --execute to run and submit"
        if budget_used:
            note += " (daily submission budget already used; execution would wait until tomorrow)"
        path = write_daily(dataset, space, search, ledger, candidate=candidate,
                           result=result, note=note)
        print(json.dumps({"candidate": {key: candidate.get(key) for key in
                                        ("name", "kind", "base", "set", "change", "rationale")},
                          "report": str(path)}, indent=2, ensure_ascii=False))
        return

    existing = experiments(ledger, dataset).get(candidate["name"])
    if existing and existing.get("status") == "proposed":
        entry = existing
    else:
        entry = propose(dataset, candidate, candidate["rationale"])
        candidate["config"] = entry.get("config")

    search["baseline_experiment"] = candidate["best_name"]
    search["baseline_hota"] = candidate["best_hota"]
    search["last_experiment"] = candidate["name"]
    search["pending"] = candidate["name"]
    search["pending_axis"] = candidate["axis"]["name"]
    search["pending_change"] = candidate["change"]
    search["pending_state"] = {key: search.get(key) for key in
                               ("axis_index", "step_index", "direction", "baseline_hota")}
    search["updated_at"] = now()
    mp.save_ledger(ledger)
    print(json.dumps({"executing": candidate["name"], "config": entry.get("config")},
                     indent=2, ensure_ascii=False))

    run_experiment(dataset, candidate["name"])
    ledger = mp.load_ledger()
    search = ledger["datasets"][dataset]["search"]
    if candidate["name"] in pending_experiments(ledger, dataset):
        poll_pending(ledger, dataset, candidate["name"])
    result = analyze_last(ledger, dataset, search, space)
    search.pop("pending", None)
    search.pop("pending_state", None)
    mp.save_ledger(ledger)
    path = write_daily(dataset, space, search, ledger, candidate=candidate,
                       result=result, note="candidate executed")
    print(json.dumps({"experiment": candidate["name"], "result": result,
                      "report": str(path)}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="dataset key present in the ledger (MOT20, MOT17, ...)")
    parser.add_argument("--execute", action="store_true",
                        help="run and submit the selected candidate; default is a dry run")
    args = parser.parse_args()
    tuner(args)


if __name__ == "__main__":
    main()
