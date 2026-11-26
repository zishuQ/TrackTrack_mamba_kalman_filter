#!/usr/bin/env python3
"""Build the canonical Codabench submission ledger (full history, our accounts).

Merges the live API listing (/tmp/all_submissions.json, filtered to jojo0 /
shangqingnie) with local submission records, maps known IDs to experiment
labels, and writes .claude/skills/shutrack-pipeline/submissions.{json,md}.
"""
from __future__ import annotations

import calendar
import json
import time
from pathlib import Path

ROOT = Path('/home/shang/workspace/TrackTrack')
SKILL = ROOT / '.claude/skills/shutrack-pipeline'
KEYS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'MOTA', 'IDSW', 'Frag')
PHASE_INFO = {
    'MOT20': {'phase': 16383, 'competition': 10050, 'task': 20804},
    'MOT17': {'phase': 16382, 'competition': 10049, 'task': 20803},
    'SportsMOT': {'phase': 21402, 'competition': 13077, 'task': 26486},
    'DanceTrack': {'phase': 24635, 'competition': None, 'task': 30364},
}
LABELS = {
    1062928: 'SportsMOT formal e100 (night final)',
    1063086: 'MOT20 formal_e100 (sqrt-size)',
    1063098: 'MOT17 formal_e50 (init MOT20 e100)',
    1063982: 'SportsMOT cost trainval formal e80',
    1064153: 'DanceTrack trainval formal e80',
    1064222: 'SportsMOT midpoint selection e80',
    1064223: 'MOT20 midpoint_e20',
    1064234: 'MOT17 midpoint_e20',
    1064320: 'MOT20 e80 round',
    1064329: 'MOT17 e80 (from MOT20 e80, 50 epochs)',
    1064623: 'MOT20 low-LR continuation',
    1064624: 'MOT17 low-LR continuation',
    1064937: 'SportsMOT ablation e80 (train45, D)',
    1066968: 'MOT20 sample-proportional e100',
    1066979: 'MOT17 sample-proportional e50 (from round-1 MOT20)',
    1066994: 'SportsMOT harm=2 E_e100 (post+AFLink)',
    1067021: 'SportsMOT harm=4 train45 e100 (post+AFLink)',
    1068003: 'SportsMOT harm=4 trainval90 e100 (post+AFLink)',
    1068301: 'SportsMOT harm=4 trainval90 e150 (post+AFLink)',
    1068664: 'MOT20 round1 harm=2 dropout=0 (sqrt-size)',
    1068665: 'MOT17 round1 harm=2 dropout=0 (init round1 MOT20)',
    1069017: 'MOT20 round2 harm=1 dropout=0.05 (sqrt-size)',
    1069047: 'MOT17 from r1-MOT20 harm=2 dropout=0.05 (50e)',
    1069361: 'SportsMOT harm=4 trainval90 e100 dropout=0 (post+AFLink)',
}
ACCOUNT_OVERRIDES = {1066994: 'jojo0', 1067021: 'shangqingnie', 1068003: 'jojo0', 1068301: 'shangqingnie', 1068664: 'jojo0', 1068665: 'jojo0', 1069017: 'jojo0', 1069047: 'jojo0', 1069361: 'jojo0'}


def local_records():
    records = {}
    for path in (ROOT / 'outputs/agentguard/experiments').rglob('*_submission.json'):
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        sid = payload.get('id')
        if not sid:
            continue
        scores = payload.get('scores')
        if isinstance(scores, list):
            scores = {item.get('column_key'): item.get('score')
                      for item in scores if isinstance(item, dict)}
        records[int(sid)] = {'record': str(path.relative_to(ROOT)),
                             'status': payload.get('status'),
                             'created_when': payload.get('created_when'),
                             'scores': scores or {}}
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Rebuild the Codabench submission ledger.')
    parser.add_argument('--api-dump', type=Path, default=None,
                        help='Optional JSON dump of /api/submissions/?phase=... per dataset '
                             '(full-history sync). Without it, local *_submission.json records are used.')
    args = parser.parse_args()
    api = json.loads(args.api_dump.read_text()) if args.api_dump and args.api_dump.is_file() else {}
    local = local_records()
    entries = {}
    for dataset, rows in api.items():
        info = PHASE_INFO[dataset]
        for row in rows:
            sid = int(row['id'])
            rec = local.get(sid, {})
            scores = rec.get('scores') or row.get('scores') or {}
            filename = row.get('filename')
            entries[sid] = {
                'id': sid,
                'dataset': dataset,
                'phase': info['phase'],
                'competition': info['competition'],
                'task': info['task'],
                'account': ACCOUNT_OVERRIDES.get(sid) or row.get('account') or rec.get('account'),
                'status': rec.get('status') or row.get('status'),
                'created_when': rec.get('created_when') or row.get('created_when'),
                'experiment': LABELS.get(sid) or f"pre-ledger: {filename or 'unknown'}",
                'filename': filename,
                'scores': scores,
                'record_file': rec.get('record'),
            }
    for sid, rec in local.items():
        if sid in entries:
            continue
        dataset = ('DanceTrack' if 'DanceTrack' in rec['record'] else
                   'SportsMOT' if ('sportsmot' in rec['record'] or 'SportsMOT' in rec['record']) else
                   'MOT20' if 'MOT20' in rec['record'] else 'MOT17')
        info = PHASE_INFO[dataset]
        entries[sid] = {
            'id': sid, 'dataset': dataset, 'phase': info['phase'],
            'competition': info['competition'], 'task': info['task'],
            'account': ACCOUNT_OVERRIDES.get(sid), 'status': rec['status'],
            'created_when': rec['created_when'],
            'experiment': LABELS.get(sid, 'recorded locally'),
            'filename': None, 'scores': rec['scores'], 'record_file': rec['record'],
        }
    ordered = sorted(entries.values(), key=lambda e: (e['created_when'] or '', e['id']))
    for entry in ordered:
        when = entry['created_when']
        entry['created_local'] = None
        if when:
            try:
                stamp = calendar.timegm(time.strptime(when[:19], '%Y-%m-%dT%H:%M:%S'))
                entry['created_local'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(stamp))
            except Exception:  # noqa: BLE001
                pass
    (SKILL / 'submissions.json').write_text(json.dumps({
        'note': ('Every Codabench submission under jojo0/shangqingnie for the four '
                 'tracking competitions (full history). Append new rows after each '
                 'submission resolves; scores are official test scores.'),
        'updated_at': time.strftime('%Y-%m-%d %H:%M'),
        'count': len(ordered),
        'submissions': ordered,
    }, indent=2) + '\n')

    def hota_of(entry):
        value = entry.get('scores', {}).get('HOTA')
        return float(value) if value else None

    dates = sorted({(e.get('created_local') or '')[:10] for e in ordered if e.get('created_local')})
    span = f'{dates[0]} 至 {dates[-1]}' if dates else '-'
    today_utc = time.strftime('%Y-%m-%d', time.gmtime())
    lines = [f'# Codabench 提交台账（官方 test，{span}）', '',
             '每次提交出分后在此追加；`submissions.json` 为机器可读版本。',
             f'范围：本项目 SHUTrack 期间（{span}）jojo0 / shangqingnie 两个账号在 MOT20 / MOT17 / SportsMOT / DanceTrack 的提交。',
             'Failed/Cancelled 的尝试也保留在册（无分数）。', '']
    today_entries = [e for e in ordered if (e.get('created_when') or '')[:10] == today_utc]
    lines += [f'## 今日使用（UTC {today_utc}）', '']
    if today_entries:
        for e in today_entries:
            s = e['scores']
            hota = s.get('HOTA')
            value = f'{float(hota):.3f}' if hota else '-'
            lines.append(f"- #{e['id']} {e['dataset']} {e.get('account') or '-'} — {e['experiment']}（HOTA {value}，{e['status']}）")
    else:
        lines.append('- 今日暂无提交。')
    lines += ['', '提交前先跑 `.claude/skills/codabench/scripts/codabench_cli.py today --phase <id>` 与 `quota` 核对当日使用与额度。', '']
    for dataset in ('SportsMOT', 'MOT20', 'MOT17', 'DanceTrack'):
        rows = [e for e in ordered if e['dataset'] == dataset]
        if not rows:
            continue
        scored = [e for e in rows if hota_of(e) is not None]
        best = max(scored, key=hota_of) if scored else None
        shutrack = [e for e in scored if not e['experiment'].startswith('pre-ledger')]
        best_shutrack = max(shutrack, key=hota_of) if shutrack else None
        info = PHASE_INFO[dataset]
        lines += [f'## {dataset}（phase {info["phase"]}，competition {info["competition"]}）', '']
        if best:
            lines.append(f"台账内最佳：**{hota_of(best):.3f}** — {best['experiment']}（#{best['id']}，{best.get('created_local')}）")
        if best_shutrack and best_shutrack['id'] != (best or {}).get('id'):
            lines.append(f"次佳/其他口径：**{hota_of(best_shutrack):.3f}** — {best_shutrack['experiment']}（#{best_shutrack['id']}）")
        lines += ['',
                  '| 提交 ID | 时间（本地） | 账号 | 实验 | 状态 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |',
                  '|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|']
        for e in rows:
            s = e['scores']
            def fmt(key, digits=3, default='-'):
                value = s.get(key)
                return f'{float(value):.{digits}f}' if value is not None else default
            lines.append(
                f"| {e['id']} | {e.get('created_local') or (e['created_when'] or '')[:10]} | "
                f"{e.get('account') or '-'} | {e['experiment']} | {e['status']} | "
                f"{fmt('HOTA')} | {fmt('AssA')} | {fmt('DetA')} | {fmt('IDF1')} | {fmt('MOTA')} | "
                f"{fmt('IDSW', 0)} |")
        lines.append('')
    (SKILL / 'submissions.md').write_text('\n'.join(lines) + '\n')
    print('wrote', SKILL / 'submissions.json', 'entries:', len(ordered))


if __name__ == '__main__':
    main()
