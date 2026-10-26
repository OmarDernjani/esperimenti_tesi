"""Audit existing scored preference pairs; never read test preferences or code."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from ..common import digest, read_jsonl, write_jsonl


def select_pairs(tasks, rows, split, target, minimum_margin=0.1, pairs_per_task=1):
    ids = {t['task_id']: t for t in tasks}
    groups, excluded, seen = defaultdict(list), [], set()
    for row in rows:
        task = ids.get(row['task_id'])
        if task is None or task['split'] != split or row['split'] != split:
            raise ValueError(f'Split leakage or unknown task: {row["task_id"]}')
        if row['input'] != task['problem']:
            raise ValueError(f'Problem differs from canonical task: {row["task_id"]}')
        if row['target_model'] != target:
            continue
        good, bad = row['chosen_dev_score'], row['rejected_dev_score']
        if not all(isinstance(s, (float, int)) and math.isfinite(s) and 0 <= s <= 1 for s in (good, bad)):
            raise ValueError('Invalid preference score')
        reason = None
        if not good > bad or good - bad < minimum_margin:
            reason = 'tie_or_small_margin'
        elif not row.get('chosen_evaluation_ids') or not row.get('rejected_evaluation_ids'):
            raise ValueError('Preference lacks evaluation provenance')
        elif not row.get('comparison_group_id'):
            raise ValueError('Preference lacks comparable evaluation group')
        elif not row['chosen'].strip() or not row['rejected'].strip():
            reason = 'empty_completion'
        elif row['input'] not in row['chosen']:
            reason = 'chosen_does_not_preserve_problem'
        elif row['chosen'] == row['rejected']:
            reason = 'identical_completions'
        signature = (row['task_id'], row['chosen'], row['rejected'])
        if signature in seen:
            reason = 'duplicate_pair'
        if reason:
            excluded.append({'sample_id': row['sample_id'], 'reason': reason})
            continue
        seen.add(signature)
        groups[row['task_id']].append(dict(row, score_margin=good - bad))
    selected = []
    for task_id in sorted(groups):
        ranked = sorted(groups[task_id], key=lambda r: (-r['score_margin'], r['sample_id']))
        selected.extend(ranked[:pairs_per_task])
        excluded.extend({'sample_id': r['sample_id'], 'reason': 'per_task_cap'} for r in ranked[pairs_per_task:])
    return selected, excluded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='CodePromptOpt/data')
    p.add_argument('--config', default='CodePromptOpt/finetuning/dpo/qwen3_8b.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if cfg['pairs_per_task'] < 1 or not 0 <= cfg['minimum_margin'] <= 1:
        raise ValueError('Invalid pair selection settings')
    root, out = Path(args.data), Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Prepared dataset already exists; choose a new output directory')
    tasks = read_jsonl(root / 'tasks.jsonl')
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate canonical task IDs')
    parts, rejected, report = {}, [], {'config': cfg, 'files': {'tasks.jsonl': digest(root / 'tasks.jsonl')}, 'splits': {}}
    for split in ('train', 'validation'):
        source = root / 'dpo' / f'{split}.jsonl'
        rows = read_jsonl(source)
        parts[split], dropped = select_pairs(tasks, rows, split, cfg['target_label'], cfg['minimum_margin'], cfg['pairs_per_task'])
        rejected.extend(dict(r, split=split) for r in dropped)
        report['files'][str(source)] = digest(source)
        report['splits'][split] = {'source_pairs': len(rows), 'selected_pairs': len(parts[split]),
                                  'tasks': len({r['task_id'] for r in parts[split]}),
                                  'excluded': dict(Counter(r['reason'] for r in dropped))}
        if not parts[split]:
            raise ValueError(f'No usable preferences in {split}')
    for split, rows in parts.items():
        write_jsonl(out / f'{split}.jsonl', rows)
        report['files'][f'{split}.jsonl'] = digest(out / f'{split}.jsonl')
    write_jsonl(out / 'excluded.jsonl', rejected)
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['splits'], indent=2))


if __name__ == '__main__':
    main()
