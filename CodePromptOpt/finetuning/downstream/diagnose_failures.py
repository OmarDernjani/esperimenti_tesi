"""Select unique failed requests for a focused sandbox rerun, preserving old scores."""
import argparse
from collections import Counter
import json
from pathlib import Path

from ..common import read_jsonl, write_jsonl
from .run import atomic_json


def select(root, output, tasks_path, variants, benchmark):
    tasks = {r['task_id']: r for r in read_jsonl(tasks_path)}
    output = Path(output)
    if output.exists():
        raise ValueError('Diagnostic output already exists; choose a new --output')
    selected, links = {}, []
    for path in sorted(Path(root).glob('*/*-results.jsonl')):
        for row in read_jsonl(path):
            if row['passed'] or row['variant'] not in variants or tasks[row['task_id']]['benchmark'] != benchmark:
                continue
            key = row['codegen_request_id']
            if key in selected and (selected[key]['code'] != row['code'] or selected[key]['task_id'] != row['task_id']):
                raise ValueError('Inconsistent reused request')
            selected[key] = {k: v for k, v in row.items() if k not in (
                'passed', 'timeout', 'returncode', 'diagnostic', 'job_id', 'test_image', 'test_timeout', 'tests_sha256')}
            links.append({'request_id': key, 'source_result': str(path), 'task_id': row['task_id']})
    if not selected:
        raise ValueError('No matching failures found')
    for key, row in selected.items():
        write_jsonl(output / key / 'diagnostic-code.jsonl', [row])
    manifest = {'requests': len(selected), 'original_failed_rows': len(links), 'benchmark': benchmark,
                'variants': variants, 'links': links}
    atomic_json(output / 'selection.json', manifest)
    return manifest


def summarize(root, output):
    records, counts = [], Counter()
    for path in sorted(Path(root).glob('*/*-results.jsonl')):
        for row in read_jsonl(path):
            info = row.get('diagnostic', {})
            category = info.get('category', 'diagnostics_missing')
            counts[category] += 1
            records.append({'task_id': row['task_id'], 'variant': row['variant'],
                'request_id': row['codegen_request_id'], 'passed': row['passed'], 'category': category,
                'stderr': info.get('stderr', {}).get('tail', ''), 'file': str(path)})
    if not records:
        raise ValueError('No diagnostic results found')
    report = {'requests': len(records), 'categories': dict(counts), 'records': records}
    atomic_json(output, report)
    return {'requests': len(records), 'categories': dict(counts), 'report': str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'summarize'])
    p.add_argument('--root', default='CodePromptOpt/finetuning/outputs/codegen/qwen25-coder-7b-h200-v1/validation')
    p.add_argument('--output', default='CodePromptOpt/finetuning/outputs/diagnostics/apps-controls-v1')
    p.add_argument('--tasks', default='CodePromptOpt/data/tasks.jsonl')
    p.add_argument('--variants', nargs='+', default=['original', 'static'])
    p.add_argument('--benchmark', default='APPS')
    args = p.parse_args()
    if args.stage == 'prepare':
        report = select(args.root, args.output, args.tasks, args.variants, args.benchmark)
        print(json.dumps({k: v for k, v in report.items() if k != 'links'}, indent=2))
    else:
        print(json.dumps(summarize(args.root, args.output), indent=2))


if __name__ == '__main__':
    main()
