"""Recover pinned benchmark sources and build task-matched sandbox harnesses."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

from ..common import digest, read_jsonl, write_jsonl
from .run import atomic_json

SOURCES = {
    'humanevalplus': ('evalplus/humanevalplus', 'd32357cf319e50e9c8d8dab5ea876c72b0fd321b', 'test.jsonl'),
    'apps': ('codeparrot/apps', '21e74ddf8de1a21436da12e3e653065c5213e9d1', 'train.jsonl'),
}


def recover(tasks, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    he_ids = {t['source_id'] for t in tasks if t['benchmark'] == 'HumanEval+'}
    apps_questions = {t['problem'] for t in tasks if t['benchmark'] == 'APPS'}
    for kind, (dataset, revision, filename) in SOURCES.items():
        url = f'https://huggingface.co/datasets/{dataset}/resolve/{revision}/{filename}'
        matches = []
        print(f'Reading pinned {dataset}/{filename}; only matched records are saved', flush=True)
        with urllib.request.urlopen(url, timeout=180) as response:
            for raw in response:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if kind == 'humanevalplus' and row.get('task_id') in he_ids:
                    retained = {k: row[k] for k in ('task_id', 'prompt', 'entry_point', 'test')}
                elif kind == 'apps' and row.get('question') in apps_questions:
                    retained = {k: row.get(k) for k in ('problem_id', 'question', 'input_output', 'difficulty', 'url')}
                else:
                    continue
                matches.append({**retained, '_provenance': {'dataset': dataset, 'revision': revision,
                    'file': filename, 'url': url, 'source_record_sha256': hashlib.sha256(raw).hexdigest()}})
        write_jsonl(directory / f'{kind}.jsonl', matches)
        print(f'{kind}: {len(matches)} matched source records', flush=True)


APPS_HARNESS = '''import ast
import json
import math
import subprocess
import sys

DATA = json.loads(DATA_JSON)

def normalize_value(value):
    if isinstance(value, (tuple, list)):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted([normalize_value(v) for v in value], key=repr)
    return value

def normalize_stdout(value):
    return '\\n'.join(line.rstrip() for line in value.splitlines()).strip()

CALL_RUNNER = """import ast, importlib, os, sys
sys.path.insert(0, os.getcwd())
payload = ast.literal_eval(sys.stdin.read())
module = importlib.import_module('solution')
fn = getattr(module, payload['fn_name'], None)
if fn is None:
    fn = getattr(module.Solution(), payload['fn_name'])
args = payload['args']
if not isinstance(args, list):
    args = [args]
print(repr(fn(*args)))
"""

for case_input, expected in zip(DATA['inputs'], DATA['outputs']):
    if DATA.get('fn_name'):
        completed = subprocess.run([sys.executable, '-B', '-c', CALL_RUNNER],
            input=repr({'fn_name': DATA['fn_name'], 'args': case_input}), text=True,
            capture_output=True, timeout=10, check=True)
        actual = ast.literal_eval(completed.stdout.strip().splitlines()[-1])
        # APPS wraps each call's return value in an outer singleton list.
        if isinstance(expected, list) and len(expected) == 1:
            expected = expected[0]
        assert normalize_value(actual) == expected, 'call-based test failed'
    else:
        assert isinstance(case_input, str) and isinstance(expected, str), 'invalid stdin/stdout fixture'
        completed = subprocess.run([sys.executable, '-B', 'solution.py'],
            input=case_input, text=True, capture_output=True, timeout=10, check=True)
        assert normalize_stdout(completed.stdout) == normalize_stdout(expected), 'stdout mismatch'
'''


def make_apps_harness(io_data):
    if not io_data.get('inputs') or len(io_data['inputs']) != len(io_data.get('outputs', [])):
        raise ValueError('Empty or unaligned APPS tests')
    return APPS_HARNESS.replace('DATA_JSON', repr(json.dumps(io_data, ensure_ascii=True)), 1)


def build(tasks, directory, output):
    directory = Path(directory)
    he = read_jsonl(directory / 'humanevalplus.jsonl')
    apps = read_jsonl(directory / 'apps.jsonl')
    rows, mapping, issues = [], [], []
    for task in tasks:
        try:
            if task['benchmark'] == 'HumanEval+':
                candidates = [r for r in he if r['task_id'] == task['source_id']]
                if len(candidates) != 1:
                    raise ValueError('HumanEval+ source ID not unique/found')
                source = candidates[0]
                if source['prompt'].strip() != task['problem'].strip() or source['entry_point'] != task['entry_point']:
                    raise ValueError('HumanEval+ source prompt/signature mismatch')
                tree = ast.parse(source['test'])
                if not any(isinstance(n, ast.FunctionDef) and n.name == 'check' for n in tree.body):
                    raise ValueError('HumanEval+ check function missing')
                # Source dataset tests are executed only later inside the sandbox.
                harness = ('import solution\n' + source['test'] +
                           f'\ncheck(getattr(solution, {task["entry_point"]!r}))\n')
                io_mode, count = 'official check(candidate)', None
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inputs' for t in node.targets):
                        try:
                            count = len(ast.literal_eval(node.value))
                        except (ValueError, TypeError):
                            pass
                source_id = source['task_id']
            elif task['benchmark'] == 'APPS':
                candidates = [r for r in apps if r['question'] == task['problem']]
                if len(candidates) != 1:
                    raise ValueError('APPS original question not uniquely matched')
                source = candidates[0]
                data = json.loads(source['input_output']) if isinstance(source['input_output'], str) else source['input_output']
                harness = make_apps_harness(data)
                io_mode = 'call-based' if data.get('fn_name') else 'stdin/stdout'
                count, source_id = len(data['inputs']), source.get('problem_id') or source.get('url')
            else:
                raise ValueError('Unsupported benchmark')
            ast.parse(harness)  # Syntax validation only; do not execute source tests here.
            provenance = source['_provenance']
            record = {'task_id': task['task_id'], 'split': task['split'], 'benchmark': task['benchmark'],
                      'source_id': source_id, 'mapping': 'verified ID + prompt' if task['benchmark'] == 'HumanEval+' else 'exact unique question',
                      'io_mode': io_mode, 'test_cases': count, 'provenance': provenance,
                      'runtime_requires': ['numpy'] if task['benchmark'] == 'HumanEval+' else []}
            rows.append({**record, 'harness': harness})
            mapping.append(record)
        except (ValueError, KeyError, TypeError, SyntaxError) as error:
            issues.append({'task_id': task['task_id'], 'error': str(error)})
    report = {'requested_tasks': len(tasks), 'matched_tasks': len(rows), 'issues': issues, 'mapping': mapping,
              'scope': 'All tests in pinned source records; not reconstruction of historical internal dev/test splits',
              'apps_comparison': 'Historical utils.py semantics: trailing-line-whitespace normalization; singleton-output unwrapping; tuple/set normalization; exact equality',
              'source_hashes': {name: digest(directory / name) for name in ('humanevalplus.jsonl', 'apps.jsonl')}}
    atomic_json(str(output) + '.mapping.json', report)
    if issues:
        raise ValueError('Incomplete verified mapping: see .mapping.json; no test file written')
    write_jsonl(output, rows)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', default='CodePromptOpt/data/tasks.jsonl')
    p.add_argument('--split', default='validation', choices=['validation', 'test'])
    p.add_argument('--sources', default='CodePromptOpt/finetuning/outputs/benchmark_sources/validation')
    p.add_argument('--output', default='verified_tests.jsonl')
    p.add_argument('--download', action='store_true', help='Stream pinned public sources, saving only matched records')
    args = p.parse_args()
    tasks = [r for r in read_jsonl(args.tasks) if r['split'] == args.split]
    if not tasks:
        p.error('No tasks in requested split')
    if args.download:
        recover(tasks, args.sources)
    report = build(tasks, args.sources, args.output)
    print(json.dumps({'requested_tasks': report['requested_tasks'], 'matched_tasks': report['matched_tasks'],
                      'issues': report['issues'], 'tests': args.output,
                      'mapping_report': str(args.output) + '.mapping.json'}, indent=2))


if __name__ == '__main__':
    main()
