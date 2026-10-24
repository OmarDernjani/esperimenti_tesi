"""Audit local optimizer outputs and call an OpenAI-compatible code generator."""
import argparse
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import fnmatch
import json
import os
from pathlib import Path
import sqlite3
import time
import urllib.error
import urllib.request

from ..common import read_jsonl
from ..codegen import extract_code

SYSTEM = 'Return only the complete executable Python solution, without Markdown.'
VARIANTS = ('original', 'static', 'base', 'sft')


def key_for(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    temporary.replace(path)


def discover(root, tasks_path, split, runs=None, variants=VARIANTS):
    tasks = {r['task_id']: r for r in read_jsonl(tasks_path) if r['split'] == split}
    if not tasks:
        raise ValueError('No benchmark tasks for selected split')
    files, issues, counts = [], [], {'rows': 0, 'truncated': 0, 'problem_not_preserved': 0}
    directories = sorted(p for p in Path(root).iterdir() if p.is_dir() and (not runs or p.name in runs))
    if runs and set(runs) - {p.name for p in directories}:
        raise ValueError('Selected run names not found')
    for directory in directories:
        for variant in variants:
            path = directory / f'{split}-{variant}-prompts.jsonl'
            if not path.exists():
                issues.append({'run': directory.name, 'variant': variant, 'reason': 'missing_file'})
                continue
            rows = read_jsonl(path)
            ids = [r['task_id'] for r in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f'Duplicate task IDs: {path}')
            if set(ids) - tasks.keys():
                raise ValueError(f'Unknown task / wrong split: {path}')
            if set(ids) != tasks.keys():
                issues.append({'run': directory.name, 'variant': variant, 'reason': 'incomplete_file',
                               'rows': len(rows), 'expected': len(tasks)})
            for row in rows:
                if row['split'] != split or row['variant'] != variant or not isinstance(row['prompt'], str):
                    raise ValueError(f'Inconsistent prompt record in {path}')
                counts['rows'] += 1
                counts['truncated'] += bool(row.get('truncated'))
                counts['problem_not_preserved'] += tasks[row['task_id']]['problem'] not in row['prompt']
            files.append((directory.name, variant, path, rows))
    if not files:
        raise ValueError('No prompt files found; check --input-root')
    return files, {'split': split, 'files': len(files), 'runs': len(directories),
                   'expected_tasks_per_file': len(tasks), **counts, 'coverage_issues': issues}


def request_json(url, body=None, timeout=600, retries=3):
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + os.environ.get('CODEGEN_API_KEY', 'local')})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors='replace')[:2000]
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f'HTTP {error.code}: {detail}') from error
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(min(2 ** attempt, 16))


def generate(base_url, body, timeout, retries):
    start = time.perf_counter()
    raw = request_json(base_url + '/chat/completions', body, timeout, retries)
    choice = raw['choices'][0]
    content = choice['message'].get('content')
    if not isinstance(content, str):
        raise ValueError('Missing textual assistant content; check reasoning/tool-call settings')
    return {'raw_response': raw, 'code': extract_code(content),
            'finish_reason': choice.get('finish_reason'), 'usage': raw.get('usage'),
            'codegen_latency_seconds': time.perf_counter() - start}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', default='CodePromptOpt/finetuning/outputs/runs')
    p.add_argument('--tasks', default='CodePromptOpt/data/tasks.jsonl')
    p.add_argument('--output-root', default='CodePromptOpt/finetuning/outputs/codegen')
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--runs', nargs='+', help='Exact run directory names; otherwise all runs')
    p.add_argument('--run-glob', nargs='+', help='Run directory patterns, e.g. qwen3_8b-*')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--config', default='CodePromptOpt/finetuning/downstream/configs/qwen25_coder_7b.json')
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--plan-only', action='store_true', help='Offline audit; no server calls')
    p.add_argument('--allow-partial', action='store_true', help='Explicitly allow missing/incomplete prompt files')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--retries', type=int, default=3)
    args = p.parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.retries < 0:
        p.error('Invalid workers, timeout or retries')
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if not cfg.get('experiment_id') or not cfg.get('model'):
        p.error('Config requires experiment_id and model')
    experiment = cfg['experiment_id']
    if not all(c.isalnum() or c in '-_.' for c in experiment) or experiment in ('.', '..'):
        p.error('experiment_id must be a directory name')
    root = Path(args.output_root) / experiment / args.split
    root.mkdir(parents=True, exist_ok=True)
    selected_runs = args.runs
    if args.run_glob:
        if args.runs:
            p.error('Choose --runs or --run-glob')
        selected_runs = [r.name for r in Path(args.input_root).iterdir() if r.is_dir() and
                         any(fnmatch.fnmatchcase(r.name, pattern) for pattern in args.run_glob)]
        if not selected_runs:
            p.error('No run directories match --run-glob')
    files, audit = discover(args.input_root, args.tasks, args.split, selected_runs, args.variants)
    parameters = {'temperature': 0, 'top_p': 1, 'max_tokens': 2048, 'seed': 42,
                  **cfg.get('parameters', {})}
    if set(parameters) & {'model', 'messages', 'stream', 'n'}:
        p.error('parameters cannot override model/messages or request multiple/streamed responses')
    if parameters['temperature'] != 0:
        p.error('This protocol reuses identical greedy requests; temperature must be 0')
    extra = cfg.get('extra_body', {})
    if set(extra) & {'model', 'messages', 'stream', 'n', *parameters.keys()}:
        p.error('extra_body cannot override model/messages/decoding fields')
    metadata = cfg.get('server_metadata', {})
    tasks, requests = {}, {}
    for run, variant, path, rows in files:
        for row in rows:
            body = {'model': cfg['model'], 'messages': [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': row['prompt']}], **parameters, **extra}
            # Include task identity so identical strings in different tasks are not mixed.
            fingerprint = key_for({'task_id': row['task_id'], 'body': body, 'server_metadata': metadata})
            requests[fingerprint] = body
            tasks[(run, variant, row['task_id'])] = fingerprint
    audit.update(unique_requests=len(requests), experiment_id=experiment,
                 repeated_requests_reused=audit['rows'] - len(requests))
    atomic_json(root / 'audit.json', audit)
    print(json.dumps({**{k: v for k, v in audit.items() if k != 'coverage_issues'},
                      'coverage_issues': len(audit['coverage_issues']), 'audit_file': str(root / 'audit.json')},
                     indent=2), flush=True)
    if args.plan_only:
        return
    if audit['coverage_issues'] and not args.allow_partial:
        p.error('Incomplete prompt coverage: inspect audit.json, select --runs or explicitly use --allow-partial')
    configuration = {'config': cfg, 'effective_parameters': parameters, 'system_message': SYSTEM}
    config_path = root / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != configuration:
        p.error('Experiment configuration changed: choose a new experiment_id')
    atomic_json(config_path, configuration)
    base_url = args.base_url.rstrip('/')
    models = request_json(base_url + '/models', timeout=args.timeout, retries=args.retries)
    if cfg['model'] not in {r['id'] for r in models['data']}:
        p.error(f'Model {cfg["model"]} is not served by this endpoint')
    # Transaction per response: Ctrl+C/network failure can resume without repeating
    # successful requests. Raw responses are stored once, not in every duplicate file.
    with closing(sqlite3.connect(root / 'responses.sqlite')) as cache:
        cache.execute('CREATE TABLE IF NOT EXISTS responses (id TEXT PRIMARY KEY, response TEXT NOT NULL)')
        responses = {key: json.loads(value) for key, value in cache.execute('SELECT id,response FROM responses')}
        pending = {key: body for key, body in requests.items() if key not in responses}
        print(f'Cached: {len(requests) - len(pending)}; to generate: {len(pending)}', flush=True)
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate, base_url, body, args.timeout, args.retries): key
                       for key, body in pending.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    response = future.result()
                except Exception as error:
                    failures.append({'request_id': key, 'error': str(error)})
                    print(f'Request failed: {error}', flush=True)
                    continue
                responses[key] = response
                cache.execute('INSERT OR REPLACE INTO responses VALUES (?,?)', (key, json.dumps(response)))
                cache.commit()
                print(f'Completed {sum(k in responses for k in requests)}/{len(requests)}', flush=True)
        completed, incomplete = 0, []
        for run, variant, path, rows in files:
            output_rows = []
            for row in rows:
                key = tasks[(run, variant, row['task_id'])]
                if key not in responses:
                    continue
                response = responses[key]
                output_rows.append({**row, **{k: v for k, v in response.items() if k != 'raw_response'},
                    'codegen_model': cfg['model'], 'codegen_parameters': {**parameters, 'extra_body': extra},
                    'codegen_request_id': key, 'codegen_server_metadata': metadata,
                    'codegen_experiment_id': experiment})
            if len(output_rows) == len(rows):
                atomic_jsonl(root / run / f'{args.split}-{variant}-code.jsonl', output_rows)
                completed += 1
            else:
                incomplete.append(str(path))
        atomic_json(root / 'status.json', {'complete_files': completed, 'incomplete_files': incomplete,
                    'failed_requests': failures, 'endpoint_models': models})
        print(f'Wrote {completed} code files under {root}', flush=True)
        if failures or incomplete:
            raise SystemExit('Some requests failed. Rerun the same command to resume from the response cache.')


if __name__ == '__main__':
    main()
