"""Recover count-consistent historical dev hypotheses without inventing provenance."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path

from ..finetuning.common import digest, read_jsonl
from ..finetuning.downstream.prepare_tests import recover, APPS_HARNESS
from ..finetuning.downstream.run import atomic_json, atomic_jsonl
from .core import fingerprint


def plus_io(source):
    tree = ast.parse(source['test'])
    checks = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check']
    if len(checks) != 1:
        raise ValueError('Unique check(candidate) function required')
    values = {}
    for node in checks[0].body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('inputs', 'results'):
                    if target.id in values:
                        raise ValueError('Repeated test-array assignment')
                    values[target.id] = ast.literal_eval(node.value)
    return {'fn_name': source['entry_point'], 'inputs': values['inputs'],
            'outputs': [[x] for x in values['results']]}


def reconstruct(task, source, n_dev, n_test):
    if type(n_dev) is not int or type(n_test) is not int or min(n_dev, n_test) < 1:
        raise ValueError('Positive logged dev/test counts required')
    if task['benchmark'] == 'HumanEval+':
        if source['task_id'] != task['source_id'] or source['prompt'].strip() != task['problem'].strip():
            raise ValueError('HumanEval ID/prompt mismatch')
        if source['entry_point'] != task['entry_point']:
            raise ValueError('HumanEval entry point mismatch')
        data = plus_io(source)
        if len(data['inputs']) != n_dev + n_test:
            raise ValueError('HumanEval full suite size differs from logged dev+test')
        truncation = None
    elif task['benchmark'] == 'APPS':
        if source['question'] != task['problem']:
            raise ValueError('APPS exact question mismatch')
        raw = source['input_output']
        data = json.loads(raw) if isinstance(raw, str) else raw
        truncation = n_dev + n_test
    else:
        raise ValueError('Unsupported benchmark')
    if not isinstance(data['inputs'], list) or not isinstance(data['outputs'], list):
        raise ValueError('Expected ordered input/output lists')
    if len(data['inputs']) != len(data['outputs']) or len(data['inputs']) < n_dev + n_test:
        raise ValueError('Missing or unaligned source cases')
    dev = {k: v for k, v in data.items() if k in ('fn_name',)}
    dev.update(inputs=data['inputs'][:n_dev], outputs=data['outputs'][:n_dev])
    test = {k: v for k, v in data.items() if k in ('fn_name',)}
    test.update(inputs=data['inputs'][n_dev:n_dev+n_test], outputs=data['outputs'][n_dev:n_dev+n_test])
    return {'dev_io': dev, 'test_io': test, 'dev_io_literal': repr(dev), 'test_io_literal': repr(test),
            'suite_hash': fingerprint({'dev': repr(dev), 'test': repr(test)}),
            'assumed_official_case_cap': truncation,
            'historical_identity_verified': False,
            'status': 'reconstructed_count_consistent_unverified',
            'assumptions': ['Pinned source matches historical dataset content and ordering',
                            'Historical run used classic contiguous-prefix dev split, no augmentation',
                            'Logged dev count determines prefix length; historical DEV_SIZE is not known'],
            'provenance': source['_provenance']}


def case_harness(data):
    # HumanEval fixtures can contain nested tuples: JSON round-tripping would
    # silently turn them into lists and change the tested input.
    return APPS_HARNESS.replace('DATA = json.loads(DATA_JSON)',
                               'DATA = ast.literal_eval(' + repr(repr(data)) + ')', 1)


def build(data_dir, sources_dir, output, augmented_path=None):
    data_dir, sources_dir, output = map(Path, (data_dir, sources_dir, output))
    tasks = {r['task_id']: r for r in read_jsonl(data_dir / 'tasks.jsonl')}
    source_rows = {}
    for benchmark, filename in [('APPS', 'apps.jsonl'), ('HumanEval+', 'humanevalplus.jsonl')]:
        source_rows[benchmark] = read_jsonl(sources_dir / filename)
    groups = {}
    for run in read_jsonl(data_dir / 'runs.jsonl'):
        key = run['comparison_group_id']
        signature = {k: run[k] for k in ('task_id', 'n_dev', 'n_test')}
        if key in groups and groups[key] != signature:
            raise ValueError('Conflicting logged counts for group ' + key)
        groups[key] = signature
    suites, issues = [], []
    for group, logged in sorted(groups.items()):
        task = tasks[logged['task_id']]
        try:
            matches = [s for s in source_rows[task['benchmark']] if
                       (s.get('task_id') == task['source_id'] if task['benchmark'] == 'HumanEval+'
                        else s.get('question') == task['problem'])]
            if len(matches) != 1:
                raise ValueError('Source mapping is missing or ambiguous')
            suite = reconstruct(task, matches[0], logged['n_dev'], logged['n_test'])
            suites.append({**logged, 'comparison_group_id': group, 'benchmark': task['benchmark'],
                           'split': task['split'], **suite})
        except (ValueError, KeyError, SyntaxError, TypeError) as error:
            issues.append({'comparison_group_id': group, **logged, 'error': str(error)})
    atomic_jsonl(output / 'dev_suites.jsonl', suites)
    # Export saved historical code and one harness per case. This makes fractional
    # dev accuracy measurable with the existing binary sandbox evaluator.
    by_group = {s['comparison_group_id']: s for s in suites}
    runs = {r['run_id']: r for r in read_jsonl(data_dir / 'runs.jsonl')}
    candidates = {c['candidate_id']: c for c in read_jsonl(data_dir / 'candidates.jsonl')}
    samples, harnesses, seen = [], [], {}
    for e in read_jsonl(data_dir / 'evaluations.jsonl'):
        c = candidates[e['candidate_id']]
        run = runs[c['run_id']]
        if run['optimizer_method'] not in ('baseline', 'zero_shot', 'human_prompt'):
            continue
        suite = by_group.get(e['comparison_group_id'])
        if suite is None or not isinstance(e['generated_code'], str) or not e['generated_code'].strip():
            continue
        identity = (e['candidate_id'], e['comparison_group_id'])
        if identity in seen:
            if seen[identity] != e['generated_code']:
                raise ValueError('Conflicting historical baseline code')
            continue
        seen[identity] = e['generated_code']
        io = suite['dev_io']
        for index, (case, expected) in enumerate(zip(io['inputs'], io['outputs'])):
            job_id = fingerprint([*identity, index])
            one = {**io, 'inputs': [case], 'outputs': [expected]}
            samples.append({'task_id': job_id, 'original_task_id': c['task_id'],
                'candidate_id': c['candidate_id'], 'comparison_group_id': e['comparison_group_id'],
                'historical_evaluation_id': e['evaluation_id'], 'optimizer_method': run['optimizer_method'],
                'target_model': run['target_model'], 'split': c['split'], 'case_index': index,
                'n_dev': len(io['inputs']), 'suite_hash': suite['suite_hash'],
                'historical_identity_verified': False, 'code': e['generated_code'],
                'preserves_problem_verbatim': tasks[c['task_id']]['problem'] in c['prompt'],
                'prompt_chars': len(c['prompt'])})
            harnesses.append({'task_id': job_id, 'harness': case_harness(one),
                              'runtime_requires': ['numpy'] if suite['benchmark'] == 'HumanEval+' else []})
    atomic_jsonl(output / 'baseline_dev_samples.jsonl', samples)
    atomic_jsonl(output / 'baseline_dev_tests.jsonl', harnesses)
    augmentation = None
    if augmented_path and Path(augmented_path).exists():
        aug = json.loads(Path(augmented_path).read_text(encoding='utf-8'))
        augmentation = {'path': str(augmented_path), 'sha256': digest(augmented_path),
                        'problems': len(aug.get('problems', {})),
                        'used': False, 'reason': 'Historical augmentation usage and exact minibatch identity are not recorded; no index-only matching.'}
    report = {'groups_requested': len(groups), 'groups_reconstructed': len(suites),
        'tasks_reconstructed': len({s['task_id'] for s in suites}),
        'verified_historical_suites': 0, 'baseline_programs_exported': len(seen),
        'case_jobs_exported': len(samples), 'issues': issues, 'augmented_file': augmentation,
        'status_counts': dict(Counter(s['status'] for s in suites)),
        'source_hashes': {name: digest(sources_dir / name) for name in ('apps.jsonl', 'humanevalplus.jsonl')},
        'canonical_hashes': {name: digest(data_dir / name) for name in ('tasks.jsonl', 'runs.jsonl', 'candidates.jsonl', 'evaluations.jsonl')},
        'warning': 'Count consistency does not prove historical identity. Scores must remain separate from canonical historical dev scores until independent suite verification.'}
    atomic_json(output / 'recovery_report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='CodePromptOpt/data')
    p.add_argument('--sources', default='CodePromptOpt/strategies/artifacts/benchmark_sources')
    p.add_argument('--output', default='CodePromptOpt/strategies/artifacts/dev_recovery')
    p.add_argument('--augmented', default='augmented_dev.json')
    p.add_argument('--download', action='store_true')
    a = p.parse_args()
    if a.download:
        recover(read_jsonl(Path(a.data) / 'tasks.jsonl'), a.sources)
    report = build(a.data, a.sources, a.output, a.augmented)
    print(json.dumps({k: v for k, v in report.items() if k not in ('issues', 'source_hashes', 'canonical_hashes')}, indent=2))


if __name__ == '__main__':
    main()
