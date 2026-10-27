"""Offline dataset, annotation requests, association analysis and SGDP export."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
import random

from ..finetuning.common import read_jsonl
from ..finetuning.downstream.run import atomic_json, atomic_jsonl
from .core import (ANNOTATOR_SYSTEM, ANNOTATION_VERSION, SELECTOR_SYSTEM, VERSION, STRATEGIES, CONTROLS,
                   build_prompt, fingerprint, validate_evidence, validate_selection)


def unique_index(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate ' + key)
    return result


def export_dataset(data):
    data = Path(data)
    tasks = unique_index(read_jsonl(data / 'tasks.jsonl'), 'task_id')
    runs = unique_index(read_jsonl(data / 'runs.jsonl'), 'run_id')
    candidates = read_jsonl(data / 'candidates.jsonl')
    evaluations = defaultdict(list)
    for e in read_jsonl(data / 'evaluations.jsonl'):
        if e['evaluation_split'] == 'dev' and e['score'] is not None:
            evaluations[(e['candidate_id'], e['comparison_group_id'])].append(e)
    baselines = defaultdict(list)
    for c in candidates:
        r = runs[c['run_id']]
        if r['optimizer_method'].lower() == 'baseline':
            baselines[(c['task_id'], r['target_model'], r['comparison_group_id'])].append(c)
    flags = {r['candidate_id']: r['reasons'] for r in read_jsonl(data / 'filtering_report.jsonl')}
    records = []
    for c in candidates:
        r = runs[c['run_id']]
        if r['optimizer_method'].upper() not in ('APE', 'APO'):
            continue
        task = tasks[c['task_id']]
        group = r['comparison_group_id']
        base = baselines[(c['task_id'], r['target_model'], group)]
        ev = evaluations[(c['candidate_id'], group)]
        bev = [e for b in base for e in evaluations[(b['candidate_id'], group)]]
        scores, bscores = {e['score'] for e in ev}, {e['score'] for e in bev}
        baseline = base[0]['prompt'] if len({b['prompt'] for b in base}) == 1 else None
        comparable = baseline is not None and len(scores) == len(bscores) == 1
        score = next(iter(scores)) if len(scores) == 1 else None
        bscore = next(iter(bscores)) if len(bscores) == 1 else None
        record = dict(candidate_id=c['candidate_id'], task_id=c['task_id'], split=task['split'],
                      benchmark=task['benchmark'], difficulty=task['difficulty'], problem=task['problem'],
                      candidate_prompt=c['prompt'], baseline_prompt=baseline, optimizer=r['optimizer_method'],
                      target_model=r['target_model'], comparison_group_id=group, dev_score=score,
                      baseline_dev_score=bscore, improvement=score - bscore if comparable else None,
                      comparable=comparable, validation_status='not_revalidated', strategies=None,
                      comparison_status=('comparable' if comparable else
                          'baseline_dev_unavailable' if not bscores else 'missing_or_conflicting_dev_scores'),
                      strategy_evidence=None, quality_flags=flags.get(c['candidate_id'], []),
                      is_selected_final=c['is_selected_final'], source_records=c['source_records'],
                      evaluation_ids=[e['evaluation_id'] for e in ev],
                      baseline_evaluation_ids=[e['evaluation_id'] for e in bev])
        record['record_hash'] = fingerprint(record)
        records.append(record)
    return records


def annotation_request(record):
    messages = [
        {'role': 'system', 'content': ANNOTATOR_SYSTEM},
        {'role': 'user', 'content': json.dumps({
            'original_problem': record['problem'], 'candidate_prompt': record['candidate_prompt']},
            ensure_ascii=False)}]
    identity = {'record_hash': record['record_hash'], 'annotation_version': ANNOTATION_VERSION,
                'annotation_reference': 'original_problem', 'messages': messages}
    return {'id': record['candidate_id'], **identity, 'request_hash': fingerprint(identity)}


def annotation_requests(records, split):
    for r in records:
        if r['split'] != split:
            continue
        yield annotation_request(r)


def merge_annotations(records, annotations):
    indexed = unique_index(annotations, 'id')
    if set(indexed) - {r['candidate_id'] for r in records}:
        raise ValueError('Unknown annotation IDs')
    result = []
    for r in records:
        a = indexed.get(r['candidate_id'])
        if a is None:
            result.append(r)
            continue
        if a['record_hash'] != r['record_hash']:
            raise ValueError('Annotation input changed')
        expected = annotation_request(r)
        if any(a.get(k) != expected[k] for k in ('annotation_version', 'annotation_reference', 'request_hash')):
            raise ValueError('Annotation reference or protocol changed; regenerate requests and responses')
        value = a['response']
        if set(value) != {'strategies', 'strategy_evidence', 'review_notes'} or not isinstance(value['review_notes'], str):
            raise ValueError('Invalid annotation schema')
        validate_evidence(value['strategies'], value['strategy_evidence'], r['candidate_prompt'], STRATEGIES + CONTROLS)
        result.append({**r, **value, 'annotation_metadata': a.get('metadata'),
                       'annotation_version': ANNOTATION_VERSION, 'annotation_reference': 'original_problem',
                       'annotation_request_hash': expected['request_hash'],
                       'annotation_status': 'annotated_unreviewed'})
    return result


def analyze(records, split):
    # Each task contributes once per strategy/optimizer/benchmark/model. Both arms
    # must occur in the same historical comparison group. No held-out scores.
    groups = defaultdict(list)
    eligible = [r for r in records if r['split'] == split and r['dev_score'] is not None
                and r['strategies'] is not None and not r['quality_flags']
                and not set(r['strategies']).intersection(CONTROLS)]
    for r in eligible:
        groups[(r['benchmark'], r['optimizer'], r['target_model'], r['task_id'], r['comparison_group_id'])].append(r)
    contrasts = defaultdict(lambda: defaultdict(list))
    for (benchmark, optimizer, target, task, group), rows in groups.items():
        for strategy in STRATEGIES:
            # A shared baseline cancels from E[delta|present]-E[delta|absent].
            # Thus a within-group score contrast remains observable even when
            # the baseline dev evaluation was never logged.
            present = [r['dev_score'] for r in rows if strategy in r['strategies']]
            absent = [r['dev_score'] for r in rows if strategy not in r['strategies']]
            if present and absent:
                contrasts[(benchmark, optimizer, target, strategy)][task].append(mean(present) - mean(absent))
    def interval(values):
        rng = random.Random(42)
        samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
        return [samples[49], samples[1949]] if len(values) > 1 else None

    return {'split': split, 'eligible_candidates': len(eligible),
            'candidates_with_baseline_delta': sum(r['comparable'] for r in eligible),
            'frequency': {s: sum(s in r['strategies'] for r in eligible) for s in STRATEGIES},
            'cooccurrence': {'+'.join(pair): sum(set(pair) <= set(r['strategies']) for r in eligible)
                             for pair in combinations(STRATEGIES, 2)},
            'metric': 'Within-comparison-group dev score contrast, averaged equally across groups within task and then tasks; shared baseline cancels.',
            'warning': 'Exploratory associations, not causal effects. Co-occurrence and search-selection bias remain; annotations require human review.',
            'contrasts': [dict(benchmark=k[0], optimizer=k[1], target_model=k[2], strategy=k[3],
                               task_count=len(v), mean_task_contrast=mean(mean(x) for x in v.values()),
                               bootstrap_95_ci=interval([mean(x) for x in v.values()]),
                               task_contrasts={t: mean(x) for t, x in v.items()})
                          for k, v in sorted(contrasts.items())]}


def export_prompts(tasks, output, split, decisions=None, threshold=0.5):
    tasks = [t for t in tasks if t['split'] == split]
    if not tasks:
        raise ValueError('Empty task split')
    unique_index(tasks, 'task_id')
    decisions = unique_index(decisions, 'id') if decisions is not None else None
    if decisions is not None and set(decisions) != {t['task_id'] for t in tasks}:
        raise ValueError('Decisions must exactly cover selected tasks')
    methods = {'original': (), 'static': None, 'all': STRATEGIES}
    if split == 'validation':
        for n in range(1, 3):
            for subset in combinations(STRATEGIES, n):
                methods['ablation_' + ''.join('CAP'[STRATEGIES.index(s)] for s in subset)] = subset
    if decisions is not None:
        methods.update(dynamic=None, dynamic_abstain=None)
    outputs = defaultdict(list)
    for task in tasks:
        decision = None
        if decisions is not None:
            d = decisions[task['task_id']]
            if d['problem_hash'] != fingerprint(task['problem']):
                raise ValueError('Selector input changed')
            decision = validate_selection(task['problem'], d['response'])
        for method, subset in methods.items():
            if method.startswith('dynamic'):
                subset = decision['strategies']
                if method == 'dynamic' and not subset:
                    subset = (STRATEGIES[2],)  # Preregistered nonempty M3 fallback.
                if method == 'dynamic_abstain' and decision['confidence'] < threshold:
                    subset = ()
            prompt = ('Write a correct Python solution. Return only the required code.\n\n' + task['problem']
                      if method == 'static' else build_prompt(task['problem'], subset))
            outputs[method].append(dict(task_id=task['task_id'], split=split, variant=method,
                prompt=prompt, truncated=False, strategies=list(subset or ()), template_version=VERSION,
                problem_hash=fingerprint(task['problem']), selector_decision=decision,
                abstention_threshold=threshold if method == 'dynamic_abstain' else None))
    for method, rows in outputs.items():
        atomic_jsonl(Path(output) / 'sgdp-v1' / f'{split}-{method}-prompts.jsonl', rows)
    atomic_json(Path(output) / 'sgdp-v1' / f'{split}-protocol.json', {
        'template_version': VERSION, 'methods': list(outputs), 'abstention_threshold': threshold,
        'task_ids': [t['task_id'] for t in tasks], 'decision_hash': fingerprint(decisions),
        'templates_hash': fingerprint({m: [r['prompt'] for r in rows] for m, rows in outputs.items()}),
        'note': 'Greedy downstream pilot only; repeated stochastic generations require a separate sampling protocol.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['dataset', 'annotation-requests', 'merge', 'analyze', 'selector-requests', 'prompts'])
    p.add_argument('--data', default='CodePromptOpt/data')
    p.add_argument('--input')
    p.add_argument('--responses')
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    p.add_argument('--abstention-threshold', type=float, default=0.5)
    a = p.parse_args()
    if not 0 <= a.abstention_threshold <= 1:
        p.error('Threshold must be in [0,1]')
    if a.command == 'dataset':
        records = export_dataset(a.data)
        atomic_jsonl(a.output, records)
        atomic_json(a.output + '.manifest.json', {'version': VERSION, 'records': len(records),
            'source_hashes': {name: fingerprint(read_jsonl(Path(a.data) / name)) for name in
                              ('tasks.jsonl', 'runs.jsonl', 'candidates.jsonl', 'evaluations.jsonl', 'filtering_report.jsonl')},
            'comparable': sum(r['comparable'] for r in records),
            'baseline_dev_unavailable': sum(r['comparison_status'] == 'baseline_dev_unavailable' for r in records)})
    elif a.command in ('annotation-requests', 'merge', 'analyze'):
        if not a.input:
            p.error('--input required')
        records = read_jsonl(a.input)
        if a.command == 'annotation-requests':
            atomic_jsonl(a.output, annotation_requests(records, a.split))
        elif a.command == 'merge':
            if not a.responses:
                p.error('--responses required')
            atomic_jsonl(a.output, merge_annotations(records, read_jsonl(a.responses)))
        else:
            if a.split == 'test':
                p.error('Exploratory analysis is restricted to train/validation')
            atomic_json(a.output, analyze(records, a.split))
    else:
        tasks = read_jsonl(Path(a.data) / 'tasks.jsonl')
        if a.command == 'selector-requests':
            atomic_jsonl(a.output, ({'id': t['task_id'], 'problem_hash': fingerprint(t['problem']),
                'messages': [{'role': 'system', 'content': SELECTOR_SYSTEM},
                             {'role': 'user', 'content': t['problem']}]} for t in tasks if t['split'] == a.split))
        else:
            if a.split == 'train':
                p.error('Prompt evaluation requires validation/test')
            export_prompts(tasks, a.output, a.split,
                           read_jsonl(a.responses) if a.responses else None, a.abstention_threshold)


if __name__ == '__main__':
    main()
