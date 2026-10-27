"""Reproducible dual-annotator pilot and agreement report; no model calls."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from ..finetuning.common import read_jsonl
from ..finetuning.downstream.run import atomic_json, atomic_jsonl
from .core import STRATEGIES, CONTROLS, ANNOTATION_VERSION, fingerprint
from .pipeline import annotation_request, merge_annotations, unique_index


def pilot(records, size=100, seed=42):
    if size < 1:
        raise ValueError('Pilot size must be positive')
    # The annotator never sees these sampling strata or performance scores.
    strata = defaultdict(list)
    for r in sorted(records, key=lambda x: x['candidate_id']):
        if r['split'] != 'train':
            continue
        score = r['dev_score']
        band = 'unknown' if score is None else 'zero' if score == 0 else 'perfect' if score == 1 else 'partial'
        strata[(r['benchmark'], r['optimizer'], r['difficulty'] or '', bool(r['quality_flags']), band)].append(r)
    if sum(map(len, strata.values())) < size:
        raise ValueError('Not enough train candidates')
    rng = random.Random(seed)
    keys = sorted(strata)
    rng.shuffle(keys)
    for rows in strata.values():
        rng.shuffle(rows)
    buckets = defaultdict(list)
    for key in keys:
        buckets[key[:2]].append(key)
    bucket_keys = sorted(buckets)
    rng.shuffle(bucket_keys)
    cursor = Counter()
    selected, task_counts, text_seen = [], Counter(), set()
    # Round-robin strata; within each stratum prefer less-represented tasks and
    # new exact problem/candidate text. This is a diagnostic, not random-population sample.
    while len(selected) < size:
        for bucket in bucket_keys:
            available = [k for k in buckets[bucket] if strata[k]]
            if not available:
                continue
            key = available[cursor[bucket] % len(available)]
            cursor[bucket] += 1
            rows = strata[key]
            index = min(range(len(rows)), key=lambda i: (
                (rows[i]['problem'], rows[i]['candidate_prompt']) in text_seen,
                task_counts[rows[i]['task_id']]))
            row = rows.pop(index)
            selected.append(row)
            task_counts[row['task_id']] += 1
            text_seen.add((row['problem'], row['candidate_prompt']))
            if len(selected) == size:
                break
    return selected


def agreement(left, right):
    if not left:
        return {'n': 0, 'agreement': None, 'kappa': None, 'positive_agreement': None,
                'a_positive': 0, 'b_positive': 0}
    n = len(left)
    same = sum(a == b for a, b in zip(left, right)) / n
    pa, pb = sum(left) / n, sum(right) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    both = sum(a and b for a, b in zip(left, right))
    return {'n': n, 'agreement': same, 'kappa': (same - chance) / (1 - chance) if chance < 1 else None,
            'positive_agreement': 2 * both / (sum(left) + sum(right)) if sum(left) + sum(right) else None,
            'a_positive': sum(left), 'b_positive': sum(right)}


def compare(records, responses_a, responses_b):
    expected = unique_index(records, 'candidate_id')
    outcomes = []
    for name, responses in [('a', responses_a), ('b', responses_b)]:
        indexed = unique_index(responses, 'id')
        if set(indexed) - set(expected):
            raise ValueError('Unknown IDs in annotator ' + name)
        valid, invalid = {}, []
        for cid, response in indexed.items():
            try:
                annotated = merge_annotations([expected[cid]], [response])[0]
                valid[cid] = annotated
            except (ValueError, KeyError, TypeError) as error:
                invalid.append({'id': cid, 'error': str(error)})
        outcomes.append((valid, invalid, sorted(set(expected) - set(indexed))))
    (a, invalid_a, missing_a), (b, invalid_b, missing_b) = outcomes
    common = sorted(set(a) & set(b))
    metrics = {label: agreement([label in a[c]['strategies'] for c in common],
                                [label in b[c]['strategies'] for c in common])
               for label in STRATEGIES + CONTROLS}
    review = []
    for cid in expected:
        ra, rb = a.get(cid), b.get(cid)
        if ra is None or rb is None or set(ra['strategies']) != set(rb['strategies']):
            review.append({'candidate_id': cid, 'task_id': expected[cid]['task_id'],
                'problem': expected[cid]['problem'], 'candidate_prompt': expected[cid]['candidate_prompt'],
                'reason': 'missing_or_invalid' if ra is None or rb is None else 'label_disagreement',
                'annotator_a': None if ra is None else {k: ra[k] for k in ('strategies', 'strategy_evidence', 'review_notes')},
                'annotator_b': None if rb is None else {k: rb[k] for k in ('strategies', 'strategy_evidence', 'review_notes')},
                'human_strategies': None, 'human_strategy_evidence': None, 'human_notes': ''})
    report = {'expected': len(expected), 'paired_valid': len(common),
        'annotator_a': {'valid': len(a), 'invalid': invalid_a, 'missing_ids': missing_a},
        'annotator_b': {'valid': len(b), 'invalid': invalid_b, 'missing_ids': missing_b},
        'exact_label_set_agreement': sum(set(a[c]['strategies']) == set(b[c]['strategies']) for c in common) / len(common) if common else None,
        'per_label': metrics, 'review_queue_size': len(review),
        'annotation_version': ANNOTATION_VERSION,
        'warning': 'Agreement is not annotation accuracy. Diagnostic stratified pilot, not a population estimate. Null kappa indicates no paired data or constant unanimous labels. Review some agreements as well as all disagreements.'}
    return report, review


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sample = sub.add_parser('pilot')
    sample.add_argument('--input', default='CodePromptOpt/strategies/artifacts/candidates.jsonl')
    sample.add_argument('--output', default='CodePromptOpt/strategies/artifacts/annotation_pilot_v2')
    sample.add_argument('--size', type=int, default=100)
    sample.add_argument('--seed', type=int, default=42)
    comp = sub.add_parser('compare')
    comp.add_argument('--records', required=True)
    comp.add_argument('--a', required=True)
    comp.add_argument('--b', required=True)
    comp.add_argument('--output', required=True)
    a = p.parse_args()
    if a.command == 'pilot':
        records = read_jsonl(a.input)
        selected = pilot(records, a.size, a.seed)
        root = Path(a.output)
        manifest = {'annotation_version': ANNOTATION_VERSION, 'size': a.size, 'seed': a.seed,
            'source_hash': fingerprint(records), 'candidate_ids': [r['candidate_id'] for r in selected],
            'task_count': len({r['task_id'] for r in selected}),
            'benchmark_optimizer_counts': dict(Counter(r['benchmark'] + '/' + r['optimizer'] for r in selected)),
            'quality_flagged': sum(bool(r['quality_flags']) for r in selected),
            'sampling': 'Balanced benchmark/optimizer buckets; round-robin difficulty/quality/dev-score strata within bucket; favors task diversity. Train only. Identical pilot for both annotators.'}
        if (root / 'manifest.json').exists():
            if json.loads((root / 'manifest.json').read_text(encoding='utf-8')) != manifest:
                raise ValueError('Pilot changed: choose a new output directory')
        atomic_json(root / 'manifest.json', manifest)
        atomic_jsonl(root / 'records.jsonl', selected)
        atomic_jsonl(root / 'requests.jsonl', [annotation_request(r) for r in selected])
        print(json.dumps({k: v for k, v in manifest.items() if k != 'candidate_ids'}, indent=2))
    else:
        records, ra, rb = read_jsonl(a.records), read_jsonl(a.a), read_jsonl(a.b)
        report, review = compare(records, ra, rb)
        report['input_hashes'] = {'records': fingerprint(records), 'a': fingerprint(ra), 'b': fingerprint(rb)}
        root = Path(a.output)
        atomic_json(root / 'agreement.json', report)
        atomic_jsonl(root / 'human_review.jsonl', review)
        print(json.dumps({k: report[k] for k in ('expected', 'paired_valid', 'exact_label_set_agreement', 'review_queue_size')}, indent=2))


if __name__ == '__main__':
    main()
