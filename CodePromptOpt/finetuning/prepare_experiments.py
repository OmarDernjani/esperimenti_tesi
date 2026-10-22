"""Build matched Best/Random/Top-k training variants from observed dev labels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from CodePromptOpt.build import quality_reasons
from .common import INSTRUCTION, digest, load_splits, read_jsonl, write_jsonl


def select_pool(pool, strategy, seed, top_k=3, score_gap=0.1):
    pool = sorted(pool, key=lambda r: (-r['label_dev_score'], len(r['output']), r['source_candidate_id']))
    best = pool[0]
    if strategy == 'best':
        return [best]
    if strategy == 'random':
        # Match character length approximately, before tokenizer-specific filtering.
        matched = [r for r in pool if 0.5 <= len(r['output']) / len(best['output']) <= 2.0]
        key = f'{seed}:{best["task_id"]}:{best["comparison_group_id"]}'
        rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))
        return [rng.choice(matched)]
    chosen, signatures = [], []
    for row in pool:
        if row['label_dev_score'] < best['label_dev_score'] - score_gap:
            continue
        signature = set(row['output'].replace(row['input'], '').split())
        if any(len(signature & prev) / max(1, len(signature | prev)) >= 0.9 for prev in signatures):
            continue
        chosen.append(row)
        signatures.append(signature)
        if len(chosen) == top_k:
            break
    return chosen or [best]


def prepare(data_dir, output_dir, target, seed=42, top_k=3):
    source, output = Path(data_dir), Path(output_dir)
    tasks, exports = load_splits(source)
    task_map = {r['task_id']: r for r in tasks}
    runs = {r['run_id']: r for r in read_jsonl(source / 'runs.jsonl')}
    # Reuse the dataset builder's conflict/quality exclusions, but not deduplication
    # of final SFT samples across independent conditions.
    blocked = {r['candidate_id'] for r in read_jsonl(source / 'filtering_report.jsonl')
               if r.get('excluded_from_training') and 'candidate_id' in r and 'sample_id' not in r}
    dev = defaultdict(list)
    for row in read_jsonl(source / 'evaluations.jsonl'):
        if row['evaluation_split'] == 'dev':
            dev[row['candidate_id']].append(row)
    # Choose one condition per training task using its ID, never ranking unrelated
    # dev suites against each other. This also matches the task counts across methods.
    anchors = defaultdict(set)
    for row in exports['train']:
        if row['target_model'] == target:
            anchors[row['task_id']].add(row['comparison_group_id'])
    selected_group = {tid: sorted(groups)[0] for tid, groups in anchors.items()}
    pools = defaultdict(list)
    seen = set()
    for candidate in read_jsonl(source / 'candidates.jsonl'):
        tid, cid = candidate['task_id'], candidate['candidate_id']
        run, task = runs[candidate['run_id']], task_map[tid]
        if task['split'] != 'train' or run['target_model'] != target:
            continue
        if run['comparison_group_id'] != selected_group.get(tid) or cid in blocked:
            continue
        if run['optimizer_method'] not in ('APE', 'APO') or quality_reasons(candidate['prompt'], task['problem']):
            continue
        observations = dev[cid]
        scores = {r['score'] for r in observations}
        if len(scores) != 1 or not (0 < observations[0]['score'] <= 1):
            continue
        key = (tid, candidate['prompt'])
        if key in seen:
            continue
        seen.add(key)
        pools[tid].append({'sample_id': f'h200_{cid}', 'task_id': tid, 'split': 'train',
            'target_model': target, 'comparison_group_id': run['comparison_group_id'],
            'instruction': INSTRUCTION, 'input': task['problem'], 'output': candidate['prompt'],
            'source_candidate_id': cid, 'label_dev_score': observations[0]['score'],
            'label_evaluation_ids': sorted(r['evaluation_id'] for r in observations)})
    if set(pools) != set(selected_group) or not pools:
        raise ValueError('Candidate eligibility differs from SFT anchors; investigate dataset before training')
    manifest = {'target': target, 'selection_seed': seed, 'train_tasks': len(pools),
                'condition_per_task': selected_group, 'variants': {}, 'hashes': {}}
    for name in ('tasks.jsonl', 'splits.json', 'runs.jsonl', 'candidates.jsonl',
                 'evaluations.jsonl', 'filtering_report.jsonl', 'sft/train.jsonl', 'sft/validation.jsonl'):
        manifest['hashes'][name] = digest(source / name)
    for strategy in ('best', 'random', 'topk'):
        folder = output / strategy
        folder.mkdir(parents=True, exist_ok=True)
        rows = [r for tid, pool in sorted(pools.items())
                for r in select_pool(pool, strategy, seed, top_k)]
        write_jsonl(folder / 'tasks.jsonl', tasks)
        write_jsonl(folder / 'sft/train.jsonl', rows)
        # Fixed validation labels across all training strategies.
        validation = [r for r in exports['validation'] if r['target_model'] == target]
        write_jsonl(folder / 'sft/validation.jsonl', validation)
        manifest['variants'][strategy] = {'train_samples': len(rows), 'validation_samples': len(validation)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='CodePromptOpt/data')
    p.add_argument('--output', required=True)
    p.add_argument('--target', default='qwen3.5-coder')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--top-k', type=int, default=3)
    args = p.parse_args()
    if args.top_k < 1:
        p.error('--top-k must be positive')
    print(json.dumps(prepare(args.data, args.output, args.target, args.seed, args.top_k), indent=2))


if __name__ == '__main__':
    main()
