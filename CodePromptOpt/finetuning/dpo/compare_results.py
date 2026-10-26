"""Compare DPO and classic SFT on identical tasks, training seeds and coder protocol."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from ..common import read_jsonl
from ..downstream.summarize_results import task_bootstrap

FIELDS = ('codegen_model', 'codegen_parameters', 'codegen_server_metadata',
          'target_label', 'split', 'tests_sha256', 'test_image', 'test_timeout')


def analyze(dpo_root, sft_root, tasks_path, seeds, split='validation', pattern='qwen3_8b-dpo-s{seed}'):
    tasks = {r['task_id'] for r in read_jsonl(tasks_path) if r['split'] == split}
    if not tasks or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Need canonical tasks and distinct training seeds')
    protocol, conditions = {}, {}

    def load(root, run, variant):
        path = Path(root) / run / f'{split}-{variant}-results.jsonl'
        rows = read_jsonl(path)
        mapping = {r['task_id']: r for r in rows}
        if len(mapping) != len(rows) or mapping.keys() != tasks:
            raise ValueError(f'Incomplete/duplicate task coverage: {path}')
        codes = {r['task_id']: r for r in read_jsonl(path.with_name(f'{split}-{variant}-code.jsonl'))}
        if codes.keys() != tasks:
            raise ValueError(f'Incomplete code coverage: {path}')
        for row in rows:
            if row['split'] != split or row['variant'] != variant:
                raise ValueError(f'Incorrect split/variant: {path}')
            if type(row['passed']) is not bool or row['passed'] != (row['returncode'] == 0):
                raise ValueError(f'Invalid pass/returncode: {path}')
            for field in FIELDS:
                if field not in protocol:
                    protocol[field] = row[field]
                elif protocol[field] != row[field]:
                    raise ValueError(f'Noncomparable {field}: {path}; do not mix coder revisions or protocols')
            original = codes[row['task_id']]
            if row['code'] != original['code'] or row['codegen_request_id'] != original['codegen_request_id']:
                raise ValueError(f'Result/code mismatch: {path}')
        score = statistics.mean(r['passed'] for r in rows)
        summary = json.loads(Path(str(path) + '.summary.json').read_text(encoding='utf-8'))
        if summary['tasks'] != len(tasks) or abs(summary['pass_at_1'] - score) > 1e-12:
            raise ValueError(f'Inconsistent summary: {path}')
        return mapping

    for seed in seeds:
        run = pattern.format(seed=seed)
        if Path(run).name != run:
            raise ValueError('Run pattern must be a directory name')
        for variant, label in [('original', 'Originale'), ('static', 'Statico'), ('base', 'Base'), ('sft', 'DPO')]:
            conditions[(label, seed)] = load(dpo_root, run, variant)
        for strategy in ('best', 'random', 'topk'):
            conditions[(f'SFT {strategy}', seed)] = load(sft_root, f'qwen3_8b-{strategy}-s{seed}', 'sft')
    labels = ['Originale', 'Statico', 'Base', 'SFT best', 'SFT random', 'SFT topk', 'DPO']
    table, comparisons = [], []
    for label in labels:
        scores = [statistics.mean(r['passed'] for r in conditions[(label, seed)].values()) for seed in seeds]
        table.append({'condition': label, 'pass_at_1': statistics.mean(scores),
                      'seed_scores': dict(zip(map(str, seeds), scores)), 'independent_tasks': len(tasks)})
        if label != 'DPO':
            deltas = [statistics.mean(int(conditions[('DPO', seed)][tid]['passed']) -
                       int(conditions[(label, seed)][tid]['passed']) for seed in seeds) for tid in sorted(tasks)]
            comparisons.append({'baseline': label, 'dpo_delta': statistics.mean(deltas),
                                'task_bootstrap_95_ci': task_bootstrap(deltas)})
    return {'split': split, 'seeds': seeds, 'independent_tasks': len(tasks), 'protocol': protocol,
            'table': table, 'comparisons': comparisons}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dpo-root', default='CodePromptOpt/finetuning/outputs/codegen/qwen3-coder-30b-dpo8b-v1/validation')
    p.add_argument('--sft-root', default='CodePromptOpt/finetuning/outputs/codegen/qwen3-coder-30b-h200-v1/validation')
    p.add_argument('--tasks', default='CodePromptOpt/data/tasks.jsonl')
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--run-pattern', default='qwen3_8b-dpo-s{seed}')
    p.add_argument('--report-prefix', default='CodePromptOpt/reports/dpo_vs_sft_qwen3_8b')
    args = p.parse_args()
    report = analyze(args.dpo_root, args.sft_root, args.tasks, args.seeds, args.split, args.run_pattern)
    prefix = Path(args.report_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with prefix.with_suffix('.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['condition', 'pass_at_1'] + [f'seed_{s}' for s in args.seeds]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report['table']:
            writer.writerow({'condition': row['condition'], 'pass_at_1': row['pass_at_1'],
                             **{f'seed_{s}': row['seed_scores'][str(s)] for s in args.seeds}})
    lines = ['# DPO vs SFT: Qwen3-8B', '',
             f"Split: {args.split}; task indipendenti: {report['independent_tasks']}; seed: {args.seeds}.", '',
             'Stesso code generator, revisione, decoding e protocollo sandbox verificati.', '',
             '| Condizione | Pass@1 medio | Pass@1 per seed |', '|---|---:|---|']
    for row in report['table']:
        scores = ' / '.join(f'{v * 100:.1f}%' for v in row['seed_scores'].values())
        lines.append(f"| {row['condition']} | {row['pass_at_1'] * 100:.1f}% | {scores} |")
    lines += ['', '| Confronto | Delta DPO (pp) | Bootstrap task 95% (pp) |', '|---|---:|---|']
    for row in report['comparisons']:
        lo, hi = row['task_bootstrap_95_ci']
        lines.append(f"| DPO - {row['baseline']} | {row['dpo_delta'] * 100:+.1f} | [{lo * 100:+.1f}, {hi * 100:+.1f}] |")
    lines += ['', 'I seed non sono nuovi problemi indipendenti. Gli intervalli sono descrittivi.',
              'La variante `sft` nelle run DPO rappresenta il modello DPO.',
              'Sono risultati di validation, se questo e lo split selezionato; non il test finale.']
    prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
