"""Validate sandbox exports and summarize paired results without executing code."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

from ..common import digest, read_jsonl
from .run import atomic_json


def task_bootstrap(deltas, samples=10000):
    rng = random.Random(42)
    values = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
    return [values[int(samples * 0.025)], values[int(samples * 0.975) - 1]]


def analyze(root, tasks_path, tests_path, split='validation'):
    root = Path(root)
    tasks = {r['task_id']: r for r in read_jsonl(tasks_path) if r['split'] == split}
    test_rows = {r['task_id']: r for r in read_jsonl(tests_path)}
    test_bytes = Path(tests_path).read_bytes()
    test_hashes = {hashlib.sha256(test_bytes).hexdigest(), hashlib.sha256(test_bytes.replace(b'\r\n', b'\n')).hexdigest()}
    common, conditions, metrics, warnings, records = {}, {}, [], [], []
    duplicates = defaultdict(set)
    expected_runs = [f'{m}-{s}-s{seed}' for m in ('qwen3_4b', 'qwen3_8b', 'qwen3_14b')
                     for s in ('best', 'random', 'topk') for seed in (42, 43, 44)]
    checked_harnesses = 0
    for run in expected_runs:
        model, strategy, seed_label = run.rsplit('-', 2)
        for variant in ('original', 'static', 'base', 'sft'):
            path = root / run / f'{split}-{variant}-results.jsonl'
            rows = read_jsonl(path)
            row_map = {r['task_id']: r for r in rows}
            if len(row_map) != len(rows) or row_map.keys() != tasks.keys():
                raise ValueError(f'Non-identical complete task coverage: {path}')
            codes = {r['task_id']: r for r in read_jsonl(root / run / f'{split}-{variant}-code.jsonl')}
            for row in rows:
                if row['split'] != split or row['variant'] != variant or type(row['passed']) is not bool:
                    raise ValueError(f'Invalid result metadata: {path}')
                if row['passed'] != (row['returncode'] == 0):
                    raise ValueError(f'Pass/exit-code disagreement: {path}')
                if row['tests_sha256'] not in test_hashes:
                    raise ValueError('Recorded test checksum differs from local tests beyond LF/CRLF')
                for field in ('codegen_model', 'codegen_parameters', 'codegen_server_metadata', 'target_label',
                              'test_image', 'test_timeout', 'tests_sha256'):
                    if field not in common:
                        common[field] = row[field]
                    elif common[field] != row[field]:
                        raise ValueError(f'Noncomparable metadata {field}: {path}')
                original = codes[row['task_id']]
                if row['code'] != original['code'] or row['codegen_request_id'] != original['codegen_request_id']:
                    raise ValueError(f'Result not aligned to code generation: {path}')
                job = root / run / f'{split}-{variant}-jobs' / str(row['job_id']) / 'check.py'
                if job.exists():
                    if job.read_text(encoding='utf-8') != test_rows[row['task_id']]['harness']:
                        raise ValueError(f'Executed harness differs: {job}')
                    checked_harnesses += 1
                duplicates[row['codegen_request_id']].add((row['code'], row['passed']))
                records.append({**row, 'run': run, 'model': model, 'strategy': strategy,
                                'training_seed': int(seed_label[1:]), 'benchmark': tasks[row['task_id']]['benchmark']})
            score = statistics.mean(r['passed'] for r in rows)
            summary = json.loads(Path(str(path) + '.summary.json').read_text(encoding='utf-8'))
            if summary['tasks'] != len(tasks) or abs(summary['pass_at_1'] - score) > 1e-12:
                raise ValueError(f'Summary inconsistent with rows: {path}')
            conditions[(model, strategy, int(seed_label[1:]), variant)] = row_map
            metrics.append({'model': model, 'strategy': strategy, 'seed': int(seed_label[1:]),
                'variant': variant, 'tasks': len(rows), 'passed': sum(r['passed'] for r in rows), 'pass_at_1': score})
    for request, values in duplicates.items():
        if len(values) != 1:
            warnings.append(f'Repeated request has differing code/outcome: {request}')
    grouped, comparisons = [], []
    for model in ('qwen3_4b', 'qwen3_8b', 'qwen3_14b'):
        for strategy in ('best', 'random', 'topk'):
            item = {'model': model, 'strategy': strategy}
            for variant in ('original', 'static', 'base', 'sft'):
                scores = [next(r['pass_at_1'] for r in metrics if r['model'] == model and
                          r['strategy'] == strategy and r['seed'] == seed and r['variant'] == variant)
                          for seed in (42, 43, 44)]
                item[variant] = statistics.mean(scores)
                item[variant + '_seed_scores'] = scores
                item[variant + '_seed_sd'] = statistics.stdev(scores)
            grouped.append(item)
            for baseline in ('original', 'static', 'base'):
                # Average seed differences within each problem, then resample ten
                # independent problem clusters, never thirty task/seed rows.
                deltas = [statistics.mean(int(conditions[(model, strategy, seed, 'sft')][tid]['passed']) -
                                         int(conditions[(model, strategy, seed, baseline)][tid]['passed'])
                                         for seed in (42, 43, 44)) for tid in sorted(tasks)]
                comparisons.append({'model': model, 'strategy': strategy, 'baseline': baseline,
                    'sft_delta': statistics.mean(deltas), 'task_cluster_bootstrap_95_ci': task_bootstrap(deltas),
                    'independent_tasks': len(tasks), 'training_seeds': 3})
    by_task = []
    for tid, task in tasks.items():
        item = {'task_id': tid, 'benchmark': task['benchmark'], 'source_id': task.get('source_id'),
                'entry_point': task.get('entry_point')}
        for variant in ('original', 'static', 'base', 'sft'):
            rows = [r for r in records if r['task_id'] == tid and r['variant'] == variant]
            item[variant + '_passes'] = sum(r['passed'] for r in rows)
            item[variant + '_rows'] = len(rows)
        by_task.append(item)
    return {'scope': split, 'independent_tasks': len(tasks), 'run_count': len(expected_runs),
            'file_count': len(metrics), 'result_rows': len(records), 'unique_codegen_requests': len(duplicates),
            'timeouts': sum(r['timeout'] for r in records), 'checked_executed_harnesses': checked_harnesses,
            'common_metadata': common, 'local_tests_sha256': digest(tests_path),
            'local_tests_lf_sha256': hashlib.sha256(test_bytes.replace(b'\r\n', b'\n')).hexdigest(),
            'warnings': warnings, 'groups': grouped, 'per_seed': metrics,
            'comparisons': comparisons, 'by_task': by_task}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='CodePromptOpt/finetuning/outputs/codegen/qwen25-coder-7b-h200-v1/validation')
    p.add_argument('--tasks', default='CodePromptOpt/data/tasks.jsonl')
    p.add_argument('--tests', default='verified_tests.jsonl')
    p.add_argument('--report-prefix', default='CodePromptOpt/reports/codegen_validation_2026-10-25')
    args = p.parse_args()
    result = analyze(args.root, args.tasks, args.tests)
    prefix = Path(args.report_prefix)
    atomic_json(str(prefix) + '.json', result)
    with Path(str(prefix) + '.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['per_seed'][0]))
        writer.writeheader()
        writer.writerows(result['per_seed'])
    lines = ['# Valutazione funzionale sulla validation', '',
        f'{result["run_count"]} run, {result["file_count"]} file risultati e riepiloghi, '
        f'{result["result_rows"]} righe; {result["independent_tasks"]} task indipendenti e 3 seed di training.',
        f'Code generator: `{result["common_metadata"]["codegen_model"]}`. '
        f'{result["unique_codegen_requests"]} richieste uniche; i controlli duplicati non sono repliche indipendenti.', '',
        '## Controlli di integrita', '',
        '- Tutti i file coprono lo stesso insieme di dieci task; i riepiloghi coincidono con i record.',
        '- Modello, decoding, revisione, immagine vLLM, suite, immagine sandbox e timeout sono identici.',
        f'- Verificati {result["checked_executed_harnesses"]} check.py esportati contro gli harness locali.',
        '- Il checksum dei test differisce fra Windows e Linux solo per CRLF/LF; quello normalizzato LF coincide.',
        f'- Timeout esterni registrati: {result["timeouts"]}. Un timeout interno APPS puo apparire come exit 1.',
        f'- Disaccordi sulle richieste riutilizzate: {len(result["warnings"])}.', '',
        '## Pass@1 medio sui tre seed di training', '',
        'Valori percentuali. Ogni seed ha dieci task: la media non rappresenta trenta problemi distinti.', '',
        '| Optimizer | Supervisione | Originale | Statico | Base | SFT | Delta SFT-base (pp) | SFT per seed 42/43/44 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in result['groups']:
        seeds = '/'.join(f'{100*x:.0f}' for x in r['sft_seed_scores'])
        lines.append(f'| {r["model"]} | {r["strategy"]} | {r["original"]*100:.1f} | {r["static"]*100:.1f} | '
                     f'{r["base"]*100:.1f} | {r["sft"]*100:.1f} | {(r["sft"]-r["base"])*100:+.1f} | {seeds} |')
    lines += ['', '## Confronto appaiato SFT-base', '',
        'Intervalli bootstrap descrittivi (10000 ricampionamenti di task), dopo media dei seed dentro ogni task. '
        'Non si trattano i seed o le copie dei controlli come nuovi task.', '',
        '| Optimizer | Supervisione | Delta (pp) | Intervallo 95% (pp) |', '|---|---|---:|---|']
    for r in result['comparisons']:
        if r['baseline'] == 'base':
            lo, hi = r['task_cluster_bootstrap_95_ci']
            lines.append(f'| {r["model"]} | {r["strategy"]} | {r["sft_delta"]*100:+.1f} | [{lo*100:+.1f}, {hi*100:+.1f}] |')
    lines += ['', '## Risultati per task', '',
        'I numeratori sotto contano righe dei run: gli original/static sono ripetuti e non indipendenti.', '',
        '| Task | Benchmark | Funzione/ID | Originale | Statico | Base | SFT |', '|---|---|---|---:|---:|---:|---:|']
    for r in result['by_task']:
        label = r['entry_point'] or r['source_id'] or r['task_id']
        values = [f'{r[v+"_passes"]}/{r[v+"_rows"]}' for v in ('original', 'static', 'base', 'sft')]
        lines.append(f'| {r["task_id"]} | {r["benchmark"]} | {label} | ' + ' | '.join(values) + ' |')
    lines += ['', '## Interpretazione e limiti', '',
        '- Nessuna configurazione SFT migliora il rispettivo optimizer base in questa validation. '
        'Tutte restano sotto originale/statico (50%). Non emerge evidenza del beneficio cercato con il protocollo attuale.',
        '- Originale/statico superano tutti e cinque i task HumanEval+ ma nessuno dei cinque APPS. '
        'Anche ogni altra variante fallisce tutti gli APPS: analizzare questi fallimenti prima di estendere le conclusioni.',
        '- Gli harness sono coerenti con i file preparati e i test sintetici del runner, ma nei risultati non e registrato '
        'stderr: exit 1 non distingue algoritmo errato, interfaccia, eccezioni o timeout del sottoprocesso. '
        'Non si puo attribuire ogni fallimento a uno specifico motivo da questi record.',
        '- Alcune soluzioni APPS originali hanno problemi visibili nel testo, come nomi funzione diversi dalla specifica '
        'o algoritmi che non gestiscono tutti i casi; questa ispezione non sostituisce una diagnosi runtime.',
        '- Le ripetizioni/troncamenti dei prompt SFT rilevati nel precedente audit sono un limite plausibile, '
        'non una spiegazione causale dimostrata di ogni errore.',
        '- Sono dieci task di validation, non il test finale. Non usare questi numeri come stima definitiva di generalizzazione '
        'o evidenza universale contro SFT. Conservare il test per un protocollo fissato.', '',
        'CSV: valori per ogni seed. JSON: metadata, controlli, confronti appaiati per tutte le baseline e dettaglio task.', '']
    Path(str(prefix) + '.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'runs': result['run_count'], 'files': result['file_count'], 'rows': result['result_rows'],
                      'unique_requests': result['unique_codegen_requests'], 'warnings': result['warnings'], 'report': str(prefix)}, indent=2))


if __name__ == '__main__':
    main()
