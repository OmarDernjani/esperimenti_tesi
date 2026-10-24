"""Offline structural audit of generated prompts; no code/model execution."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

from ..common import digest, read_jsonl
from .run import atomic_json, discover


def inspect_prompt(prompt, problem, entry_point):
    words = prompt.split()
    eightgrams = Counter(tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7)))
    repeats = max(eightgrams.values(), default=0)
    return {'preserved_exact': problem in prompt,
            'preserved_whitespace_normalized': ' '.join(problem.split()) in ' '.join(words),
            'entry_point_checked': bool(entry_point),
            'entry_point_present': bool(entry_point and re.search(r'\b' + re.escape(entry_point) + r'\b', prompt)),
            'max_repeated_8gram': repeats, 'repetition_warning': repeats >= 8,
            'empty': not prompt.strip()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', default='CodePromptOpt/finetuning/outputs/runs')
    p.add_argument('--tasks', default='CodePromptOpt/data/tasks.jsonl')
    p.add_argument('--split', default='validation', choices=['validation', 'test'])
    p.add_argument('--models', nargs='+', default=['qwen3_4b', 'qwen3_8b', 'qwen3_14b'])
    p.add_argument('--report-prefix', default='CodePromptOpt/reports/prompt_audit_2026-10-24')
    args = p.parse_args()
    root = Path(args.input_root)
    expected = [f'{model}-{strategy}-s{seed}' for model in args.models
                for strategy in ('best', 'random', 'topk') for seed in (42, 43, 44)]
    selected = [name for name in expected if (root / name).is_dir()]
    if not selected:
        raise ValueError('No selected model runs found')
    files, coverage = discover(root, args.tasks, args.split, selected)
    coverage['missing_run_directories'] = sorted(set(expected) - set(selected))
    tasks = {r['task_id']: r for r in read_jsonl(args.tasks)}
    grouped, records, fingerprints = defaultdict(Counter), [], {}
    for run, variant, path, rows in files:
        fingerprints[str(path)] = digest(path)
        model = run.rsplit('-', 2)[0]
        for row in rows:
            task = tasks[row['task_id']]
            flags = inspect_prompt(row['prompt'], task['problem'], task.get('entry_point'))
            detail = {'run': run, 'variant': variant, 'task_id': row['task_id'],
                      'truncated': bool(row.get('truncated')), **flags,
                      'prompt_chars': len(row['prompt']), 'file': str(path)}
            records.append(detail)
            counts = grouped[(model, variant)]
            counts['rows'] += 1
            counts['preserved_exact'] += flags['preserved_exact']
            counts['preserved_whitespace_normalized'] += flags['preserved_whitespace_normalized']
            counts['truncated'] += detail['truncated']
            counts['repetition_warning'] += flags['repetition_warning']
            counts['entry_point_checked'] += flags['entry_point_checked']
            counts['entry_point_missing'] += flags['entry_point_checked'] and not flags['entry_point_present']
            counts['empty'] += flags['empty']
    training = []
    for name in selected:
        folder = root / name
        history_path = folder / 'history.json'
        history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
        evaluations = [r for r in history if 'eval_loss' in r]
        experiment_path = folder / 'experiment.json'
        experiment = json.loads(experiment_path.read_text(encoding='utf-8')) if experiment_path.exists() else {}
        training.append({'run': name, 'adapter_config_present': (folder / 'adapter/adapter_config.json').is_file(),
                         'adapter_weights_present': any((folder / 'adapter').glob('adapter_model.*')),
                         'experiment_present': experiment_path.exists(), 'smoke_run': experiment.get('smoke_run'),
                         'first_eval_loss': evaluations[0]['eval_loss'] if evaluations else None,
                         'best_eval_loss': min(r['eval_loss'] for r in evaluations) if evaluations else None,
                         'final_step': history[-1].get('step') if history else None})
    summary = [{'model': model, 'variant': variant, **counts} for (model, variant), counts in sorted(grouped.items())]
    prefix = Path(args.report_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(str(prefix) + '.json', {'input_root': str(root), 'models': args.models, 'coverage': coverage,
                'summary': summary, 'training': training, 'records': records, 'sha256': fingerprints})
    with Path(str(prefix) + '.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ['# Audit dei prompt generati', '',
             f'Input: `{root}`. Split: `{args.split}`. Modelli: {", ".join(args.models)}.', '',
             'Phi-4 escluso intenzionalmente: non addestrato. Nessun file di input modificato.', '',
             f'Copertura: {coverage["runs"]} run, {coverage["files"]} file, {coverage["rows"]} righe; '
             f'{len(coverage["coverage_issues"])} problemi di copertura file; '
             f'{len(coverage["missing_run_directories"])} cartelle run mancanti.', '',
             '| Modello | Variante | Righe | Problema esatto | Problema normalizzato | Troncati | Ripetizioni* | Firma assente** |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["model"]} | {r["variant"]} | {r["rows"]} | {r["preserved_exact"]} | '
                     f'{r["preserved_whitespace_normalized"]} | {r["truncated"]} | {r["repetition_warning"]} | '
                     f'{r["entry_point_missing"]}/{r["entry_point_checked"]} |')
    lines += ['', '*Euristica: una sequenza di 8 parole compare almeno 8 volte; non e una prova di degenerazione.',
              '**Si controlla il nome della funzione solo quando il dataset lo documenta. '
              'La presenza del nome non prova che interfaccia e semantica siano corrette.', '',
              'Normalizzazione: collasso degli spazi e dei newline, senza riscrivere parole o simboli. '
              'Un problema non preservato puo essere riformulato correttamente: non equivale automaticamente a errore funzionale.', '',
              '## Dettaglio SFT Qwen3-14B per run', '',
              '| Run | Righe | Problema normalizzato | Troncati | Ripetizioni* |', '|---|---:|---:|---:|---:|']
    for name in selected:
        if not name.startswith('qwen3_14b-'):
            continue
        rows = [r for r in records if r['run'] == name and r['variant'] == 'sft']
        lines.append(f'| {name} | {len(rows)} | {sum(r["preserved_whitespace_normalized"] for r in rows)} | '
                     f'{sum(r["truncated"] for r in rows)} | {sum(r["repetition_warning"] for r in rows)} |')
    lines += ['', '## Training', '', '| Run | Primo eval loss | Miglior eval loss | Step finale |', '|---|---:|---:|---:|']
    for r in training:
        if r['run'].startswith('qwen3_14b-'):
            lines.append(f'| {r["run"]} | {r["first_eval_loss"]} | {r["best_eval_loss"]} | {r["final_step"]} |')
    lines += ['', 'Questa analisi non esegue modelli o codice e non misura pass@1. '
              'Loss bassa, conservazione del testo e assenza di warning non dimostrano un miglioramento funzionale.',
              'CSV: una riga per prompt. JSON: copertura, contatori, artefatti training e checksum dei file esaminati.', '']
    Path(str(prefix) + '.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'coverage': coverage, 'summary': summary, 'reports': str(prefix)}, indent=2))


if __name__ == '__main__':
    main()
