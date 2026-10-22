"""Summarize observed perfect-score prompt texts per task, without new evaluation."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('CodePromptOpt/data'))
    parser.add_argument('--output-dir', type=Path, default=Path('CodePromptOpt/reports'))
    args = parser.parse_args()
    tasks = read(args.data_dir / 'tasks.jsonl')
    candidates = {c['candidate_id']: c for c in read(args.data_dir / 'candidates.jsonl')}
    runs = {r['run_id']: r for r in read(args.data_dir / 'runs.jsonl')}
    observations = defaultdict(lambda: defaultdict(set))
    for e in read(args.data_dir / 'evaluations.jsonl'):
        if e['score'] == 1:
            observations[e['candidate_id']][e['evaluation_split']].add(e['generated_code'])
    groups = defaultdict(list)
    for cid, c in candidates.items():
        groups[c['task_id']].append((cid, c, runs[c['run_id']]))
    rows = []
    for task in sorted(tasks, key=lambda t: (t['benchmark'], t['local_problem_indices'][0], t['task_id'])):
        perfect = {key: set() for key in ('dev', 'test', 'both', 'optimized_test')}
        candidate_test = set()
        models = defaultdict(set)
        for cid, c, run in groups[task['task_id']]:
            obs = observations[cid]
            if obs['dev']:
                perfect['dev'].add(c['prompt'])
            if obs['test']:
                perfect['test'].add(c['prompt'])
                candidate_test.add(cid)
                models[run['target_model']].add(c['prompt'])
                if run['optimizer_method'] in ('APE', 'APO'):
                    perfect['optimized_test'].add(c['prompt'])
            if (obs['dev'] & obs['test']) - {None}:
                perfect['both'].add(c['prompt'])
        rows.append({'task_id': task['task_id'], 'benchmark': task['benchmark'],
                     'problem_label': task['source_id'] or ('APPS/local-' + str(task['local_problem_indices'][0])),
                     'difficulty': task['difficulty'] or '', 'split': task['split'],
                     'unique_prompts_perfect_dev': len(perfect['dev']),
                     'unique_prompts_perfect_test': len(perfect['test']),
                     'unique_prompts_perfect_dev_and_test_same_code': len(perfect['both']),
                     'unique_optimized_prompts_perfect_test': len(perfect['optimized_test']),
                     'candidates_perfect_test': len(candidate_test),
                     'qwen3.5_unique_prompts_perfect_test': len(models['qwen3.5']),
                     'qwen3.5-coder_unique_prompts_perfect_test': len(models['qwen3.5-coder'])})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'correct_prompts_per_task.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Prompt con correttezza osservata per problema', '',
             'Conteggi di testi di prompt distinti esatti per task. Un prompt viene contato se almeno una sua soluzione salvata ha score 1 sullo split indicato. Si aggregano tutti i metodi e le configurazioni: il successo non implica correttezza universale o successo su ogni modello. Le colonne per modello sono nel CSV.', '',
             'Dev e test sono separati. «Entrambi» richiede la stessa completion salvata nello stesso candidato/run con score 1 su dev e test. Un prompt senza valutazione test non viene contato nel test; ciò non implica che sia errato. «Ottimizzati test» considera solo APE/APO. I numeri non applicano i filtri di ammissibilità SFT/DPO.', '',
             'APPS/local-N è l’indice locale dell’esperimento, non l’ID ufficiale APPS. Questo report include i task test interni per audit: non usarlo per selezionare etichette o regolare il modello.', '',
             '| Problema | Difficoltà | Dev 100% | Test 100% | Entrambi | Ottimizzati test |',
             '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {problem_label} | {difficulty} | {unique_prompts_perfect_dev} | {unique_prompts_perfect_test} | {unique_prompts_perfect_dev_and_test_same_code} | {unique_optimized_prompts_perfect_test} |'.format(**row))
    (args.output_dir / 'correct_prompts_per_task.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary = {}
    for label in ('all', 'APPS', 'HumanEval+'):
        subset = [r for r in rows if label == 'all' or r['benchmark'] == label]
        summary[label] = {'tasks': len(subset)}
        for key in ('unique_prompts_perfect_dev', 'unique_prompts_perfect_test',
                    'unique_prompts_perfect_dev_and_test_same_code', 'unique_optimized_prompts_perfect_test'):
            values = [r[key] for r in subset]
            summary[label][key] = {'total': sum(values), 'mean': statistics.mean(values),
                                   'median': statistics.median(values), 'min': min(values), 'max': max(values),
                                   'tasks_with_zero': [r['problem_label'] for r in subset if r[key] == 0]}
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
