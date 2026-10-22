"""Read-only source inspection; no experiment modules are imported."""
import collections
import json
from pathlib import Path

counts = collections.Counter()
tasks = set()
schemas = collections.defaultdict(set)
for path in sorted(Path('dataset_code_prompt').rglob('*')):
    if path.suffix not in ('.json', '.jsonl'):
        continue
    raw = path.read_text(encoding='utf-8-sig')
    decoder = json.JSONDecoder()
    docs, pos = [], 0
    while pos < len(raw):
        while pos < len(raw) and raw[pos].isspace():
            pos += 1
        if pos == len(raw):
            break
        value, pos = decoder.raw_decode(raw, pos)
        docs.append(value)
    rows = [r for doc in docs for r in (doc if isinstance(doc, list) else [doc])]
    print(path.as_posix(), 'documents', len(docs), 'rows', len(rows))
    for r in rows:
        if not isinstance(r, dict) or 'question' not in r:
            continue
        tasks.add(('HE' if 'task_id' in r else 'APPS', r['question']))
        counts['task_configurations'] += 1
        for method in ['zero_shot', 'human_prompt', 'baseline', 'APE', 'APO']:
            if method not in r:
                continue
            counts['runs'] += 1
            b = r[method]
            schemas[method].update(b)
            observations = []
            for it in b.get('iterations', []):
                for key in ['candidates', 'all_candidates', 'beam']:
                    for c in it.get(key, []):
                        schemas[method+'/'+key].update(c)
                        observations.append(c)
            observations += b.get('final_population', []) + b.get('final_pool', [])
            for c in observations:
                counts['candidate_mentions'] += 1
                for field in ['dev_score', 'test_score', 'code']:
                    counts[field+'_mentions'] += field in c
    if rows and 'question' in rows[0]:
        print(' first task', rows[0].get('task_id', rows[0].get('problem_idx')), 'n_dev', sorted(set(r.get('n_dev', 0) for r in rows)))
print('UNIQUE_TASKS', len(tasks), 'COUNTS', dict(counts))
print('SCHEMAS', {k: sorted(v) for k, v in schemas.items()})
