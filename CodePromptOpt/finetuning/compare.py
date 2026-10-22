"""Paired task bootstrap for the downstream pass@1 difference."""
import argparse
import json
import random
import statistics
from .common import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    args = p.parse_args()
    files = [read_jsonl(f) for f in (args.baseline, args.candidate)]
    maps = [{r['task_id']: r for r in rows} for rows in files]
    if any(len(m) != len(rows) for m, rows in zip(maps, files)) or maps[0].keys() != maps[1].keys() or not maps[0]:
        raise ValueError('Need identical nonempty task sets with one generation per task')
    differences = []
    for key in sorted(maps[0]):
        a, b = maps[0][key], maps[1][key]
        for field in ('codegen_model', 'codegen_parameters', 'target_label', 'split',
                      'tests_sha256', 'test_image', 'test_timeout'):
            if a[field] != b[field]:
                raise ValueError(f'Noncomparable {field}')
        differences.append(int(b['passed']) - int(a['passed']))
    # A paired bootstrap resamples problems, not prompt rows.
    rng = random.Random(42)
    bootstrap = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    print(json.dumps({'tasks': len(differences), 'pass_at_1_delta': statistics.mean(differences),
                      'paired_bootstrap_95_ci': [bootstrap[250], bootstrap[9749]]}, indent=2))


if __name__ == '__main__':
    main()
