"""Prepare/collect isolated test jobs; the host needs Bash + Podman, not Python."""
import argparse
import json
from pathlib import Path
import statistics

from .common import digest, read_jsonl, write_jsonl


def workspace_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(Path('/workspace')):
        raise ValueError('Evaluation paths must be inside the mounted /workspace repository')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'collect'])
    p.add_argument('--jobs', required=True)
    p.add_argument('--samples')
    p.add_argument('--tests')
    p.add_argument('--image')
    p.add_argument('--output')
    args = p.parse_args()
    jobs = workspace_path(args.jobs)
    if args.stage == 'prepare':
        if not all((args.samples, args.tests, args.image)):
            p.error('prepare requires --samples, --tests, --image')
        samples = read_jsonl(workspace_path(args.samples))
        tests_rows = read_jsonl(workspace_path(args.tests))
        tests = {r['task_id']: r['harness'] for r in tests_rows}
        if len(tests) != len(tests_rows) or not samples:
            raise ValueError('Empty samples or duplicate test IDs')
        if len({r['task_id'] for r in samples}) != len(samples):
            raise ValueError('Expected one generation per task')
        if {r['task_id'] for r in samples} - tests.keys():
            raise ValueError('Missing harnesses')
        jobs.mkdir(parents=True, exist_ok=False)
        rows = []
        for idx, row in enumerate(samples):
            folder = jobs / str(idx)
            folder.mkdir(mode=0o755)
            for name, text in (('solution.py', row['code']), ('check.py', tests[row['task_id']])):
                file = folder / name
                file.write_text(text, encoding='utf-8')
                file.chmod(0o644)
            rows.append({**row, 'job_id': idx, 'tests_sha256': digest(args.tests),
                         'test_image': args.image, 'test_timeout': 20})
        write_jsonl(jobs / 'manifest.jsonl', rows)
        (jobs / 'jobs.txt').write_text(''.join(f'{idx}\n' for idx in range(len(rows))), encoding='utf-8')
    else:
        if not args.output:
            p.error('collect requires --output')
        rows = read_jsonl(jobs / 'manifest.jsonl')
        for row in rows:
            rc = int((jobs / str(row['job_id']) / 'exit_code').read_text())
            if rc in (125, 126, 127):
                raise ValueError('Infrastructure failure')
            row.update(passed=rc == 0, timeout=rc in (124, 137), returncode=rc)
        output = workspace_path(args.output)
        write_jsonl(output, rows)
        summary = {'tasks': len(rows), 'pass_at_1': statistics.mean(r['passed'] for r in rows),
                   'problem_verbatim_rate': statistics.mean(r['preserves_problem_verbatim'] for r in rows),
                   'mean_prompt_chars': statistics.mean(r['prompt_chars'] for r in rows)}
        Path(str(output) + '.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
