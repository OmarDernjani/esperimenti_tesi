"""Execute trusted test harnesses and untrusted generated code in isolated Podman containers."""
import argparse
import json
import statistics
import subprocess
import tempfile
import uuid
from pathlib import Path

from .common import digest, read_jsonl, write_jsonl


def run_case(code, harness, image, timeout):
    name = 'cpo-test-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='cpo-') as tmp:
        folder = Path(tmp)
        (folder / 'solution.py').write_text(code, encoding='utf-8')
        (folder / 'check.py').write_text(harness, encoding='utf-8')
        command = ['podman', 'run', '--rm', '--name', name, '--network=none', '--read-only',
                   '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=64',
                   '--memory=512m', '--cpus=1', '--user=65534:65534',
                   '--tmpfs=/tmp:rw,noexec,nosuid,size=64m', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                   '-v', f'{folder.resolve()}:/work:ro,Z', '-w', '/work', image, 'python', '-B', 'check.py']
        try:
            result = subprocess.run(command, capture_output=True, timeout=timeout)
            if result.returncode in (125, 126, 127):
                raise RuntimeError('Podman infrastructure error: ' + result.stderr.decode(errors='replace')[-2000:])
            return {'passed': result.returncode == 0, 'timeout': False, 'returncode': result.returncode}
        except subprocess.TimeoutExpired:
            return {'passed': False, 'timeout': True, 'returncode': None}
        finally:
            subprocess.run(['podman', 'rm', '-f', name], capture_output=True, timeout=30)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', required=True)
    p.add_argument('--tests', required=True, help='JSONL: task_id and harness (imports solution and asserts)')
    p.add_argument('--output', required=True)
    p.add_argument('--image', default='docker.io/library/python:3.11-slim')
    p.add_argument('--timeout', type=int, default=20)
    args = p.parse_args()
    samples = read_jsonl(args.samples)
    test_rows = read_jsonl(args.tests)
    tests = {r['task_id']: r['harness'] for r in test_rows}
    if len(tests) != len(test_rows):
        raise ValueError('Duplicate harness task IDs')
    if len({r['task_id'] for r in samples}) != len(samples):
        raise ValueError('Evaluate one variant / one generation per task per file')
    missing = {r['task_id'] for r in samples} - tests.keys()
    if missing or not samples:
        raise ValueError(f'Empty samples or missing test harnesses: {sorted(missing)}')
    results = []
    for row in samples:
        results.append({**row, 'tests_sha256': digest(args.tests), 'test_image': args.image,
                        'test_timeout': args.timeout,
                        **run_case(row['code'], tests[row['task_id']], args.image, args.timeout)})
        write_jsonl(args.output, results)
    summary = {'tasks': len(results), 'pass_at_1': statistics.mean(r['passed'] for r in results),
               'problem_verbatim_rate': statistics.mean(r['preserves_problem_verbatim'] for r in results),
               'mean_prompt_chars': statistics.mean(r['prompt_chars'] for r in results),
               'tests_sha256': digest(args.tests), 'image': args.image}
    Path(args.output + '.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
