"""Only author-controlled toy solutions are executed by these unit tests."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ..common import read_jsonl, write_jsonl
from .prepare_tests import build, make_apps_harness


class HarnessTests(unittest.TestCase):
    def run_toy(self, solution, harness):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'solution.py').write_text(solution, encoding='utf-8')
            (root / 'check.py').write_text(harness, encoding='utf-8')
            return subprocess.run([sys.executable, '-B', 'check.py'], cwd=root, capture_output=True, timeout=10)

    def test_call_based_singleton_outputs_and_solution_class(self):
        harness = make_apps_harness({'fn_name': 'add', 'inputs': [[1, 2], [-1, 1]], 'outputs': [[3], [0]]})
        self.assertEqual(self.run_toy('def add(a,b): return a+b\n', harness).returncode, 0)
        self.assertNotEqual(self.run_toy('def add(a,b): return 0\n', harness).returncode, 0)
        self.assertEqual(self.run_toy('class Solution:\n def add(self,a,b): return a+b\n', harness).returncode, 0)

    def test_stdin_stdout_and_empty_fixture(self):
        harness = make_apps_harness({'inputs': ['2\n'], 'outputs': ['4\n']})
        self.assertEqual(self.run_toy('print(int(input())*2)\n', harness).returncode, 0)
        self.assertNotEqual(self.run_toy('print(99)\n', harness).returncode, 0)
        with self.assertRaises(ValueError):
            make_apps_harness({'inputs': [], 'outputs': []})

    def test_verified_mapping_and_duplicate_question_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = {'question': 'PROBLEM', 'input_output': json.dumps({'inputs': ['1\n'], 'outputs': ['2\n']}),
                      'url': 'https://example.com/problem/1', '_provenance': {'revision': 'commit'}}
            write_jsonl(root / 'apps.jsonl', [source])
            write_jsonl(root / 'humanevalplus.jsonl', [])
            tasks = [{'task_id': 't', 'benchmark': 'APPS', 'split': 'validation', 'problem': 'PROBLEM'}]
            report = build(tasks, root, root / 'verified.jsonl')
            self.assertEqual(report['matched_tasks'], 1)
            record = read_jsonl(root / 'verified.jsonl')[0]
            self.assertEqual(record['source_id'], source['url'])
            ast.parse(record['harness'])
            write_jsonl(root / 'apps.jsonl', [source, source])
            with self.assertRaises(ValueError):
                build(tasks, root, root / 'ambiguous.jsonl')
            self.assertFalse((root / 'ambiguous.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
