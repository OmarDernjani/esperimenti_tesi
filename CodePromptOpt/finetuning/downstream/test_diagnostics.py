"""Use only author-controlled fixtures; do not execute generated benchmark solutions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ..common import read_jsonl, write_jsonl
from .diagnose_failures import select


class DiagnosticTests(unittest.TestCase):
    def probe(self, script):
        probe = Path(__file__).resolve().parents[1] / 'sandbox_probe.py'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'toy_check.py'
            path.write_text(script, encoding='utf-8')
            result = subprocess.run([sys.executable, '-B', str(probe), str(path)],
                                    capture_output=True, text=True, timeout=15)
            return result, json.loads(result.stdout)

    def test_pass_and_bounded_stderr(self):
        result, record = self.probe('assert 2 + 2 == 4\n')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(record['category'], 'passed')
        result, record = self.probe("import sys\nsys.stderr.write('x' * 40000)\nraise AssertionError('toy failure')\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(record['category'], 'assertion_failed')
        self.assertTrue(record['stderr']['truncated'])
        self.assertLessEqual(len(record['stderr']['tail']), 32768)

    def test_nested_process_errors_are_visible(self):
        result, record = self.probe("import subprocess,sys\nsubprocess.run([sys.executable,'-c',\"raise AttributeError('toy missing function')\"],capture_output=True,text=True,check=True)\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('toy missing function', record['stderr']['tail'])
        self.assertEqual(record['category'], 'interface_or_attribute_error')

    def test_selection_deduplicates_without_modifying_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / 'tasks.jsonl', [{'task_id': 't', 'benchmark': 'APPS'}])
            row = {'task_id': 't', 'variant': 'original', 'passed': False, 'codegen_request_id': 'abc',
                   'code': 'author-controlled placeholder', 'returncode': 1}
            for name in ('run1', 'run2'):
                write_jsonl(root / 'results' / name / 'original-results.jsonl', [row])
            report = select(root / 'results', root / 'diagnostics', root / 'tasks.jsonl', ['original', 'static'], 'APPS')
            self.assertEqual(report['requests'], 1)
            self.assertEqual(report['original_failed_rows'], 2)
            self.assertNotIn('passed', read_jsonl(root / 'diagnostics/abc/diagnostic-code.jsonl')[0])
            self.assertEqual(read_jsonl(root / 'results/run1/original-results.jsonl'), [row])


if __name__ == '__main__':
    unittest.main()
