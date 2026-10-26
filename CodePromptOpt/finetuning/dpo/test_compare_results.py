import json
from pathlib import Path
import tempfile
import unittest

from ..common import write_jsonl
from .compare_results import analyze


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks = self.root / 'tasks.jsonl'
        write_jsonl(self.tasks, [{'task_id': 't', 'split': 'validation'}])
        for variant in ('original', 'static', 'base', 'sft'):
            self.make('dpo', 'qwen3_8b-dpo-s42', variant, variant == 'sft')
        for strategy in ('best', 'random', 'topk'):
            self.make('sft', f'qwen3_8b-{strategy}-s42', 'sft', False)

    def make(self, folder, run, variant, passed):
        row = {'task_id': 't', 'split': 'validation', 'variant': variant, 'passed': passed,
               'returncode': 0 if passed else 1, 'code': 'fixture', 'codegen_request_id': variant,
               'codegen_model': 'coder', 'codegen_parameters': {'temperature': 0},
               'codegen_server_metadata': {'model_revision': 'same'}, 'target_label': 'target',
               'tests_sha256': 'same', 'test_image': 'same', 'test_timeout': 20}
        path = self.root / folder / run / f'validation-{variant}-results.jsonl'
        write_jsonl(path, [row])
        write_jsonl(path.with_name(f'validation-{variant}-code.jsonl'), [row])
        Path(str(path) + '.summary.json').write_text(json.dumps({'tasks': 1, 'pass_at_1': int(passed)}))

    def analyze(self):
        return analyze(self.root / 'dpo', self.root / 'sft', self.tasks, [42])

    def test_paired_improvement(self):
        result = self.analyze()
        self.assertEqual(len(result['table']), 7)
        self.assertEqual(result['comparisons'][3]['dpo_delta'], 1)

    def test_revision_mismatch_rejected(self):
        path = self.root / 'sft/qwen3_8b-best-s42/validation-sft-results.jsonl'
        row = json.loads(path.read_text())
        row['codegen_server_metadata']['model_revision'] = 'different'
        write_jsonl(path, [row])
        with self.assertRaisesRegex(ValueError, 'Noncomparable'):
            self.analyze()

    def test_duplicate_tasks_rejected(self):
        path = self.root / 'dpo/qwen3_8b-dpo-s42/validation-sft-results.jsonl'
        row = json.loads(path.read_text())
        write_jsonl(path, [row, row])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.analyze()


if __name__ == '__main__':
    unittest.main()
