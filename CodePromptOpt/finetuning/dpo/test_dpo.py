"""CPU tests for preference validity, leakage checks, and token-budget handling."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .prepare import select_pairs
from .train import tokenize_pairs
from .pipeline import main as pipeline_main


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        return messages[0]['content']


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [{'task_id': 't', 'split': 'train', 'problem': 'P'}]
        self.row = dict(task_id='t', split='train', input='P', instruction='Rewrite', target_model='coder',
            chosen='P good', rejected='P bad', chosen_dev_score=1.0, rejected_dev_score=0.0,
            chosen_evaluation_ids=['e1'], rejected_evaluation_ids=['e2'],
            comparison_group_id='g', sample_id='a')

    def select(self, rows):
        return select_pairs(self.tasks, rows, 'train', 'coder')

    def test_leakage_is_fatal(self):
        row = dict(self.row, split='validation')
        with self.assertRaisesRegex(ValueError, 'leakage'):
            self.select([row])

    def test_canonical_problem_is_checked(self):
        with self.assertRaisesRegex(ValueError, 'canonical'):
            self.select([dict(self.row, input='another problem')])

    def test_ties_and_missing_provenance(self):
        self.assertEqual(len(self.select([dict(self.row, rejected_dev_score=1.0)])[0]), 0)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.select([dict(self.row, chosen_evaluation_ids=[])])

    def test_invalid_score(self):
        with self.assertRaisesRegex(ValueError, 'score'):
            self.select([dict(self.row, chosen_dev_score=float('nan'))])

    def test_dedup_and_margin_cap(self):
        lower = dict(self.row, sample_id='b', chosen='P alternative', chosen_dev_score=0.5)
        selected, excluded = self.select([lower, self.row, copy.deepcopy(self.row)])
        self.assertEqual([r['sample_id'] for r in selected], ['a'])
        self.assertEqual({r['reason'] for r in excluded}, {'per_task_cap', 'duplicate_pair'})

    def test_chosen_preserves_original(self):
        rows, excluded = self.select([dict(self.row, chosen='wrong task')])
        self.assertFalse(rows)
        self.assertEqual(excluded[0]['reason'], 'chosen_does_not_preserve_problem')

    def test_token_budget_includes_eos_no_truncation(self):
        tokenizer = FakeTokenizer()
        encoded, excluded = tokenize_pairs(tokenizer, [self.row], 1000)
        self.assertFalse(excluded)
        self.assertEqual(encoded[0]['chosen_input_ids'][-1], 0)
        limit = len(encoded[0]['prompt_input_ids']) + len(encoded[0]['chosen_input_ids'])
        self.assertEqual(len(tokenize_pairs(tokenizer, [self.row], limit)[0]), 1)
        self.assertEqual(len(tokenize_pairs(tokenizer, [self.row], limit - 1)[0]), 0)


class PipelineTests(unittest.TestCase):
    def invoke(self, root, *args):
        with patch('sys.argv', ['pipeline', *args, '--root', str(root), '--config', 'config.json']), \
             patch('CodePromptOpt.finetuning.dpo.pipeline.subprocess.run') as call:
            pipeline_main()
            return call.call_args_list

    def test_smoke_has_separate_output_and_two_steps(self):
        calls = self.invoke('output', 'smoke', '--seed', '43')
        command = calls[0].args[0]
        self.assertIn(str(Path('output/smoke/qwen3_8b-dpo-s43')), command)
        self.assertEqual(command[-2:], ['--max-steps', '2'])

    def test_unsafe_run_name(self):
        with self.assertRaises(SystemExit):
            self.invoke('output', 'train', '--run-name', '../escape')

    def test_generate_requires_complete_training(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'not complete'):
                self.invoke(temp, 'generate')

    def test_generate_dpo_adapter_and_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'runs/qwen3_8b-dpo-s42'
            run.mkdir(parents=True)
            (run / 'metrics.json').write_text('{}')
            (run / 'audit.json').write_text(json.dumps({'initial_adapter': None}))
            calls = self.invoke(temp, 'generate')
            self.assertEqual(len(calls), 4)
            self.assertNotIn('--adapter', calls[2].args[0])
            self.assertEqual(calls[3].args[0][-2:], ['--adapter', str(run / 'adapter')])
            (run / 'validation-base-prompts.jsonl').write_text('partial')
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                self.invoke(temp, 'generate')


if __name__ == '__main__':
    unittest.main()
