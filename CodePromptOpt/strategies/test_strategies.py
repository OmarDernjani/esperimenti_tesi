import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from CodePromptOpt.finetuning.common import read_jsonl
from CodePromptOpt.finetuning.downstream.run import discover
from .core import STRATEGIES, build_prompt, fingerprint, validate_selection
from .pipeline import analyze, annotation_request, annotation_requests, export_dataset, export_prompts, merge_annotations


class StrategiesTests(unittest.TestCase):
    def test_missing_baseline_dev_never_uses_test_score(self):
        task = dict(task_id='t', split='train', benchmark='APPS', difficulty='easy', problem='problem')
        run = dict(target_model='coder', comparison_group_id='g')
        candidate = dict(task_id='t', is_selected_final=False, source_records=[])
        source = {
            'tasks.jsonl': [task],
            'runs.jsonl': [{**run, 'run_id': 'r', 'optimizer_method': 'APE'},
                           {**run, 'run_id': 'b', 'optimizer_method': 'baseline'}],
            'candidates.jsonl': [{**candidate, 'candidate_id': 'c', 'run_id': 'r', 'prompt': 'new problem'},
                                 {**candidate, 'candidate_id': 'b', 'run_id': 'b', 'prompt': 'problem'}],
            'evaluations.jsonl': [dict(candidate_id='c', comparison_group_id='g', evaluation_split='dev',
                                       evaluation_id='e', score=.8),
                                  dict(candidate_id='b', comparison_group_id='g', evaluation_split='test',
                                       evaluation_id='test', score=1)],
            'filtering_report.jsonl': []}
        with patch('CodePromptOpt.strategies.pipeline.read_jsonl', side_effect=lambda p: source[Path(p).name]):
            record = export_dataset('unused')[0]
            self.assertIsNone(record['improvement'])
            self.assertIsNone(record['baseline_dev_score'])
            self.assertEqual(record['comparison_status'], 'baseline_dev_unavailable')
            source['evaluations.jsonl'][1]['score'] = 0
            self.assertEqual(export_dataset('unused')[0], record)
            source['evaluations.jsonl'][1]['evaluation_split'] = 'dev'
            self.assertEqual(export_dataset('unused')[0]['improvement'], .8)

    def test_preservation_and_abstention(self):
        problem = '\r\n  def f(x):\n    """caffè 🙂"""\n'
        self.assertEqual(build_prompt(problem, []), problem)
        self.assertTrue(build_prompt(problem, STRATEGIES).encode().endswith(problem.encode()))
        with self.assertRaises(ValueError):
            build_prompt(problem, ['invented'])

    def test_selector_rejects_invented_evidence_and_free_text(self):
        decision = {'strategies': [STRATEGIES[0]], 'evidence': {STRATEGIES[0]: ['n <= 10']}, 'confidence': .9}
        validate_selection('n <= 10', decision)
        with self.assertRaises(ValueError):
            validate_selection('no bound given', decision)
        decision['algorithm'] = 'Dijkstra'
        with self.assertRaises(ValueError):
            validate_selection('n <= 10', decision)

    def test_annotation_binding(self):
        record = {'candidate_id': 'c', 'record_hash': 'hash', 'problem': 'original', 'candidate_prompt': 'new instruction'}
        annotation = {**annotation_request(record), 'record_hash': 'old', 'response': {
            'strategies': [], 'strategy_evidence': {}, 'review_notes': ''}}
        with self.assertRaises(ValueError):
            merge_annotations([record], [annotation])
        annotation['record_hash'] = 'hash'
        self.assertEqual(merge_annotations([record], [annotation])[0]['annotation_status'], 'annotated_unreviewed')

    def test_annotation_compares_original_even_when_baseline_has_strategy(self):
        record = dict(candidate_id='c', record_hash='hash', split='train', problem='Solve the task.',
                      candidate_prompt='Solve the task. Check boundary cases.',
                      baseline_prompt='Optimizer baseline: Check boundary cases.', dev_score=.9)
        request = list(annotation_requests([record], 'train'))[0]
        payload = json.loads(request['messages'][1]['content'])
        self.assertEqual(payload, {'original_problem': record['problem'], 'candidate_prompt': record['candidate_prompt']})
        self.assertNotIn('dev_score', payload)
        response = {**request, 'response': {'strategies': [STRATEGIES[2]],
            'strategy_evidence': {STRATEGIES[2]: ['Check boundary cases.']}, 'review_notes': ''}}
        annotated = merge_annotations([record], [response])[0]
        self.assertEqual(annotated['strategies'], [STRATEGIES[2]])
        self.assertEqual(annotated['baseline_prompt'], record['baseline_prompt'])
        self.assertEqual(annotated['dev_score'], .9)
        record['baseline_prompt'] = None
        self.assertEqual(len(list(annotation_requests([record], 'train'))), 1)
        self.assertEqual(list(annotation_requests([record], 'test')), [])

    def test_legacy_annotation_reference_is_rejected(self):
        record = dict(candidate_id='c', record_hash='hash', problem='original', candidate_prompt='candidate')
        response = {**annotation_request(record), 'response': {
            'strategies': [], 'strategy_evidence': {}, 'review_notes': ''}}
        response.pop('request_hash')
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            merge_annotations([record], [response])
        response = {**annotation_request(record), 'response': response['response']}
        response['annotation_reference'] = 'baseline_prompt'
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            merge_annotations([record], [response])

    def test_task_weighted_association_and_controls(self):
        common = dict(split='train', comparable=True, quality_flags=[], benchmark='APPS', optimizer='APE',
                      target_model='coder', comparison_group_id='g', task_id='t')
        records = [{**common, 'strategies': [STRATEGIES[0]], 'dev_score': .5},
                   {**common, 'strategies': [], 'dev_score': 0},
                   {**common, 'strategies': [STRATEGIES[0], 'semantic_change'], 'dev_score': -1}]
        report = analyze(records, 'train')
        self.assertEqual(report['contrasts'][0]['mean_task_contrast'], .5)
        changed = copy.deepcopy(records)
        changed.append({**records[0], 'split': 'test', 'dev_score': -100})
        self.assertEqual(analyze(changed, 'train'), report)

    def test_export_is_discoverable_and_test_has_no_ablations(self):
        tasks = [{'task_id': 't', 'problem': '  original\r\n', 'split': 'validation'}]
        decisions = [{'id': 't', 'problem_hash': fingerprint(tasks[0]['problem']),
                      'response': {'strategies': [], 'evidence': {}, 'confidence': .9}}]
        with tempfile.TemporaryDirectory() as directory:
            from CodePromptOpt.finetuning.downstream.run import atomic_jsonl
            path = Path(directory)
            atomic_jsonl(path / 'tasks.jsonl', tasks)
            export_prompts(tasks, path / 'prompts', 'validation', decisions)
            files, audit = discover(path / 'prompts', path / 'tasks.jsonl', 'validation',
                                   variants=('original', 'all', 'dynamic', 'dynamic_abstain'))
            self.assertEqual(len(files), 4)
            self.assertEqual(audit['coverage_issues'], [])
            root = path / 'prompts' / 'sgdp-v1'
            original = read_jsonl(root / 'validation-original-prompts.jsonl')[0]['prompt']
            abstain = read_jsonl(root / 'validation-dynamic_abstain-prompts.jsonl')[0]['prompt']
            dynamic = read_jsonl(root / 'validation-dynamic-prompts.jsonl')[0]['prompt']
            self.assertEqual(original, abstain)
            self.assertNotEqual(original, dynamic)
            tasks[0]['split'] = 'test'
            export_prompts(tasks, path / 'final', 'test')
            self.assertFalse(list((path / 'final').rglob('*ablation*')))


if __name__ == '__main__':
    unittest.main()
