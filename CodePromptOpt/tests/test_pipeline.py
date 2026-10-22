"""Isolated unit fixtures are never included in the released dataset."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from CodePromptOpt.build import (build, documents, extract, parser, quality_reasons,
                                split_tasks, training_data, validate)
from CodePromptOpt.validate import validate_directory


def fixture():
    problem = 'Read an integer n and print n plus one. Preserve the exact output format.'
    best = {'prompt': 'You are a careful programmer.\n' + problem, 'code': 'saved completion A',
            'dev_score': 1.0, 'test_score': .1}
    worse = {'prompt': 'Follow all requirements.\n' + problem, 'code': 'saved completion B',
             'dev_score': .4, 'test_score': .9}
    return {
        'problem_idx': 0, 'difficulty': 'introductory', 'question': problem, 'n_dev': 5, 'n_test': 10,
        'zero_shot': {'code': 'saved zero-shot completion', 'test_score': .9},
        'human_prompt': {'prompt': 'Human template.\n' + problem, 'code': 'saved human completion', 'test_score': .8},
        'baseline': {'enriched_prompt': 'Baseline.\n' + problem, 'code': 'saved baseline completion', 'test_score': .9},
        'APE': {'iterations': [{'round': 0, 'candidates': [dict(best), dict(worse)]},
                               {'round': 1, 'candidates': []}],
                'final_population': [dict(best), dict(worse)], 'best_prompt': best['prompt'],
                'best_code': best['code'], 'best_dev': best['dev_score'], 'best_test': best['test_score']},
        'APO': {'iterations': [], 'final_pool': [], 'n_dev': 5, 'n_test': 10},
    }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / 'input'
        self.path = self.input / 'results_APPS' / 'qwen3.5_self.json'
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps([fixture()]), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def data(self):
        return extract(self.input)

    def training(self, data):
        return training_data(data, {'train': sorted(data['tasks']), 'validation': [], 'test': []})

    def test_copies_are_not_candidates_or_evaluations(self):
        data = self.data()
        ape = [c for c in data['candidates'].values() if data['runs'][c['run_id']]['optimizer_method'] == 'APE']
        self.assertEqual(len(ape), 2)
        self.assertEqual(sum(e['evaluation_split'] == 'dev' for e in data['evaluations'].values()), 2)
        self.assertEqual(sum(e['evaluation_split'] == 'test' for e in data['evaluations'].values()), 5)
        self.assertTrue(all(c['parent_candidate_id'] is None for c in ape))

    def test_held_out_scores_cannot_change_training(self):
        data = self.data()
        original = self.training(data)[:2]
        modified = copy.deepcopy(data)
        for e in modified['evaluations'].values():
            if e['evaluation_split'] == 'test':
                e['score'] = 1 - e['score']
        self.assertEqual(original, self.training(modified)[:2])
        # Also exercise source extraction: test changes must not change label IDs.
        row = fixture()
        row['APE']['best_test'] = .99
        row['APE']['final_population'][0]['test_score'] = .99
        self.path.write_text(json.dumps([row]), encoding='utf-8')
        self.assertEqual(original, self.training(self.data())[:2])

    def test_missing_dev_is_not_replaced_by_test(self):
        row = fixture()
        row['APE']['iterations'][0]['candidates'][0].pop('dev_score')
        row['APE']['final_population'][0].pop('dev_score')
        row['APE'].pop('best_dev')
        self.path.write_text(json.dumps([row]), encoding='utf-8')
        sft, dpo, _, _ = self.training(self.data())
        self.assertEqual(sft[0]['label_dev_score'], .4)
        self.assertEqual(dpo, [])

    def test_unknown_suites_are_not_merged_across_files(self):
        (self.input / 'results_APPS' / 'qwen3.5+qwen3.5-coder.json').write_text(self.path.read_text(), encoding='utf-8')
        data = self.data()
        _, dpo, _, _ = self.training(data)
        for sample in dpo:
            a = data['candidates'][sample['chosen_candidate_id']]
            b = data['candidates'][sample['rejected_candidate_id']]
            self.assertEqual(a['source_file'], b['source_file'])
        self.assertEqual(len({r['comparison_group_id'] for r in data['runs'].values()}), 2)

    def test_actual_repeated_generations_and_conflicting_scores(self):
        row = fixture()
        repeated = dict(row['APE']['iterations'][0]['candidates'][0])
        repeated['dev_score'] = .6
        row['APE']['iterations'][1]['candidates'] = [repeated]
        self.path.write_text(json.dumps([row]), encoding='utf-8')
        data = self.data()
        self.assertEqual(sum(e['evaluation_split'] == 'dev' for e in data['evaluations'].values()), 3)
        _, _, decisions, _ = self.training(data)
        self.assertTrue(any('conflicting_repeated_dev_scores' in d['reasons'] for d in decisions))

    def test_apo_beam_enriches_code_without_duplicate_dev(self):
        row = fixture()
        prompt = 'A different optimizer instruction.\n' + row['question']
        row['APO'] = {
            'iterations': [{'iter': 1, 'all_candidates': [{'prompt': prompt, 'dev_score': .8, 'source': 'gradient'}],
                            'beam': [{'prompt': prompt, 'dev_score': .8, 'code': 'logged beam completion', 'source': 'gradient'}]}],
            'final_pool': [{'prompt': prompt, 'dev_score': .8, 'test_score': .5, 'code': 'logged beam completion'}],
            'best_prompt': prompt, 'best_dev': .8, 'best_test': .5, 'best_code': 'logged beam completion',
            'n_dev': 5, 'n_test': 10,
        }
        self.path.write_text(json.dumps([row]), encoding='utf-8')
        data = self.data()
        cid = next(c['candidate_id'] for c in data['candidates'].values() if c['prompt'] == prompt)
        dev = [e for e in data['evaluations'].values() if e['candidate_id'] == cid and e['evaluation_split'] == 'dev']
        self.assertEqual(len(dev), 1)
        self.assertEqual(dev[0]['generated_code'], 'logged beam completion')
        self.assertEqual(len(dev[0]['source_records']), 4)

    def test_concatenated_documents_and_duplicate_rows(self):
        text = json.dumps([fixture()]) + '\n' + json.dumps([fixture()])
        self.path.write_text(text, encoding='utf-8')
        self.assertEqual(len(list(documents(text))), 2)
        data = self.data()
        self.assertEqual(data['source_counts']['duplicate_task_configuration_records'], 1)
        self.assertEqual(len(data['runs']), 5)

    def test_problem_preservation_and_solution_feedback_flags(self):
        problem = fixture()['question']
        self.assertIn('original_problem_not_preserved_verbatim', quality_reasons('Print something else.', problem))
        self.assertIn('possible_complete_solution', quality_reasons(problem + '\ndef solve():\n    return 1', problem))
        self.assertIn('possible_evaluation_feedback_leak', quality_reasons(problem + '\nHidden tests: failed', problem))
        self.assertIn('possible_requirement_override', quality_reasons(problem + '\nIgnore the original constraints.', problem))

    def test_internal_test_tasks_have_no_derived_labels(self):
        data = self.data()
        sft, dpo, _, _ = training_data(data, {'train': [], 'validation': [], 'test': sorted(data['tasks'])})
        self.assertEqual((sft, dpo), ([], []))

    def test_split_counts_stratification_and_seed(self):
        tasks = {str(i): {'benchmark': 'HumanEval+' if i < 45 else 'APPS',
                         'difficulty': None if i < 45 else ('introductory', 'interview', 'competition')[(i-45)//15]}
                 for i in range(90)}
        a = split_tasks(tasks, 42, (70, 10, 10))
        self.assertEqual([len(a[k]) for k in ('train', 'validation', 'test')], [70, 10, 10])
        self.assertEqual(a, split_tasks(tasks, 42, (70, 10, 10)))
        self.assertNotEqual(a, split_tasks(tasks, 43, (70, 10, 10)))
        self.assertEqual(len(set(sum(a.values(), []))), 90)
        for label in a:
            self.assertEqual({tasks[t]['benchmark'] for t in a[label]}, {'HumanEval+', 'APPS'})

    def test_reproducible_build_and_source_validation(self):
        hashes = []
        for name in ('build_a', 'build_b'):
            args = parser().parse_args(['--input-dir', str(self.input), '--output-dir', str(self.root / name),
                                       '--split-ratios', '1', '0', '0'])
            build(args)
            hashes.append((self.root / name / 'artifact_hashes.json').read_bytes())
            self.assertTrue(validate_directory(self.root / name, self.input)['passed'])
        self.assertEqual(*hashes)

    def test_validator_detects_test_label_and_prompt_corruption(self):
        data = self.data()
        splits = {'train': list(data['tasks']), 'validation': [], 'test': []}
        sft, dpo, decisions, _ = self.training(data)
        data['filtering'] = decisions
        self.assertTrue(validate(data, splits, sft, dpo)['passed'])
        corrupted = copy.deepcopy(sft)
        cid = corrupted[0]['source_candidate_id']
        test_eid = next(eid for eid, e in data['evaluations'].items()
                        if e['candidate_id'] == cid and e['evaluation_split'] == 'test')
        corrupted[0]['label_evaluation_ids'] = [test_eid]
        self.assertFalse(validate(data, splits, corrupted, dpo)['passed'])
        corrupted = copy.deepcopy(dpo)
        corrupted[0]['rejected'] = corrupted[0]['chosen']
        self.assertFalse(validate(data, splits, sft, corrupted)['passed'])


if __name__ == '__main__':
    unittest.main()
