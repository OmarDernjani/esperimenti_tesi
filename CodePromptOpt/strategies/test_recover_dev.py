import ast
import unittest

from .recover_dev import case_harness, reconstruct


class RecoverDevTests(unittest.TestCase):
    def test_apps_prefix_is_disjoint_and_count_consistency_is_not_verification(self):
        task = {'benchmark': 'APPS', 'problem': 'question'}
        source = {'question': 'question', '_provenance': {'revision': 'pinned'},
                  'input_output': {'inputs': ['a', 'b', 'c', 'd'], 'outputs': ['1', '2', '3', '4']}}
        suite = reconstruct(task, source, 2, 1)
        self.assertEqual(suite['dev_io']['inputs'], ['a', 'b'])
        self.assertEqual(suite['test_io']['inputs'], ['c'])
        self.assertFalse(suite['historical_identity_verified'])
        self.assertEqual(suite['assumed_official_case_cap'], 3)

    def test_humaneval_mismatched_suite_or_entry_point_is_rejected(self):
        task = dict(benchmark='HumanEval+', source_id='HumanEval/0', problem='def f(): pass', entry_point='f')
        source = dict(task_id='HumanEval/0', prompt=task['problem'], entry_point='f', _provenance={},
                      test='def check(candidate):\n    inputs = [[1], [2], [3]]\n    results = [1, 2, 3]\n')
        self.assertEqual(reconstruct(task, source, 2, 1)['dev_io']['outputs'], [[1], [2]])
        with self.assertRaises(ValueError):
            reconstruct(task, source, 1, 1)
        source['entry_point'] = 'other'
        with self.assertRaises(ValueError):
            reconstruct(task, source, 2, 1)

    def test_harness_preserves_nested_tuples_without_executing_code(self):
        data = {'fn_name': 'f', 'inputs': [[(1, 2)]], 'outputs': [[3]]}
        tree = ast.parse(case_harness(data))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'DATA' for t in n.targets))
        recovered = ast.literal_eval(ast.literal_eval(assignment.value.args[0]))
        self.assertEqual(recovered, data)
        self.assertIsInstance(recovered['inputs'][0][0], tuple)


if __name__ == '__main__':
    unittest.main()
