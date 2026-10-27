import unittest

from .annotation_study import agreement, compare, pilot
from .pipeline import annotation_request
from .core import STRATEGIES


def record(cid, split='train'):
    return dict(candidate_id=cid, task_id=cid, split=split, benchmark='APPS', optimizer='APE',
                difficulty='easy', quality_flags=[], dev_score=0, problem='Problem',
                candidate_prompt='Problem. Check boundary cases.', record_hash=cid)


def response(r, labels):
    return {**annotation_request(r), 'response': {'strategies': labels,
        'strategy_evidence': {s: ['Check boundary cases.'] for s in labels}, 'review_notes': ''}}


class AnnotationStudyTests(unittest.TestCase):
    def test_pilot_is_reproducible_and_train_only(self):
        records = [record(str(i)) for i in range(20)] + [record('heldout', 'test')]
        a = pilot(records, 10)
        self.assertEqual(a, pilot(list(reversed(records)), 10))
        self.assertEqual(len({r['candidate_id'] for r in a}), 10)
        self.assertTrue(all(r['split'] == 'train' for r in a))
        with self.assertRaises(ValueError):
            pilot(records, 30)

    def test_kappa_and_constant_labels(self):
        self.assertEqual(agreement([True, False], [True, False])['kappa'], 1)
        self.assertEqual(agreement([True, False], [False, True])['kappa'], -1)
        self.assertIsNone(agreement([False], [False])['kappa'])
        self.assertIsNone(agreement([], [])['agreement'])

    def test_missing_and_invalid_remain_in_coverage(self):
        records = [record('a'), record('b'), record('c')]
        left = [response(r, [STRATEGIES[2]]) for r in records]
        right = [response(records[0], []), response(records[1], [STRATEGIES[2]])]
        right[1]['response']['strategy_evidence'][STRATEGIES[2]] = ['invented quote']
        report, review = compare(records, left, right)
        self.assertEqual(report['expected'], 3)
        self.assertEqual(report['paired_valid'], 1)
        self.assertEqual(len(report['annotator_b']['invalid']), 1)
        self.assertEqual(report['annotator_b']['missing_ids'], ['c'])
        self.assertEqual(len(review), 3)
        self.assertEqual(report['exact_label_set_agreement'], 0)


if __name__ == '__main__':
    unittest.main()
