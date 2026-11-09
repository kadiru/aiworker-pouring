import copy
import json
import unittest
from pathlib import Path
from scale_pair_plan import extend

ROOT = Path(__file__).resolve().parents[1] / 'evidence/scale-A'


class PairPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / 'plan-collection-r0.json').read_text())
        self.proposal = json.loads((ROOT / 'fill-pair-supplement-proposal.json').read_text())

    def test_adds_only_training_and_keeps_reserved_cases_and_gates(self):
        before = copy.deepcopy(self.plan)
        result = extend(self.plan, self.proposal)
        self.assertEqual(self.plan, before)
        self.assertEqual(len(result['collect']), 108)
        self.assertEqual(result['holdout'], before['holdout'])
        self.assertEqual(result['component_gates'], before['component_gates'])
        self.assertEqual(result['collect'][:54], before['collect'])

    def test_rejects_reserved_id_reuse(self):
        self.proposal['cases'][0]['name'] = self.plan['holdout'][0]['name']
        with self.assertRaises(AssertionError): extend(self.plan, self.proposal)

    def test_rejects_changed_physics_and_split(self):
        for key, value in (('cohesion', .1), ('study_split', 'validation')):
            proposal = copy.deepcopy(self.proposal)
            proposal['cases'][0][key] = value
            with self.assertRaises(AssertionError): extend(self.plan, proposal)

    def test_rejects_duplicate_fill_instead_of_missing_pair(self):
        case = self.proposal['cases'][0]
        source = next(s for s in self.plan['collect'] if s['name'] == case['proposed_pair_of'])
        case['calibration_initial_height'] = source['calibration_initial_height']
        with self.assertRaises(AssertionError): extend(self.plan, self.proposal)


if __name__ == '__main__': unittest.main()
