"""Meaningful coverage, separation and unchanged evaluation-contract checks."""
import json
from pathlib import Path
import unittest
from pour_workspace_study import make_plan,RANGES


class PlanTests(unittest.TestCase):
    def test_expanded_disjoint_balanced_plan(self):
        root=Path(__file__).resolve().parents[2]
        parent=json.loads((root/'evidence/reliable-A/frozen-plan.json').read_text())
        p=make_plan(parent); all_s=p['demonstrations']+p['development']+p['test']
        signature=lambda s:tuple(s['cup_world']+s['source_xy']+[s['initial_ml'],s['cup_initial_ml']])
        self.assertEqual(len(all_s),58)
        self.assertEqual(len(set(signature(s) for s in all_s)),58)
        self.assertFalse({signature(s) for s in all_s}&{signature(s) for s in parent['test']})
        self.assertEqual(sum(s['expected_stop']=='cup_near_full' for s in p['test']),15)
        for s in p['test']+p['development']:
            d=[s['cup_world'][0]-.49,s['cup_world'][1]+.15,s['source_xy'][0]+.12,s['source_xy'][1]]
            self.assertTrue(all(abs(x)<=r+1e-9 for x,r in zip(d,RANGES)))
            self.assertGreaterEqual(max(abs(x)/(r/3) for x,r in zip(d,RANGES)),1.25)
            self.assertTrue(s['policy_evaluation']);self.assertFalse(s['expert_recovery'])
            for k in ('spacing','dt','mouth_z','goal_definition','max_seconds','grasp_deadline'):
                self.assertEqual(s[k],parent['test'][0][k])
        self.assertEqual(p,make_plan(parent))


if __name__=='__main__':unittest.main()
