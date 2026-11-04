"""Meaningful coverage, separation and unchanged evaluation-contract checks."""
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pour_workspace_study as study
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

    def test_repairs_retain_failed_names_and_cannot_change_scene_or_test_plan(self):
        root=Path(__file__).resolve().parents[2]
        p=make_plan(json.loads((root/'evidence/reliable-A/frozen-plan.json').read_text()))
        base=p['demonstrations'][8]
        repair=dict(base,name=base['name']+'_repair_r1',reference_case=base['name'],
            reference_repair_round=1,approach_clearance_m=.04)
        with TemporaryDirectory() as temporary,patch.object(study,'ROOT',Path(temporary)):
            frozen=study.ROOT/'frozen-plan.json';frozen.write_text(json.dumps(p));before=frozen.read_bytes()
            path=study.ROOT/'reference-repair-plan.json'
            path.write_text(json.dumps(dict(demonstrations=[repair])))
            self.assertEqual(len(study.reference_specs()),21)
            self.assertEqual(study.reference_specs()[8],base)
            self.assertEqual(frozen.read_bytes(),before)
            bad=dict(repair,cup_world=[.50,-.15,1.04])
            path.write_text(json.dumps(dict(demonstrations=[bad])))
            with self.assertRaises(AssertionError):study.reference_specs()
            path.write_text(json.dumps(dict(demonstrations=[repair,repair])))
            with self.assertRaises(AssertionError):study.reference_specs()


if __name__=='__main__':unittest.main()
