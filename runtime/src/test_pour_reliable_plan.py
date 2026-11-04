import json
from pathlib import Path
import unittest
from pour_reliable_study import make_plan


class ReliablePlanTests(unittest.TestCase):
    def test_reserved_batch_is_new_balanced_and_physically_unchanged(self):
        root=Path(__file__).resolve().parents[2]
        prior=json.loads((root/'evidence/settled-A/frozen-plan.json').read_text())
        earlier=json.loads((root/'evidence/robust-A/frozen-plan.json').read_text())
        new=make_plan(prior)
        self.assertEqual(new,make_plan(prior));self.assertEqual(len(new['test']),30)
        self.assertEqual(sum(x['expected_stop']=='bottle_empty' for x in new['test']),15)
        key=lambda x:(tuple(x['cup_world']),tuple(x['source_xy']),x['initial_ml'],x['cup_initial_ml'])
        self.assertFalse({key(x) for x in new['test']} & {key(x) for x in prior['test']+earlier['test']})
        self.assertEqual({x['development_case'] for x in new['development']},{19,26,29,1,14,0,10})
        for x in new['test']:
            for field in ('spacing','dt','viscosity','cohesion','surface_tension','geometry' ,'max_seconds','grasp_deadline','goal_definition','vessel_appearance'):
                self.assertEqual(x.get(field),prior['test'][0].get(field))
            self.assertFalse(x['expert_recovery']);self.assertTrue(x['record_cameras'])


if __name__=='__main__':unittest.main()
