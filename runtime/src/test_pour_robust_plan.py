"""CPU plan/provenance checks; no simulator or GPU imports."""
import copy,json,unittest
from pathlib import Path
from pour_robust_study import make_plan


class PlanTests(unittest.TestCase):
    def test_actual_saved_parent_material_is_explicit(self):
        p=Path(__file__).resolve().parents[2]/'evidence/guard-B/frozen-plan.json'
        base=json.loads(p.read_text())['test'][1]
        self.assertTrue(all(x['vessel_appearance']=='A' for x in make_plan(base)['test']))

    def base(self):
        return dict(spacing=.0018,dt=.0025,vessel_appearance='A',policy_evaluation=True,expert_recovery=False,max_seconds=40.,
            policy_family='visual_guard_hierarchy',threshold_marker={'unchanged':True},isosurface_max_subgrids=32768)

    def test_reserved_plan_is_balanced_unique_and_deterministic(self):
        a=make_plan(self.base());self.assertEqual(a,make_plan(self.base()));rows=a['test']
        self.assertEqual(len(rows),30);self.assertEqual(len({x['name'] for x in rows}),30)
        self.assertEqual(sum(x['expected_stop']=='bottle_empty' for x in rows),15)

    def test_randomization_stays_in_validated_domain(self):
        for x in make_plan(self.base())['test']:
            self.assertLessEqual(abs(x['cup_world'][0]-.49),.008);self.assertLessEqual(abs(x['cup_world'][1]+.15),.008)
            self.assertLessEqual(abs(x['source_xy'][0]+.12),.009);self.assertLessEqual(abs(x['source_xy'][1]),.004)
            if x['expected_stop']=='bottle_empty':self.assertLess(x['initial_ml']+x['cup_initial_ml'],70)
            else:self.assertGreaterEqual(x['initial_ml']+x['cup_initial_ml'],70)

    def test_generator_preserves_parent_controller_and_physics(self):
        base=self.base();before=copy.deepcopy(base);rows=make_plan(base)['test'];self.assertEqual(base,before)
        for x in rows:
            for name,value in before.items():self.assertEqual(x[name],value,name)
        rows[0]['threshold_marker']['unchanged']=False
        self.assertTrue(base['threshold_marker']['unchanged']);self.assertTrue(rows[1]['threshold_marker']['unchanged'])


if __name__=='__main__':unittest.main()
