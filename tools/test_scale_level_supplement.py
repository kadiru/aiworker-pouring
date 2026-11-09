"""Data-plan preservation checks, using the actual frozen development plan."""
import copy,json,unittest
from pathlib import Path
from scale_level_supplement import extend
class SupplementChecks(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((Path(__file__).resolve().parents[1]/'evidence/scale-A/plan-collection-r1.json').read_text())
        self.inventory={s['name']:dict(particles=100) for s in self.plan['collect'] if s['study_split']=='train' and s['name'].startswith('SC_A_static_train_')}
    def test_preserves_all_original_cases_and_gates(self):
        old=copy.deepcopy(self.plan);p=extend(self.plan,self.inventory)
        self.assertEqual(self.plan,old);self.assertEqual(p['collect'][:108],old['collect']);self.assertEqual(p['holdout'],old['holdout']);self.assertEqual(p['component_gates'],old['component_gates'])
        self.assertEqual(len(p['collect']),144);self.assertEqual(len(p['collection_rounds']['r2']),36)
    def test_only_training_liquid_initial_condition_changes(self):
        p=extend(self.plan,self.inventory);old={s['name']:s for s in self.plan['collect']}
        for s in p['collect'][108:]:
            source=old[s['level_supplement_of']];self.assertEqual(s['study_split'],'train');self.assertEqual(s['calibration_initial_height'],.35)
            for k,v in source.items():
                if k not in ('name','calibration_initial_height','cup_initial_ml'):self.assertEqual(s[k],v)
if __name__=='__main__':unittest.main()
