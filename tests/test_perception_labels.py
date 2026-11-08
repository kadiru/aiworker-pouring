import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/src'))
from pour_perception_geometry import settled_label,calibration_seed_points,upper_surface
from pour_perception_study import check_freeze,check_vessel_scope

class Labels(unittest.TestCase):
    def test_baseline_validation_does_not_authorize_resized_autonomy(self):
        manifest={'validated_cup_scope':'baseline_only'}
        check_vessel_scope({},manifest)
        for key in ('cup_height_scale','cup_diameter_scale','bottle_height_scale','bottle_diameter_scale','bottle_neck_scale'):
            for size in (.9,1.1):
                with self.assertRaises(AssertionError):check_vessel_scope({key:size},manifest)
    def test_frozen_evaluation_rejects_changed_identity(self):
        import copy
        selection=dict(checkpoint_sha256='weights',plan_sha256='plan',runtime_sources={'actor.py':'source'})
        check_freeze(selection,copy.deepcopy(selection))
        for key,value in [('checkpoint_sha256','other'),('plan_sha256','other'),('runtime_sources',{'actor.py':'other'})]:
            changed=copy.deepcopy(selection);changed[key]=value
            with self.assertRaises(AssertionError):check_freeze(selection,changed)
    def rows(self):
        return [dict(t=i*.05,cup_fill_fraction=.93,cup_ml=60.,ever_table_ml=0.,spilled_ml=0.,airborne_ml=0.) for i in range(61)]
    def test_endpoint_dip_retains_raw_and_robust_training_target(self):
        rows=self.rows();rows[-1]['cup_fill_fraction']=.905
        result=settled_label(rows)
        self.assertTrue(result['accepted']);self.assertAlmostEqual(result['target'],.93)
        self.assertEqual(rows[-1]['cup_fill_fraction'],.905)
        self.assertEqual(result['final_sample'],.905)
    def test_reject_loss_inventory_change_and_drift(self):
        for key,value in [('ever_table_ml',.005832),('airborne_ml',.005832),('cup_ml',59.)]:
            rows=self.rows();rows[-1][key]=value
            self.assertFalse(settled_label(rows)['accepted'],key)
        rows=self.rows()
        for i,r in enumerate(rows):r['cup_fill_fraction']+=i*.001
        self.assertFalse(settled_label(rows)['accepted'])
    def test_reject_invalid_and_short_window(self):
        self.assertFalse(settled_label(self.rows()[:8])['accepted'])
        rows=self.rows();rows[-1]['cup_fill_fraction']=float('nan')
        with self.assertRaises(ValueError):settled_label(rows)
        rows=self.rows();rows[-1]['t']=rows[-2]['t']
        with self.assertRaises(ValueError):settled_label(rows)
    def test_geometry_subsets_without_scaling_and_requires_calibration(self):
        points=np.array([[x,y,z] for x in np.arange(-.02,.021,.003) for y in np.arange(-.02,.021,.003) for z in np.arange(.004,.038,.003)],dtype=np.float32)
        spec=dict(calibration_only=True,calibration_initial_height=.7,spacing=.0018,cup_height_scale=.9,cup_diameter_scale=.9)
        result=calibration_seed_points(spec,points)
        self.assertTrue({tuple(p) for p in result}.issubset({tuple(p) for p in points}))
        self.assertLess(result[:,2].max(),.004+.7*.034*.9)
        with self.assertRaises(AssertionError):calibration_seed_points(dict(spec,policy_evaluation=True),points)
    def test_surface_ignores_bottom_population_and_isolated_droplet(self):
        xy=np.array([[x,y] for x in np.arange(-.012,.013,.0015) for y in np.arange(-.012,.013,.0015)])
        p=np.array([[*v,z] for v in xy for z in (.005,.01,.02,.03)])
        profile=((.004,.02),(.038,.024))
        value=upper_surface(p,profile,.0018)
        crowded=np.vstack([p,np.array([[*v,.008] for v in xy for _ in range(20)]),[[0.,0.,.037]]])
        other=upper_surface(crowded,profile,.0018)
        self.assertTrue(value['valid']);self.assertEqual(value['height_fraction'],other['height_fraction'])
        self.assertAlmostEqual(value['height_m'],.0309)

if __name__=='__main__':unittest.main()
