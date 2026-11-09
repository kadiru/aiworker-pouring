"""Synthetic CPU regressions; these are not native task results."""
import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/src'))
from pour_side_metrics import side_metrics

class RecordedReturnChecks(unittest.TestCase):
    def fixture(self):
        s=[];d=[];cup=[.5,-.12,1.04];target=[.5,-.30]
        for i in range(41):
            u=i/40;s.append(dict(t=i*.05,bottle_pose_xyz=[.5,-.12-.18*u,1.12],bottle_tilt_deg=80*(1-u),cup_fill_fraction=.94))
            d.append(dict(mode='Returning upright',stop_latched='cup_near_full',side_return_plan=dict(side='right',placement_xy=target)))
        return s,d,dict(cup_world=cup)
    def test_simultaneous_withdrawal(self):
        self.assertTrue(side_metrics(*self.fixture())['passed'])
    def test_old_centered_upright_then_translate_is_rejected(self):
        s,d,spec=self.fixture()
        for i,x in enumerate(s):
            x['bottle_tilt_deg']=max(0,80*(1-i/20))
            x['bottle_pose_xyz'][1]=-.12-.18*max(0,(i-20)/20)
        r=side_metrics(s,d,spec);self.assertFalse(r['passed']);self.assertGreater(r['upright_over_cup_samples'],0)
    def test_wrong_side_is_rejected(self):
        s,d,spec=self.fixture()
        for x in s:x['bottle_pose_xyz'][1]=-.24-x['bottle_pose_xyz'][1]
        self.assertFalse(side_metrics(s,d,spec)['passed'])
    def test_absent_plan_is_not_success(self):
        s,d,spec=self.fixture()
        for x in d:x.pop('side_return_plan')
        self.assertFalse(side_metrics(s,d,spec)['passed'])

class PlacementBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:from pour_side_return import placement_candidates
        except ImportError as e:raise unittest.SkipTest('Native scipy dependency: '+str(e))
        cls.candidates=staticmethod(placement_candidates)
    def test_prefer_robot_right(self):
        c=next(self.candidates([.5,-.12]));self.assertEqual(c['side'],'right');self.assertAlmostEqual(c['xy'][1],-.30)
    def test_edge_uses_left(self):
        c=next(self.candidates([.5,-.32]));self.assertEqual(c['side'],'left');self.assertGreaterEqual(c['offset_m'],.1524)
    def test_entire_footprint_on_table(self):
        for y in np.linspace(-.395,-.005,20):
            for c in self.candidates([.5,y]):
                self.assertGreaterEqual(c['xy'][1],-.395);self.assertLessEqual(c['xy'][1],-.005+1e-10)

if __name__=='__main__':unittest.main()
