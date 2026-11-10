import copy,unittest
import numpy as np
from pour_dynamic_motion import motion_pose
from pour_scale_fill import SeedFill
from bottle_pour import initial_points
def spec():return dict(calibration_only=True,policy_evaluation=False,calibration_hold_angle_deg=0.,source_xy=[.15,0.],mouth_z=.056,calibration_motion=dict(maximum_angle_deg=82.,lateral_amplitude_m=.02,start_s=.5,duration_s=3.),max_seconds=4.)
class DynamicTests(unittest.TestCase):
    def test_bounds_continuity_and_motion(self):
        s=spec();values=[motion_pose(s,t) for t in np.linspace(0,4,801)];p=np.array([v[0] for v in values]);a=np.array([v[2] for v in values])
        self.assertGreater(np.ptp(a),81.9);self.assertLessEqual(a.max(),82.00001)
        self.assertGreater(np.max(np.linalg.norm(p[:,None]-p[None,:],axis=2)),.02)
        self.assertLess(np.max(np.linalg.norm(np.diff(p,axis=0),axis=1)),.004)
        np.testing.assert_allclose(p[0],p[-1],atol=1e-10)
    def test_fixture_cannot_run_as_policy_or_emit(self):
        for change in (dict(policy_evaluation=True),dict(calibration_only=False),dict(calibration_fill={'x':1})):
            with self.assertRaises(ValueError):motion_pose(dict(spec(),**change),1.)
    def test_empty_bottle_only_allowed_for_authored_motion(self):
        s=dict(spec(),spacing=.0018,initial_ml=0.);self.assertEqual(initial_points(s).shape,(0,3))
        with self.assertRaises(ValueError):initial_points(dict(s,policy_evaluation=True))
    def test_high_fill_protocol_keeps_original_bounds(self):
        s=dict(calibration_only=True,policy_evaluation=False,spacing=.0018,cup_world=[.5,-.2,1.04],calibration_fill=dict(flow_nominal_ml_s=.75,target_height_fraction=.956,start_s=.5,max_feed_s=6.,settle_s=3.))
        with self.assertRaises(ValueError):SeedFill(s,10000)
        SeedFill(dict(s,calibration_protocol='size_dynamic_v1'),10000)
        s=copy.deepcopy(s);s['calibration_fill']['target_height_fraction']=.99
        with self.assertRaises(ValueError):SeedFill(dict(s,calibration_protocol='size_dynamic_v1'),10000)
if __name__=='__main__':unittest.main()
