import unittest
import numpy as np
from pour_scale_fill import SeedFill


def specification():
    return dict(calibration_only=True, policy_evaluation=False, spacing=.0018,
                cup_world=[.45, -.2, 1.04], cup_height_scale=1.1,
                calibration_fill=dict(flow_nominal_ml_s=8., target_height_fraction=.9,
                                      start_s=.5, max_feed_s=12., settle_s=3.))


class SeedFillTests(unittest.TestCase):
    def test_autonomous_or_held_scene_cannot_emit(self):
        for change in (dict(calibration_only=False), dict(policy_evaluation=True),
                       dict(calibration_hold_angle_deg=80)):
            with self.assertRaises(ValueError): SeedFill(dict(specification(), **change), 5000)

    def test_resolution_and_nonfinite_parameters_rejected(self):
        for spacing in (.0012, float('nan')):
            with self.assertRaises(ValueError): SeedFill(dict(specification(), spacing=spacing), 5000)
        s=specification();s['calibration_fill']['flow_nominal_ml_s']=float('inf')
        with self.assertRaises(ValueError): SeedFill(s, 5000)

    def test_finite_budget_and_mass_flow_at_native_substeps(self):
        f=SeedFill(specification(), 5000);count=0
        for t in np.arange(0., 16., .0025):
            p,v=f.additions(float(t), None);count+=len(p)
            self.assertEqual(p.shape, v.shape)
            if len(p):
                self.assertTrue((p[:,2]>1.04+.0414).all())
                self.assertTrue((v[:,2]<0).all())
        self.assertEqual(count, f.emitted_count)
        self.assertAlmostEqual(f.evidence()['emitted_nominal_ml'], 96., delta=.13)
        self.assertEqual(f.stop_reason, 'finite_feed_limit')
        self.assertTrue(f.complete(16.))

    def test_one_camera_sample_cannot_count_twice_or_restart_after_stop(self):
        f=SeedFill(specification(),5000)
        for t in np.arange(0., .51, .0025):f.additions(float(t),None)
        obs=dict(t=.5,cup_fill_fraction=.95)
        f.additions(.51,obs);f.additions(.5125,obs)
        self.assertIsNone(f.stop_time)
        f.additions(.515,dict(t=.515,cup_fill_fraction=.95))
        count=f.emitted_count
        p,_=f.additions(5.,dict(t=5.,cup_fill_fraction=.2))
        self.assertEqual(len(p),0);self.assertEqual(f.emitted_count,count)
        self.assertEqual(f.stop_reason,'measured_target')

    def test_clock_jump_cannot_inject_a_dense_burst(self):
        f=SeedFill(specification(),5000)
        with self.assertRaises(ValueError):f.additions(1.,None)


if __name__=='__main__':unittest.main()
