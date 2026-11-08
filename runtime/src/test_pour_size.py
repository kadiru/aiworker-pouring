import unittest
import numpy as np
from bottle_pour import CUP, CAPACITY_ML, vessel_mask
from pour_vessel_geometry import cup_profile, cup_geometry, initial_cup_points, capacity_ml,settled_cup_points
from pour_source_watchdog import SourceWatchdog, DEFAULT


class SizeTests(unittest.TestCase):
    def test_default_profile_and_capacity_are_compatible(self):
        self.assertEqual(cup_profile({}), CUP)
        self.assertAlmostEqual(capacity_ml(CUP), CAPACITY_ML, places=12)

    def test_grid_containment_and_independent_capacity_scaling(self):
        for h in (.9,1.,1.1):
            for d in (.9,1.,1.1):
                spec=dict(cup_height_scale=h,cup_diameter_scale=d,spacing=.0018,cup_initial_ml=16*h*d*d)
                p=initial_cup_points(spec);g=cup_geometry(spec)
                self.assertTrue(vessel_mask(p,cup_profile(spec)).all())
                self.assertAlmostEqual(g['capacity_geometric_ml']/CAPACITY_ML,h*d*d)
                self.assertAlmostEqual(g['bottom_z_m'],.004)
                self.assertAlmostEqual(g['rim_z_m'],.004+.034*h)
                self.assertLessEqual(abs(len(p)*.0018**3*1e6-spec['cup_initial_ml']),.0018**3*1e6/2)

    def test_rejects_unsafe_or_unsupported_geometry_and_seed(self):
        for key,value in [('cup_height_scale',float('nan')),('cup_diameter_scale',1.2),('bottle_neck_scale',.9)]:
            with self.assertRaises(ValueError):cup_profile({key:value})
        with self.assertRaises(ValueError):initial_cup_points(dict(spacing=.0018,cup_initial_ml=100))

    def test_settled_seed_preserves_particle_positions_and_inventory(self):
        seed=initial_cup_points(dict(spacing=.0018,cup_initial_ml=20.))
        spec=dict(spacing=.0018,cup_initial_ml=4.,cup_height_scale=1.1,cup_diameter_scale=.9)
        points=settled_cup_points(spec,seed)
        self.assertEqual(len(points),round(4./.0018**3/1e6))
        self.assertTrue(vessel_mask(points,cup_profile(spec)).all())
        originals={tuple(v) for v in seed}
        self.assertTrue(all(tuple(v) in originals for v in points))
        with self.assertRaises(ValueError):settled_cup_points(dict(spec,cup_initial_ml=40.),seed)


class WatchdogTests(unittest.TestCase):
    def run_trace(self, **kwargs):
        w=SourceWatchdog(DEFAULT);base=dict(held=True,present=.8,empty=.01,tilt=100.,phase=100.,cap=131.,flow_ml_s=2.)
        base.update(kwargs);reasons=[w.update(i*.05,**base) for i in range(51)]
        return reasons,w

    def test_ambiguous_empty_recovers_but_never_claims_success(self):
        reasons,_=self.run_trace(present=.62,empty=.99,tilt=131.,phase=131.,flow_ml_s=0.)
        self.assertTrue(all(r is None for r in reasons[:24]))
        self.assertEqual(reasons[24],'source_state_unconfirmed')
        self.assertNotIn('bottle_empty',reasons)

    def test_early_visual_false_negative_keeps_verified_encoder_grasp(self):
        reasons,_=self.run_trace(present=.02,tilt=10.,phase=10.)
        self.assertTrue(all(r is None for r in reasons))

    def test_encoder_loss_still_aborts_early(self):
        reasons,_=self.run_trace(held=False,present=.02,tilt=10.,phase=10.)
        self.assertEqual(reasons[7],'source_tracking_lost')

    def test_loss_and_nonfinite_input_abort(self):
        for kwargs in (dict(present=.02),dict(empty=float('nan'))):
            reasons,_=self.run_trace(**kwargs)
            self.assertEqual(reasons[7],'source_tracking_lost')

    def test_evidence_must_be_continuous(self):
        w=SourceWatchdog(DEFAULT);v=dict(held=True,present=.6,empty=.99,tilt=110.,phase=110.,cap=131.,flow_ml_s=0.)
        for i in range(101):
            p=.8 if i%20==0 else .6
            self.assertIsNone(w.update(i*.05,**dict(v,present=p)))

    def test_cap_and_global_progress_are_bounded(self):
        for phase,limit in ((131.,8.),(100.,48.)):
            w=SourceWatchdog(DEFAULT)
            v=dict(held=True,present=.8,empty=.1,tilt=phase,phase=phase,cap=131.,flow_ml_s=0.)
            self.assertIsNone(w.update(0.,**v))
            self.assertEqual(w.update(limit,**v),'pour_progress_unconfirmed')


if __name__=='__main__':unittest.main()
