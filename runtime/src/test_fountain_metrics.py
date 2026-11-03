"""Meaningful synthetic checks for physical failure gates; not simulator validation."""
import unittest
import numpy as np
from fountain_plan import candidate
from fountain_metrics import sample,summarize,slosh_response

class MetricsTests(unittest.TestCase):
    def test_floor_gate_distinguishes_retention_from_leak(self):
        spec=candidate('test');n=4000
        p=np.tile([0.,0.,.005],(n,1));v=np.zeros_like(p)
        row=sample(p,v,spec,6.,n,75.,.3)
        self.assertEqual(row['leak_count'],0)
        self.assertTrue(summarize([row],spec)['eligible'])
        p[:,2]=-.005
        row=sample(p,v,spec,6.,n,75.,.3)
        self.assertEqual(row['leak_count'],n)
        self.assertIn('floor_leak',summarize([row],spec)['gates'])
    def test_missing_particles_cannot_win(self):
        spec=candidate('test');p=np.array([[0,0,.005]])
        row=sample(p,np.zeros_like(p),spec,6.,100,75.,.3)
        self.assertIn('particle_count_changed',summarize([row],spec)['gates'])
    def test_render_visibility_is_independent_of_physics(self):
        spec=candidate('test');p=np.array([[0.,0.,.005],[.0027,0,.005]])
        row=sample(p,np.zeros_like(p),spec,6.,2,75.,.3,mesh=np.array([[2.,2.,2.]]))
        self.assertEqual(row['retention'],1.)
        self.assertEqual(row['mesh_particle_coverage'],0.)
    def test_slosh_fit_recovers_known_frequency_and_decay(self):
        spec=candidate('test',slosh_time=3.5,slosh_speed=.08)
        ts=np.arange(.2,4.5,.05)
        rows=[dict(finite=True,t=3.5+t,pool_com_x=.003*np.exp(-.3*t)*np.sin(2*np.pi*.95*t),pool_volume_ml=75.) for t in ts]
        response=slosh_response(rows,spec)
        self.assertTrue(response['valid'])
        self.assertAlmostEqual(response['frequency_hz'],.95,places=2)
        self.assertAlmostEqual(response['decay_per_s'],.3,places=2)
    def test_monotonic_drift_is_not_sloshing(self):
        spec=candidate('test',slosh_time=3.5,slosh_speed=.08)
        rows=[dict(finite=True,t=3.5+t,pool_com_x=.003*np.exp(-t),pool_volume_ml=75.) for t in np.arange(.2,4.5,.05)]
        self.assertFalse(slosh_response(rows,spec)['valid'])
    def test_escaped_free_fall_is_not_local_instability(self):
        spec=candidate('test')
        p=np.array([[0.,0.,.005],[.5,0.,-2.]])
        v=np.array([[0.,0.,0.],[0.,0.,-6.]])
        row=sample(p,v,spec,6.,2,75.,.3)
        self.assertEqual(row['speed_p99'],0.)
        self.assertGreater(row['all_particle_speed_p99'],5.)
        p[1]=[.01,0.,.01]
        self.assertGreater(sample(p,v,spec,6.,2,75.,.3)['speed_p99'],5.)

if __name__=='__main__': unittest.main()
