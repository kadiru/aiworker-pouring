import math,unittest
from bottle_curriculum_geometry import BASE,profile,volume,geometry

class BottleGeometryTests(unittest.TestCase):
    def test_baseline_shape_capacity_and_mass(self):
        self.assertAlmostEqual(volume(BASE),volume(profile()))
        self.assertAlmostEqual(geometry(1,1)['proposed_empty_mass_kg'],.18)
    def test_height_changes_above_grasp_band(self):
        for h in (.85,1,1.15):
            p=profile(h,1)
            self.assertEqual(p[:2],((.004,.025),(.065,.025)))
            self.assertAlmostEqual(p[-1][0],.135*h)
            self.assertTrue(all(a[0]<b[0] for a,b in zip(p,p[1:])))
    def test_width_and_neck(self):
        for w in (.9,.95,1):
            g=geometry(1,w)
            self.assertLessEqual(g['outer_body_diameter_m'],.0548)
            self.assertEqual(g['inner_mouth_diameter_m'],.018)
            self.assertEqual(g['grasp_height_m'],.048)
    def test_reject_wider_unbounded_or_nonfinite(self):
        for h,w in [(1,1.01),(.84,1),(1.16,1),(1,.89),(float('nan'),1),(1,float('inf'))]:
            with self.assertRaises(ValueError):profile(h,w)
    def test_no_physical_acceptance(self):
        for h in (.85,1.15):
            for w in (.9,1):
                g=geometry(h,w)
                self.assertGreater(g['capacity_geometric_ml'],0)
                self.assertFalse(g['native_grasp_accepted'])
                self.assertFalse(g['native_inertia_accepted'])
                self.assertFalse(g['autonomy_accepted'])

if __name__=='__main__':unittest.main()
