import unittest
import numpy as np
from scale_bottle_mass import moments


class MassTests(unittest.TestCase):
    def test_uniform_geometric_scaling_has_correct_physical_powers(self):
        p=np.array(((.004,.025),(.082,.025),(.107,.009),(.135,.009)))
        a=moments(p,density=2500);factor=1.7
        b=moments(p*factor,thickness=.0024*factor,density=2500)
        self.assertAlmostEqual(b['mass_kg']/a['mass_kg'],factor**3)
        np.testing.assert_allclose(np.array(b['diagonal_inertia_kg_m2'])/a['diagonal_inertia_kg_m2'],factor**5)
        self.assertAlmostEqual(b['center_of_mass_m'][2]/a['center_of_mass_m'][2],factor)

    def test_subdividing_linear_profile_does_not_change_properties(self):
        a=moments(((.004,.025),(.13,.012)))
        b=moments(((.004,.025),(.067,.0185),(.13,.012)))
        for k in a:np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-15)

    def test_rejects_reversed_or_nonphysical_profile(self):
        for p in (((.1,.02),(.05,.01)),((-.001,.02),(.1,.01)),((.01,float('nan')),(.1,.01))):
            with self.assertRaises(ValueError):moments(p)


if __name__=='__main__':unittest.main()
