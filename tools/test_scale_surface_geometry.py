"""Projection, scale normalization and synthetic surface recovery checks."""
import sys,unittest
sys.path[:0]=['/workspace/src','/workspace']
import numpy as np
import torch
from scale_surface_geometry import surface_lookup,crop_transform,envelope_features,COLUMNS
from pour_guard_vision import project

CAL=dict(K=[[1465.99853515625,0,512],[0,1465.99853515625,512],[0,0,1]],
 T=[[.675254067189361,-.3806611182473653,.6317666165593837,.85],
    [.7375852118529941,.34849257304336273,-.5783778883994359,-.55],
    [0,.8565337352307157,.5160910388794966,1.35],[0,0,0,1]],table_z=1.04)
def geometry(h=1.,d=1.):return dict(bottom_z_m=.004,rim_z_m=.004+.034*h,rim_radius_m=.024*d,profile=[[.004,.018*d],[.004+.034*h,.024*d]])
class SurfaceChecks(unittest.TestCase):
    def test_warp_preserves_rim_and_base_centers_for_all_sizes(self):
        for h in (.9,1.,1.1):
            for d in (.9,1.,1.1):
                g=geometry(h,d);xy=[.5,-.2];a=crop_transform(xy,CAL,g)
                np.testing.assert_allclose(a[:,:2]@[96,48]+a[:,2],project(np.r_[xy,1.04+g['rim_z_m']],CAL),atol=1e-9)
                np.testing.assert_allclose(a[:,:2]@[96,128]+a[:,2],project(np.r_[xy,1.04+g['bottom_z_m']],CAL),atol=1e-9)
    def test_projected_ring_heights_are_monotonic(self):
        for xy in ([.5,-.2],[.45,.03]):
            lookup=surface_lookup(xy,CAL,geometry())
            self.assertEqual(lookup.shape,(13,101));self.assertTrue((np.diff(lookup,axis=1)>0).all())
    def test_recovers_synthetic_surface_and_rejects_absence(self):
        g=geometry();lookup=surface_lookup([.5,-.2],CAL,g);image=np.zeros((1,192,192,3),dtype=np.uint8)
        # Independent raster construction from an actual projected ring at 60%.
        top=lookup[:,40]
        for column,y in zip(COLUMNS,top):
            x=int(round(column));row=int(round(y));image[0,row:row+24,x-2:x+3,0]=200
        f,baseline,valid=envelope_features(torch.from_numpy(image),lookup[None],g,[.5,-.2])
        self.assertTrue(valid.all());self.assertTrue(torch.isfinite(f).all());self.assertLess(abs(float(baseline[0])-.6),.02)
        _,_,valid=envelope_features(torch.zeros_like(torch.from_numpy(image)),lookup[None],g,[.5,-.2])
        self.assertFalse(valid.any())
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
