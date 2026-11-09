"""Native CPU coordinate/inference-contract checks, separate from accuracy."""
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path[:0]=['/workspace/src','/workspace']
import scale_rim_refiner as r
from scale_rim_augmentation import affine

class RimChecks(unittest.TestCase):
    def test_affine_readout_bound_rejects_instead_of_clipping(self):
        bank=dict(affine_uv_correction=[[1.,0.],[0.,0.],[0.,0.]],max_affine_correction_pixels=2.)
        uv,score=r.correct_rim([500,400],.8,bank)
        np.testing.assert_allclose(uv,[501,400]);self.assertEqual(score,.8)
        bank['affine_uv_correction'][0][0]=3.
        uv,score=r.correct_rim([500,400],.8,bank)
        np.testing.assert_allclose(uv,[503,400]);self.assertEqual(score,0.)
    def test_affine_moves_image_and_label_together(self):
        image=torch.zeros((1,32,32,3));image[0,12,20]=255
        point=torch.tensor([[20.,12.]])
        for matrix,shift,expected in (([[1.,0.],[0.,1.]],[2.,3.],[22.,15.]),
                                      ([[0.,-1.],[1.,0.]],[0.,0.],[19.,20.])):
            out,label=affine(image,point,torch.tensor([matrix]),torch.tensor([shift]))
            np.testing.assert_allclose(label.numpy(),[expected],atol=1e-5)
            x,y=map(int,expected);self.assertGreater(float(out[0,y,x,0]),254.9)
    def test_padded_crop_has_exact_origin(self):
        image=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
        crop,origin=r.crop_native(image,[2.2,3.1],16)
        self.assertEqual(origin.tolist(),[-6,-5]);np.testing.assert_array_equal(crop[5:,6:],image[:11,:10])
        self.assertTrue((crop[:5]==180).all())
        target=np.array([7.25,8.5]);np.testing.assert_allclose((target-origin)+origin,target)
    def test_off_image_crop_does_not_wrap(self):
        crop,_=r.crop_native(np.zeros((32,32,3),dtype=np.uint8),[-100,100],16)
        self.assertTrue((crop==180).all())
    def test_softargmax_coordinates_are_native_pixels(self):
        logits=torch.full((1,64,64),-100.);logits[0,20,30]=0
        np.testing.assert_allclose(r.coordinates(logits).numpy(),[[122,82]],atol=1e-5)
    def test_subpixel_heatmap_and_translation(self):
        logits=torch.full((1,64,64),-100.);logits[0,20,30]=np.log(.75);logits[0,20,31]=np.log(.25)
        np.testing.assert_allclose(r.coordinates(logits).numpy(),[[123,82]],atol=1e-5)
        shifted=torch.roll(logits,(3,-2),(1,2))
        np.testing.assert_allclose(r.coordinates(shifted).numpy(),[[115,94]],atol=1e-5)
    def test_architecture_covers_native_crop(self):
        m=r.RimRefiner().eval()
        with torch.inference_mode():out=m(torch.zeros((2,256,256,3)))
        self.assertEqual(tuple(out.shape),(2,64,64));self.assertTrue(torch.isfinite(r.coordinates(out)).all())
    def test_inference_uses_predicted_crop_and_rejects_large_correction(self):
        class FakeFine:
            def __call__(self,x):
                logits=torch.full((1,64,64),-100.);logits[0,32,32]=0;return logits
        bank=dict(kind='native_rim_refiner',identity='test-contract',crop_size=256,max_correction_pixels=32.)
        r._CACHE[('test-contract','cpu')]=(object(),FakeFine())
        image=np.zeros((1024,1024,3),dtype=np.uint8)
        with patch.object(r,'native_coarse',return_value=(np.array([500.2,400.1]),.8)):
            uv,score=r.locate_static(image,bank,'cpu')
        np.testing.assert_allclose(uv,[502,402]);self.assertEqual(score,.8)
        bank['max_correction_pixels']=1.
        with patch.object(r,'native_coarse',return_value=(np.array([500.2,400.1]),.8)):
            _,score=r.locate_static(image,bank,'cpu')
        self.assertEqual(score,0.)
        r._CACHE.pop(('test-contract','cpu'))
if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
