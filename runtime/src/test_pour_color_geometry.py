"""Regression checks for old color models and missing-color observations."""
import unittest
import torch
from pour_guard_vision import surface_features,color_geometry_features,color_geometry_readout

class ColorGeometry(unittest.TestCase):
    def test_old_checkpoint_readout_is_unchanged(self):
        torch.manual_seed(7)
        images=torch.randint(0,256,(3,96,96,3),dtype=torch.uint8)
        old=dict(dimensions=4,coef=[1.1,-1.4,.2,.1],bias=.015)
        features,_,_=surface_features(images)
        expected=(features@torch.tensor(old['coef'])+old['bias']).clamp(0,1)
        self.assertTrue(torch.equal(expected,color_geometry_readout(images,old)))
    def test_empty_and_saturated_views_remain_finite(self):
        images=torch.zeros(3,96,96,3,dtype=torch.uint8)
        images[1]=255;images[2,:,:,0]=255
        features=color_geometry_features(images)
        self.assertEqual(tuple(features.shape),(3,19))
        self.assertTrue(torch.isfinite(features).all())
        fit=dict(feature_kind='color_envelope_v1',dimensions=19,mean=[0.]*19,scale=[1.]*19,coef=[.5]+[0.]*18,bias=0.)
        self.assertTrue(torch.equal(color_geometry_readout(images,fit),torch.full((3,),.5)))
    def test_view_correction_requires_localized_position(self):
        images=torch.zeros(2,96,96,3,dtype=torch.uint8)
        fit=dict(feature_kind='color_envelope_pose_v2',dimensions=27,mean=[0.]*27,scale=[1.]*27,coef=[.5]+[0.]*26,bias=0.)
        with self.assertRaises(AssertionError):color_geometry_readout(images,fit)
        self.assertTrue(torch.equal(color_geometry_readout(images,fit,[.5,-.225]),torch.full((2,),.5)))
    def test_kernel_uses_only_supplied_visual_features_and_requires_pose(self):
        images=torch.zeros(1,96,96,3,dtype=torch.uint8)
        center=color_geometry_features(images,[.5,-.225]).double().tolist()
        fit=dict(feature_kind='color_kernel_v3',dimensions=27,mean=[0.]*27,scale=[1.]*27,centers=center,alpha=[.2],bias=.3,length_scale=1.)
        with self.assertRaises(AssertionError):color_geometry_readout(images,fit)
        self.assertAlmostEqual(float(color_geometry_readout(images,fit,[.5,-.225])[0]),.5)

if __name__=='__main__':unittest.main()
