"""Boundary contracts: reject color leakage and preserve fail-closed validity."""
import unittest
from unittest.mock import patch
import torch
import ensemble_level_model as module


class EnsembleContracts(unittest.TestCase):
    def setUp(self):
        self.artifact = dict(kind='rgb_equal_ensemble_v1', branch='rgb', weights=[.5,.5], members=[
            dict(kind='crop_tolerant_outline_v2',branch='rgb'),dict(kind='geometry_landmarks_v1',branch='rgb')])

    def test_color_cannot_enter_rgb(self):
        self.artifact['members'][1]['branch']='color'
        with self.assertRaises(ValueError):module.predict_ensemble_level(None,None,None,self.artifact)

    def test_invalid_member_is_not_hidden_by_average(self):
        with patch.object(module,'predict_repair',return_value=(torch.tensor([.4,float('nan')]),torch.tensor([True,False]))), patch.object(module,'predict_landmark_level',return_value=(torch.tensor([.6,.6]),torch.tensor([True,True]))):
            p,v=module.predict_ensemble_level(None,None,None,self.artifact)
            self.assertAlmostEqual(float(p[0]),.5);self.assertTrue(v[0]);self.assertFalse(v[1]);self.assertTrue(torch.isnan(p[1]))

    def test_precision_restored_even_on_error(self):
        before=torch.backends.cuda.matmul.allow_tf32;conv=torch.backends.cudnn.allow_tf32
        def broken(*args):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            raise RuntimeError('test failure')
        with patch.object(module,'predict_landmark_level',side_effect=broken):
            with self.assertRaises(RuntimeError):module.predict_ensemble_level(None,None,None,self.artifact)
        self.assertEqual(before,torch.backends.cuda.matmul.allow_tf32);self.assertEqual(conv,torch.backends.cudnn.allow_tf32)

if __name__=='__main__':unittest.main()
