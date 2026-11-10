"""Regression: per-episode errors cannot hide behind balanced fill averages."""
import unittest
import torch
from repair_level_model_v2 import episode_mae,repair_context
from repair_level_model import color_outline

class Checks(unittest.TestCase):
    def test_low_fill_failure_visible_despite_good_average(self):
        a=dict(y=torch.tensor([.3,.3,.95,.95]),meta=[dict(name=n) for n in ('low','low','high','high')])
        prediction=torch.tensor([.33,.33,.95,.95])
        self.assertLess(float((prediction-a['y']).abs().mean()),.02)
        self.assertGreater(episode_mae(prediction,a)['low'],.02)
    def test_absolute_table_position_not_fill_input(self):
        x=torch.zeros((2,96,96,3),dtype=torch.uint8)
        g=dict(rim_z_m=.038,bottom_z_m=.004,rim_radius_m=.024)
        c=repair_context(x,g,[[.45,-.2],[.55,-.3]])
        self.assertTrue(torch.equal(c[0],c[1]))
    def test_visible_outline_does_not_make_absent_liquid_valid(self):
        x=torch.full((1,96,96,3),180,dtype=torch.uint8);x[:,20:80,25:30]=30
        transformed,valid=color_outline(x)
        self.assertFalse(bool(valid[0]));self.assertTrue(bool((transformed[...,2]>0).any()))
    def test_color_observation_preserves_reference_outline(self):
        x=torch.full((1,96,96,3),180,dtype=torch.uint8)
        x[:,50:70,40:65]=torch.tensor([150,20,20],dtype=torch.uint8)
        transformed,valid=color_outline(x)
        self.assertTrue(bool(valid[0]));self.assertGreater(int(transformed[...,0].sum()),0)
        self.assertGreater(int(transformed[...,1].sum()),0);self.assertGreater(int(transformed[...,2].sum()),0)

if __name__=='__main__':unittest.main()
