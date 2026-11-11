"""Check that geometry conditioning preserves initialization and can be learned."""
import unittest
import torch
from landmark_level_model import LandmarkLevelNet as Base
from landmark_level_model_v2 import LandmarkLevelNet, initialize_from_base


class ConditionedLandmarkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(45)
        self.image = torch.randint(0, 256, (2, 96, 96, 3), dtype=torch.uint8)
        self.geometry = [dict(bottom_z_m=.004, rim_z_m=.0346, rim_radius_m=.0264)]*2

    def test_base_initialization_is_preserved(self):
        base=Base().eval();model=LandmarkLevelNet().eval()
        initialize_from_base(model,base.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(model(self.image,self.geometry),base(self.image),atol=2e-7,rtol=1e-5)

    def test_geometry_channels_have_finite_gradients(self):
        model=LandmarkLevelNet();model(self.image,self.geometry).square().mean().backward()
        grad=model.net[0].weight.grad[:,3:]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
