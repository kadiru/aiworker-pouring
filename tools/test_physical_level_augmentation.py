"""Contracts for image/geometry-consistent perturbation, independent of fit scores."""
import unittest
import numpy as np
import torch
from pour_guard_vision import project
from physical_level_augmentation import crop_maps, perturb_crop


class PhysicalCropTests(unittest.TestCase):
    def setUp(self):
        self.cal = dict(K=[[900., 0., 512.], [0., 900., 512.], [0., 0., 1.]],
                        T=[[1., 0., 0., .5], [0., 1., 0., -.2], [0., 0., 1., 2.], [0., 0., 0., 1.]], table_z=1.04)
        self.g = dict(rim_z_m=.038, bottom_z_m=.004, rim_radius_m=.024)
        self.xy = torch.tensor([[.48, -.24], [.53, -.21]], dtype=torch.float64)

    def test_matches_native_projection(self):
        matrix, offset = crop_maps(self.xy, [self.g]*2, self.cal)
        for i, xy in enumerate(self.xy.numpy()):
            for point, height in (([48., 24.], .038), ([48., 64.], .004)):
                actual = matrix[i].numpy()@point+offset[i].numpy()
                expected = project(np.r_[xy, 1.04+height], self.cal)
                np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_zero_offset_preserves_image(self):
        image = torch.arange(2*96*96*3, dtype=torch.float64).reshape(2, 96, 96, 3)%256
        result = perturb_crop(image, self.xy, [self.g]*2, self.cal, torch.zeros_like(self.xy))
        torch.testing.assert_close(result, image, atol=1e-9, rtol=0)

    def test_same_world_point_maps_consistently(self):
        delta = torch.tensor([[.001, -.001], [-.0015, .0003]], dtype=torch.float64)
        old, old_b = crop_maps(self.xy, [self.g]*2, self.cal)
        new, new_b = crop_maps(self.xy+delta, [self.g]*2, self.cal)
        point = torch.tensor([[43., 38.], [53., 60.]], dtype=torch.float64)
        world_pixel = (old@point[..., None])[..., 0]+old_b
        destination = (torch.linalg.inv(new)@(world_pixel-new_b)[..., None])[..., 0]
        torch.testing.assert_close((new@destination[..., None])[..., 0]+new_b, world_pixel)
        self.assertGreater(float((destination-point).abs().max()), .1)


if __name__ == '__main__':
    unittest.main()
