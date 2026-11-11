"""Known geometry round trips and crop-origin invariance, independent of fit scores."""
import unittest
import numpy as np
import torch
from pour_guard_vision import project
from physical_level_augmentation import crop_maps
from landmark_level_model import level_from_points


class LandmarkGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cal = dict(K=[[900., 0., 512.], [0., 900., 512.], [0., 0., 1.]],
                        T=[[1., 0., 0., .5], [0., 1., 0., -.2], [0., 0., 1., 2.], [0., 0., 0., 1.]], table_z=1.04)
        self.g = dict(rim_z_m=.038, bottom_z_m=.004, rim_radius_m=.024)
        self.actual = np.array([.55, -.24])

    def points(self, crop_xy, level):
        A, b = crop_maps(crop_xy, [self.g], self.cal)
        uv = np.stack([project(np.r_[self.actual, z], self.cal) for z in
                       (1.078, 1.044+.034*level)])
        return torch.linalg.solve(A[0], (torch.tensor(uv)-b[0]).T).T[None]

    def test_low_mid_high_round_trip(self):
        xy = torch.tensor(self.actual[None])
        for value in (.1, .28, .5, .9, .98):
            predicted = level_from_points(self.points(xy, value), xy, [self.g], self.cal)
            self.assertAlmostEqual(float(predicted[0]), value, places=10)

    def test_estimated_crop_origin_does_not_bias_height(self):
        for dx, dy in ((.002, -.001), (-.001, .0015), (0, 0)):
            xy = torch.tensor((self.actual+[dx, dy])[None])
            value = level_from_points(self.points(xy, .285), xy, [self.g], self.cal)
            self.assertAlmostEqual(float(value[0]), .285, places=10)

    def test_geometry_has_finite_nonzero_gradient(self):
        xy = torch.tensor(self.actual[None]);points = self.points(xy, .7).requires_grad_()
        value = level_from_points(points, xy, [self.g], self.cal)
        value.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(float(points.grad.abs().sum()), .001)


if __name__ == '__main__': unittest.main()
