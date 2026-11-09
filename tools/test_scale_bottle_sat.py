import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scale_bottle_sat import convex_overlap


class SATTests(unittest.TestCase):
    def test_rotated_intersection_separation_and_touching(self):
        box=np.array([(x,y,z) for x in (-1,1) for y in (-1,1) for z in (-1,1)],float)*.01
        rot=Rotation.from_euler('xyz',[17,38,21],degrees=True).as_matrix()
        for center,expected in (([0.,0.,0.],True),([.018,0.,0.],True),([.05,.05,.05],False)):
            a=box@rot.T+np.array([.5,-.2,1.04]);b=box@rot.T+np.array([.5,-.2,1.04])+np.asarray(center)@rot.T
            self.assertEqual(convex_overlap(a,b)['intersect'],expected)
        self.assertTrue(convex_overlap(box,box+[.02,0,0])['intersect'])

    def test_rejects_nonfinite_coordinates(self):
        with self.assertRaises(ValueError):convex_overlap(np.full((8,3),np.nan),np.ones((8,3)))


if __name__=='__main__':unittest.main()
