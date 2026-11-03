import unittest
import numpy as np
from fountain_hinge import PIVOT, OBSTACLES, hinge_trial, motion, rotation, to_container, touchdown, obstacle_metrics, floor_diagnostics
from fountain_plan import BASE


class HingeGeometryTests(unittest.TestCase):
    def setUp(self): self.spec=hinge_trial(BASE)

    def test_pivot_stationary_and_bottom_stays_above_table(self):
        corners=np.array([[x,y,-.012] for x in [-.055,.055] for y in [-.045,.045]])
        for t in np.linspace(0,10,201):
            angle,_,_=motion(t,self.spec)
            world=(corners-PIVOT)@rotation(angle).T+PIVOT
            self.assertGreaterEqual(world[:,2].min(),-.012-1e-10)
            np.testing.assert_allclose((PIVOT-PIVOT)@rotation(angle).T+PIVOT,PIVOT)
        angle,_,_=motion(4.6,self.spec)
        self.assertAlmostEqual(np.degrees(angle),22.)

    def test_container_frame_removes_only_rigid_motion(self):
        angle,rate,_=motion(4.1,self.spec)
        local=np.array([[.02,.01,.012],[-.03,-.02,.004]])
        r=rotation(angle); world=(local-PIVOT)@r.T+PIVOT
        velocity=np.cross([0.,-rate,0.],world-PIVOT)
        p,v=to_container(world,velocity,angle,rate)
        np.testing.assert_allclose(p,local,atol=1e-12)
        np.testing.assert_allclose(v,0,atol=1e-12)

    def test_returns_flush_and_control_never_moves(self):
        self.assertAlmostEqual(touchdown(self.spec),5.15)
        for t in [5.15,6.,10.]: self.assertAlmostEqual(motion(t,self.spec)[0],0.)
        for t in np.linspace(0,10,101): self.assertEqual(motion(t,hinge_trial(BASE,control=True))[0],0.)
        self.assertEqual(self.spec['slosh_speed'],0.)

    def test_obstacle_penetration_not_just_proximity(self):
        c=np.array(OBSTACLES[0]['center']); half=np.array(OBSTACLES[0]['size'])/2
        result=obstacle_metrics(np.array([c,c+[half[0]+.001,0,0]]),.0018)
        self.assertEqual(result['obstacle_penetration_counts'][0],1)
        self.assertEqual(result['obstacle_proximity_counts'][0],1)

    def test_finite_floor_and_actual_crossings(self):
        # One falling escaped droplet, one tunneling through the floor, one
        # entering below the slab from the side, and one inside the slab.
        old=np.array([[.08,0,-2.],[0,0,.01],[.08,0,-.03],[.01,0,.002]])
        new=np.array([[.03,0,-2.8],[0,0,-.10],[.03,0,-.04],[.01,0,-.005]])
        result=floor_diagnostics(new,old,.0018)
        self.assertEqual(result['below_floor_plane_count'],4)
        self.assertEqual(result['floor_intrusion_count'],1)
        self.assertEqual(result['floor_crossing_count'],2)
        self.assertEqual(result['leak_count'],2)


if __name__=='__main__': unittest.main()
