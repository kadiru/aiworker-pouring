"""Regressions for opt-in continuous departure and level feature stability."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from pour_wide_motion import centered_return_path,UPRIGHT_SECONDS,TRANSFER_SECONDS
from pour_settled_model import features,predict

class FinishTests(unittest.TestCase):
    def test_natural_return_has_no_stationary_drain_segment(self):
        def pose(planner,q,source,cup,z,angle,mouth_z,centered):return np.full(7,angle/180.)
        def transit(planner,start,end,**kw):return np.stack((start,end))
        with patch('pour_wide_motion.project_target',side_effect=pose),patch('pour_workspace_path.clearance_approach',side_effect=transit):
            args=(None,np.ones(7),100.,[.5,-.1],[.5,-.3],1.04,np.full(7,.4),.052)
            old_t,old_q=centered_return_path(*args)
            t,q=centered_return_path(*args,drain_hold_s=0.)
        self.assertAlmostEqual(t[-1],UPRIGHT_SECONDS+TRANSFER_SECONDS)
        self.assertAlmostEqual(old_t[-1]-t[-1],1.2)
        after=np.flatnonzero(t>UPRIGHT_SECONDS+1e-6)[0]
        self.assertGreater(np.linalg.norm(q[after]-q[after-1]),0.)
        np.testing.assert_allclose(q[-1],np.full(7,.4))

    def test_new_envelope_features_do_not_change_legacy_predictions(self):
        a=torch.full((2,96,96,3),180,dtype=torch.uint8)
        a[0,35:75,35:73]=torch.tensor([180,15,30],dtype=torch.uint8)
        a[1,45:75,35:73]=torch.tensor([180,15,30],dtype=torch.uint8)
        x=features(a,torch.tensor([.9,.7]),[.49,-.15])
        self.assertTrue(torch.isfinite(x).all());self.assertEqual(x.shape,(2,709))
        self.assertLess(x[0,340],x[1,340])
        old=dict(columns=[0,4],mean=[0.,0.],scale=[1.,1.],coef=[0.,1.])
        np.testing.assert_allclose(predict(x,old),[.9,.7],atol=1e-6)

    def test_spatial_surface_shape_distinguishes_equal_row_averages(self):
        a=torch.full((2,96,96,3),180,dtype=torch.uint8)
        a[0,35:65,40:48]=torch.tensor([180,15,30],dtype=torch.uint8)
        a[1,35:65,60:68]=torch.tensor([180,15,30],dtype=torch.uint8)
        x=features(a,torch.tensor([.9,.9]),[.49,-.15])
        torch.testing.assert_close(x[0,:604],x[1,:604])
        self.assertGreater(float((x[0,604:]-x[1,604:]).abs().max()),.1)

    def test_residual_uses_the_same_clipped_baseline_as_training(self):
        x=torch.tensor([[1.,1.2],[1.,-.2]])
        fitted=dict(columns=[0,1],mean=[0.,0.],scale=[1.,1.],coef=[0.,1.],residual_mlp=dict(columns=[0],mean=[0.],scale=[1.],layers=[dict(weight=[[0.]],bias=[-.1])]))
        np.testing.assert_allclose(predict(x,fitted),[.9,0.],atol=1e-6)

    def test_calibration_cannot_trigger_timeout_motion(self):
        from robot_pour import RobotPour
        from types import SimpleNamespace
        robot=object.__new__(RobotPour);robot.spec=dict(calibration_only=True,max_seconds=2.5)
        robot.approach=np.ones((2,7));robot.drives=[SimpleNamespace(Set=lambda value:None) for _ in range(11)]
        robot.stop_time=.1
        for t in (.1,1.,2.5,3.):
            robot.drive(t);np.testing.assert_allclose(robot.command,np.r_[np.ones(7),np.zeros(4)])
        self.assertFalse(robot.complete(2.4));self.assertTrue(robot.complete(2.5))

    def test_calibration_pack_stays_inside_tapered_cup(self):
        from pour_finish_calibration import cup_points
        from bottle_pour import CUP
        spec=dict(calibration_only=True,calibration_version=1,calibration_pitch_fraction=.8,calibration_initial_height=.99,spacing=.0018)
        points=cup_points(spec)
        self.assertGreater(len(points),10000)
        self.assertLess(points[:,2].max(),CUP[-1][0])
        radius=np.interp(points[:,2],[p[0] for p in CUP],[p[1] for p in CUP])
        self.assertTrue((np.linalg.norm(points[:,:2],axis=1)<radius-.002).all())

if __name__=='__main__':unittest.main()
