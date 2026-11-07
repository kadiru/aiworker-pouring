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
        self.assertTrue(torch.isfinite(x).all());self.assertEqual(x.shape,(2,604))
        self.assertLess(x[0,340],x[1,340])
        old=dict(columns=[0,4],mean=[0.,0.],scale=[1.,1.],coef=[0.,1.])
        np.testing.assert_allclose(predict(x,old),[.9,.7],atol=1e-6)

    def test_residual_uses_the_same_clipped_baseline_as_training(self):
        x=torch.tensor([[1.,1.2],[1.,-.2]])
        fitted=dict(columns=[0,1],mean=[0.,0.],scale=[1.,1.],coef=[0.,1.],residual_mlp=dict(columns=[0],mean=[0.],scale=[1.],layers=[dict(weight=[[0.]],bias=[-.1])]))
        np.testing.assert_allclose(predict(x,fitted),[.9,0.],atol=1e-6)

if __name__=='__main__':unittest.main()
