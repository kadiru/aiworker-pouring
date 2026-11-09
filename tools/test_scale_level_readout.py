"""Synthetic invariants for size features and level inference, not accuracy."""
import sys,unittest
sys.path[:0]=['/workspace/src','/workspace']
import torch
from scale_level_readout import geometry_tensor,level_features,predict_readout,predict_static_level

class LevelChecks(unittest.TestCase):
    def test_declared_geometry_is_explicit_and_bounded(self):
        im=torch.zeros((2,96,96,3),dtype=torch.uint8)
        g=[dict(bottom_z_m=.004,rim_z_m=.0346,rim_radius_m=.0264),dict(bottom_z_m=.004,rim_z_m=.0414,rim_radius_m=.0216)]
        a=geometry_tensor(g,im)
        torch.testing.assert_close(a,torch.tensor([[.9,1.1,1.1/.9],[1.1,.9,.9/1.1]],dtype=torch.float64))
        g[0]['rim_z_m']=.2
        with self.assertRaises(ValueError):geometry_tensor(g,im)
    def test_size_features_change_without_changing_rgb(self):
        im=torch.zeros((2,96,96,3),dtype=torch.uint8);im[:,30:60,36:70,0]=180
        g=[dict(bottom_z_m=.004,rim_z_m=.038,rim_radius_m=.024),dict(bottom_z_m=.004,rim_z_m=.038,rim_radius_m=.0264)]
        f=level_features(im,torch.tensor([.5,.5]),[.5,-.2],g,'compact_geometry')
        self.assertTrue(torch.isfinite(f).all());self.assertGreater(float((f[0]-f[1]).abs().max()),.01)
    def test_ridge_readout_and_clamp(self):
        f=dict(kind='ridge',mean=[1,2],scale=[2,1],coef=[.5,.1],bias=.2)
        torch.testing.assert_close(predict_readout(torch.tensor([[1.,2.],[3,3],[100,100]],dtype=torch.float64),f),torch.tensor([.2,.8,1.],dtype=torch.float64))
    def test_kernel_center_and_shared_inference(self):
        im=torch.zeros((1,96,96,3),dtype=torch.uint8);im[:,30:60,36:70,0]=180
        geom=dict(bottom_z_m=.004,rim_z_m=.038,rim_radius_m=.024);cnn=torch.tensor([.5]);xy=[.5,-.2]
        x=level_features(im,cnn,xy,geom,'compact_geometry');d=x.shape[1]
        f=dict(kind='rbf',features='compact_geometry',mean=x[0].tolist(),scale=[1]*d,centers=[[0]*d],alpha=[.3],bias=.2,length_scale=1.)
        torch.testing.assert_close(predict_static_level(im,cnn,xy,geom,dict(size_level_readout=f)),torch.tensor([.5],dtype=torch.float64))
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
