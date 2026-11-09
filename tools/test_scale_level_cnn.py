"""Context and zero-initialization invariants for supervised CNN fine-tuning."""
import sys,unittest
sys.path[:0]=['/workspace/src','/workspace']
import torch
from scale_level_cnn import ScaleLevelCNN,level_context,predict_size_cnn
from scale_level_readout import predict_static_level
class CNNChecks(unittest.TestCase):
    def test_zero_correction_retains_image_prediction(self):
        torch.manual_seed(1);m=ScaleLevelCNN().eval();im=torch.randint(0,256,(2,96,96,3),dtype=torch.uint8)
        g=[dict(bottom_z_m=.004,rim_z_m=.0346,rim_radius_m=.0264),dict(bottom_z_m=.004,rim_z_m=.0414,rim_radius_m=.0216)]
        c=level_context(im,g,[[.5,-.2],[.4,.03]])
        self.assertEqual(tuple(c.shape),(2,5))
        with torch.inference_mode():torch.testing.assert_close(m(im,c),m.cup(im),atol=0,rtol=0)
    def test_shared_dispatch_matches_model_and_batch_order(self):
        torch.manual_seed(2);m=ScaleLevelCNN().eval();im=torch.randint(0,256,(2,96,96,3),dtype=torch.uint8)
        g=dict(bottom_z_m=.004,rim_z_m=.038,rim_radius_m=.024);xy=[[.5,-.2],[.4,.03]]
        artifact=dict(size_cnn_identity='unit-test',size_cnn_state=m.state_dict())
        with torch.inference_mode():expected=m(im,level_context(im,g,xy))
        actual=predict_static_level(im,torch.zeros(2),xy,g,artifact)
        torch.testing.assert_close(actual,expected)
        reversed_prediction=predict_size_cnn(im.flip(0),list(reversed(xy)),g,artifact)
        torch.testing.assert_close(reversed_prediction,actual.flip(0))
if __name__=='__main__':torch.set_num_threads(2);unittest.main()
