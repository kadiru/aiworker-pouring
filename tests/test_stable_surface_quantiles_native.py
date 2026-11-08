"""Run with Isaac's Python; the Mac project environment need not install Torch."""
import importlib.util,sys,unittest
from pathlib import Path
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    sys.path[:0]=['/workspace/learning/A-perception-20261108-v1/staged','/workspace/src',str(Path(__file__).resolve().parents[1]/'runtime/src')]
    from pour_settled_model import stable_color_quantiles

@unittest.skipUnless(HAS_TORCH,'Native Torch checks run inside the owned Isaac environment')
class Quantiles(unittest.TestCase):
    def histogram(self):
        h=torch.zeros(3,96)
        h[0,20]=1/38;h[0,40]=1/38
        h[1,10:90]=1/38
        return h
    def test_exact_boundary_chooses_first_cumulative_row(self):
        q=stable_color_quantiles(self.histogram())
        torch.testing.assert_close(q[0]*95,torch.tensor([20,20,20,20,20,40,40,40,40.]),atol=1e-5,rtol=0)
        torch.testing.assert_close(q[1,4]*95,torch.tensor(49.),atol=1e-5,rtol=0)
    def test_empty_color_support_is_zero(self):
        self.assertTrue(torch.equal(stable_color_quantiles(self.histogram())[2],torch.zeros(9)))
    def test_cpu_gpu_agree_at_count_boundaries(self):
        if not torch.cuda.is_available():self.skipTest('CUDA comparison requires guarded native GPU')
        torch.testing.assert_close(stable_color_quantiles(self.histogram()),stable_color_quantiles(self.histogram().cuda()).cpu(),atol=1e-7,rtol=0)
if __name__=='__main__':unittest.main()
