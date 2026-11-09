import unittest
import torch
from scale_locator_augmentation import translate


class TranslationTests(unittest.TestCase):
    def test_integer_and_fractional_image_motion_matches_labels(self):
        image=torch.full((1,32,48,3),180.);image[0,12,20]=255.;uv=torch.tensor([[20.,12.]])
        for dx,dy in ((3.,-2.),(.375,-.625),(-1.25,2.75)):
            shifted,target=translate(image,uv,dx,dy);mass=shifted[0,:,:,0]-180
            yy,xx=torch.meshgrid(torch.arange(32),torch.arange(48),indexing='ij')
            center=torch.stack(((mass*xx).sum(),(mass*yy).sum()))/mass.sum()
            torch.testing.assert_close(center,torch.tensor([20+dx,12+dy]),rtol=0,atol=1e-3)
            torch.testing.assert_close(target[0],center,rtol=0,atol=1e-3)
            torch.testing.assert_close(shifted[0,0,0],torch.full((3,),180.))
        self.assertEqual(float(image[0,12,20,0]),255.)
        torch.testing.assert_close(uv,torch.tensor([[20.,12.]]))

    def test_identity_preserves_pixels_and_coordinates(self):
        torch.manual_seed(12);image=torch.randint(0,256,(2,32,48,3)).float();uv=torch.tensor([[20.,12.],[10.,8.]])
        shifted,target=translate(image,uv,0.,0.)
        torch.testing.assert_close(shifted,image,rtol=0,atol=.002)
        torch.testing.assert_close(target,uv)


if __name__=='__main__':unittest.main()
