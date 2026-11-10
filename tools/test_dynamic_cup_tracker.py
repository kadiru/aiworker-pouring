"""CPU safety-contract tests; synthetic inputs are not native task successes."""
import sys,unittest
from pathlib import Path
sys.path[:0]=['/workspace/src',str(Path(__file__).parent)]
import numpy as np
from dynamic_cup_tracker import CupTracker

class Contracts(unittest.TestCase):
    def tracker(self):
        t=object.__new__(CupTracker);t.device='cpu';t.history=[];t.anchor=None;t.template=None;t.failed=False
        t.geometry=dict(bottom_z_m=.004,rim_z_m=.038)
        t.artifact=dict(threshold=.4,consensus=dict(window=20,votes=8,radius_px=4.,fraction=.6,score_floor=.2),settings=dict(association_pixels=24.,minimum_similarity=.75,maximum_movement_mm=1.5,search_pixels=16),calibration=dict(K=[[1000,0,512],[0,1000,512],[0,0,1]],T=[[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]],table_z=1.04))
        t.detect=lambda image,refine=True:(np.array([512.,512.]),.9);t.establish=lambda image:True;t.match=lambda image:(1.,0.,[0,0]);return t
    def ready(self):
        t=self.tracker();image=np.full((1024,1024,3),180,np.uint8)
        for _ in range(8):r=t.update(image)
        self.assertTrue(r['valid']);return t,image
    def test_requires_eight_views(self):
        t=self.tracker();im=np.full((1024,1024,3),180,np.uint8)
        for _ in range(7):self.assertFalse(t.update(im)['valid'])
        self.assertTrue(t.update(im)['valid'])
    def test_anchor_does_not_chase_coarse_jitter(self):
        t,im=self.ready();t.detect=lambda *a,**k:(np.array([518.,509.]),.8)
        self.assertEqual(t.update(im)['uv'],[512.,512.])
    def test_absence_latches_rejection(self):
        t,im=self.ready();t.detect=lambda *a,**k:(np.array([512.,512.]),.1)
        self.assertEqual(t.update(im)['reason'],'cup_absent_or_uncertain')
        t.detect=lambda *a,**k:(np.array([512.,512.]),.9)
        self.assertEqual(t.update(im)['reason'],'latched_rejection')
    def test_association_loss_rejects(self):
        t,im=self.ready();t.detect=lambda *a,**k:(np.array([550.,512.]),.9)
        self.assertFalse(t.update(im)['valid'])
    def test_motion_and_changed_reference_reject(self):
        for similarity,movement in ((1.,1.6),(.7,0.)):
            t,im=self.ready();t.match=lambda image:(similarity,movement,[3,0])
            self.assertEqual(t.update(im)['reason'],'cup_moved_or_reference_changed')
    def test_native_pixel_registration(self):
        t=self.tracker();del t.match;t.bounds=np.array([450,450,570,490])
        rng=np.random.default_rng(713);im=np.full((1024,1024,3),180,np.uint8);im[450:490,450:570]=rng.integers(30,220,(40,120,3),dtype=np.uint8)
        t.template=t.features(im)[:,:,450:490,450:570].clone();score,movement,shift=t.match(im)
        self.assertGreater(score,.999);self.assertEqual(shift,[0.,0.]);self.assertEqual(movement,0.)
        moved=np.full_like(im,180);moved[:,8:]=im[:,:-8];score,movement,shift=t.match(moved)
        self.assertGreater(score,.999);self.assertEqual(shift,[8.,0.]);self.assertGreater(movement,1.5)
    def test_partial_occlusion_needs_two_stationary_patches(self):
        t=self.tracker();del t.match;t.bounds=np.array([450,450,570,490]);t.artifact['settings'].update(template_parts=3,minimum_support=2)
        rng=np.random.default_rng(714);im=np.full((1024,1024,3),180,np.uint8);im[450:490,450:570]=rng.integers(30,220,(40,120,3),dtype=np.uint8)
        t.template=t.features(im)[:,:,450:490,450:570].clone()
        partial=im.copy();partial[450:490,450:488]=180;score,movement,shift=t.match(partial)
        self.assertGreaterEqual(score,.75);self.assertLessEqual(movement,1.5)
        hidden=im.copy();hidden[450:490,450:530]=180;score,_,_=t.match(hidden);self.assertLess(score,.75)
        moved=np.full_like(im,180);moved[:,8:]=im[:,:-8];score,_,_=t.match(moved);self.assertLess(score,.75)
if __name__=='__main__':unittest.main()
