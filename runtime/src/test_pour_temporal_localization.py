import unittest
from pour_temporal_localization import consensus


class TemporalLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(window=20,votes=6,radius_px=1.5,fraction=.6,score_floor=.24)

    def test_one_good_frame_cannot_move_robot(self):
        self.assertIsNone(consensus([([121.,144.],.7)],.3438,self.cfg))

    def test_consistent_weak_observations_need_original_confidence_anchor(self):
        rows=[([121.+i*.02,144.],.32) for i in range(6)]
        self.assertIsNone(consensus(rows,.3438,self.cfg))
        rows.append(([121.1,144.],.35))
        self.assertIsNotNone(consensus(rows,.3438,self.cfg))

    def test_retained_large_outliers_do_not_shift_consensus(self):
        rows=[([121.6,144.4],.30),([142.8,140.],.31)]*2
        rows += [([121.7,144.3],.30)]*5+[([121.8,144.2],.35)]
        got=consensus(rows,.3438,self.cfg)
        self.assertIsNotNone(got);self.assertLess(abs(got['uv'][0]-121.7),.2)

    def test_disagreement_and_nonfinite_inputs_cannot_pass(self):
        self.assertIsNone(consensus([([100.,100.],.9),([150.,150.],.9)]*6,.3438,self.cfg))
        self.assertIsNone(consensus([([float('nan'),100.],.9)]*6,.3438,self.cfg))
        self.assertIsNone(consensus([([100.,100.],float('nan'))]*6,.3438,self.cfg))

    def test_expired_anchor_cannot_confirm_new_weak_window(self):
        rows=[([121.,144.],.7)]+[([121.,144.],.3)]*20
        self.assertIsNone(consensus(rows,.3438,self.cfg))


if __name__=='__main__':unittest.main()
