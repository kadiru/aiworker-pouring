import unittest
from pour_cup_locator import combine_predictions,correct_prediction

class CupFusion(unittest.TestCase):
    def test_one_high_confidence_hallucination_cannot_confirm_position(self):
        _,score=combine_predictions([([35.,35.],.99),([100.,100.],.99)],6.)
        self.assertEqual(score,0.)
    def test_agreement_does_not_hide_one_uncertain_detector(self):
        _,score=combine_predictions([([35.,35.],.95),([36.,35.],.2)],6.)
        self.assertEqual(score,.2)
    def test_large_calibration_adjustment_is_rejected(self):
        bank=dict(affine_uv_correction=[[4.,0.],[0.,0.],[0.,0.]],max_correction_px=3.)
        _,score=correct_prediction(([192.,192.],.99),bank)
        self.assertEqual(score,0.)

if __name__=='__main__':unittest.main()
