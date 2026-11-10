"""Synthetic gate checks; not native perception or task-success evidence."""
import unittest
from evaluate_dynamic_holdout import error_metrics,level_ok,localization_ok
G=dict(level_validation_mae=.02,level_validation_near_max_error=.035,level_validation_near_overestimate=.025)
class Checks(unittest.TestCase):
    def test_exact_predictions(self):self.assertTrue(level_ok(error_metrics([.3,.95],[.3,.95]),G))
    def test_near_overestimate_cannot_hide_in_average(self):self.assertFalse(level_ok(error_metrics([.3,.976],[.3,.95]),G))
    def test_near_underestimate_maximum(self):self.assertFalse(level_ok(error_metrics([.3,.914],[.3,.95]),G))
    def test_low_fill_mean_limit(self):self.assertFalse(level_ok(error_metrics([.33,.33],[.3,.3]),G))
    def test_missing_prediction_rejected(self):self.assertFalse(level_ok(error_metrics([float('nan')],[.95]),G))
    def test_low_fill_has_no_near_full_samples(self):
        m=error_metrics([.31],[.3]);self.assertIsNone(m['near_max']);self.assertTrue(level_ok(m,G))
    def test_all_forty_locations_required(self):
        row=dict(localization_valid=True,localization_error_mm=1.,score=.9)
        self.assertTrue(localization_ok([row]*40,3.,.4));self.assertFalse(localization_ok([row]*39,3.,.4))
    def test_tracking_failure_cannot_hide_behind_good_level(self):
        row=dict(localization_valid=True,localization_error_mm=1.,score=.9)
        for bad in (dict(row,localization_valid=False,localization_error_mm=None),dict(row,localization_error_mm=float('nan')),dict(row,localization_error_mm=3.),dict(row,score=.3)):
            self.assertFalse(localization_ok([row]*39+[bad],3.,.4))
if __name__=='__main__':unittest.main()
