"""Synthetic gate checks; not native perception or task-success evidence."""
import unittest
from evaluate_dynamic_holdout import error_metrics,level_ok
G=dict(level_validation_mae=.02,level_validation_near_max_error=.035,level_validation_near_overestimate=.025)
class Checks(unittest.TestCase):
    def test_exact_predictions(self):self.assertTrue(level_ok(error_metrics([.3,.95],[.3,.95]),G))
    def test_near_overestimate_cannot_hide_in_average(self):self.assertFalse(level_ok(error_metrics([.3,.976],[.3,.95]),G))
    def test_near_underestimate_maximum(self):self.assertFalse(level_ok(error_metrics([.3,.914],[.3,.95]),G))
    def test_low_fill_mean_limit(self):self.assertFalse(level_ok(error_metrics([.33,.33],[.3,.3]),G))
    def test_missing_prediction_rejected(self):self.assertFalse(level_ok(error_metrics([float('nan')],[.95]),G))
    def test_low_fill_has_no_near_full_samples(self):
        m=error_metrics([.31],[.3]);self.assertIsNone(m['near_max']);self.assertTrue(level_ok(m,G))
if __name__=='__main__':unittest.main()
