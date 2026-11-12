import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pour_guard_actor import empty_confirmation
from pour_guard_vision import locate_initial_source
from pour_size_autonomy import (MAX_SETTLED_RGB_BIAS, MIDFILL_RGB_KNOTS, SizedCupPerception,
                                batch_specs, settled_pair, settled_rgb_weight, sha)
from pour_wide_motion import late_mouth_offset


class SizedStopTests(unittest.TestCase):
    def test_failed_cup_consensus_retries_before_establishing_an_anchor(self):
        cup = SizedCupPerception.__new__(SizedCupPerception)
        cup.retry_initial_cup = True
        cup.tracker = SimpleNamespace(anchor=None, failed=True, history=[(1., 2.)], update=Mock())
        cup.track(None)
        self.assertFalse(cup.tracker.failed)
        self.assertEqual(cup.tracker.history, [])
        cup.tracker.update.assert_called_once()

        cup.tracker.anchor = [4., 5.]
        cup.tracker.failed = True
        self.assertFalse(cup.track(None)["valid"])
        cup.tracker.update.assert_called_once()

    def test_tall_neck_fallback_preserves_primary_and_rejects_weak_matches(self):
        checkpoint = dict(source_bank={}, thresholds=dict(source_localization_score=.65),
                          source_initial_fallback=dict(bank={}, threshold=.5, presence_floor=.23))
        primary, alternate = [1., 2.], [3., 4.]
        with patch("pour_guard_vision.locate", side_effect=[(primary, .7), (alternate, .3)]):
            self.assertEqual(locate_initial_source(None, checkpoint), (primary, .7, "primary"))
        with patch("pour_guard_vision.locate", side_effect=[(primary, .7), (alternate, .2)]):
            self.assertEqual(locate_initial_source(None, checkpoint), (primary, 0., "neck_unconfirmed"))
        with patch("pour_guard_vision.locate", side_effect=[(primary, .6), (alternate, .49)]):
            self.assertEqual(locate_initial_source(None, checkpoint), (primary, .6, "primary"))
        with patch("pour_guard_vision.locate", side_effect=[(primary, .6), (alternate, .57)]):
            point, score, method = locate_initial_source(None, checkpoint)
            self.assertEqual((point, method), (alternate, "neck_fallback"))
            self.assertAlmostEqual(score, .72)

    def test_short_empty_requires_consecutive_drain_evidence(self):
        threshold = dict(slow_empty_votes=24)
        self.assertIsNone(empty_confirmation(threshold, 40, 23, True, False, True))
        self.assertIsNone(empty_confirmation(threshold, 40, 24, False, False, True))
        self.assertEqual(empty_confirmation(threshold, 40, 24, True, False, True), "drain_dwell")
        self.assertEqual(empty_confirmation(threshold, 12, 0, False, False, False), "ordinary_visual")

    def test_late_drip_setback_applies_only_at_the_end_of_the_pour(self):
        for angle in (80., 116., 120., 122., 123.):
            self.assertEqual(late_mouth_offset(angle, .008, tail_setback_m=.008),
                             late_mouth_offset(angle, .008))
        self.assertAlmostEqual(late_mouth_offset(124.5, .008, tail_setback_m=.008), .004)
        self.assertAlmostEqual(late_mouth_offset(126., .008, tail_setback_m=.008), 0.)

    def test_midfill_rgb_calibration_keeps_full_and_low_readings(self):
        self.assertEqual(settled_rgb_weight(.64, MIDFILL_RGB_KNOTS), 0.)
        self.assertEqual(settled_rgb_weight(.725, MIDFILL_RGB_KNOTS), 1.)
        self.assertEqual(settled_rgb_weight(.925, MIDFILL_RGB_KNOTS), 0.)

    def test_settled_rgb_correction_is_monotone_and_keeps_full_readings(self):
        readings = [i / 1000 for i in range(1001)]
        corrected = [x - MAX_SETTLED_RGB_BIAS * settled_rgb_weight(x) for x in readings]
        self.assertTrue(all(a < b for a, b in zip(corrected, corrected[1:])))
        self.assertEqual(corrected[755], .755)
        self.assertEqual(corrected[925], .925)
        self.assertAlmostEqual(corrected[825], .790)

    def test_midfill_repair_allows_topup_without_confirming_full(self):
        rgb, color = .825, .781
        self.assertFalse(settled_pair([(rgb, color)] * 6, .925, .035, .01)[1])
        calibrated = rgb - .032 * settled_rgb_weight(rgb)
        self.assertEqual(settled_pair([(calibrated, color)] * 6, .925, .035, .01),
                         (color, True, False))

    def test_both_branches_have_to_confirm_full(self):
        self.assertEqual(settled_pair([[.94, .93]]*6, .925, .035, .01), (.93, True, True))
        self.assertEqual(settled_pair([[.94, .917]]*6, .925, .035, .01), (.917, True, False))
        self.assertFalse(settled_pair([[.98, .93]]*6, .925, .035, .01)[2])

    def test_missing_or_unsettled_samples_cannot_authorize_full(self):
        self.assertEqual(settled_pair([[.94, .94]]*5, .925, .035, .01), (None, False, False))
        self.assertFalse(settled_pair([[.97, .97], [.93, .93]]*3, .925, .035, .01)[2])
        self.assertEqual(settled_pair([[None, .94]]*6, .925, .035, .01), (None, False, False))

    def test_batch_rejects_changed_weights_and_teacher_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            weight = path/"weight"
            weight.write_text("frozen")
            scene = dict(name="SA_A_test_r0_00", policy_evaluation=True,
                         policy_family="visual_guard_hierarchy")
            manifest = dict(run_authorized=True, cases=[scene], files={str(weight):sha(weight)})
            batch = path/"batch.json"
            batch.write_text(json.dumps(manifest))
            self.assertEqual(batch_specs(batch),[scene])
            scene["expert_recovery"] = True
            batch.write_text(json.dumps(manifest))
            with self.assertRaises(AssertionError):
                batch_specs(batch)
            del scene["expert_recovery"]
            batch.write_text(json.dumps(manifest))
            weight.write_text("changed")
            with self.assertRaises(AssertionError):
                batch_specs(batch)


if __name__ == "__main__":
    unittest.main()
