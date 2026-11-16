import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from pour_guard_actor import empty_confirmation
from pour_guard_vision import locate, locate_initial_source
from pour_size_autonomy import (MAX_SETTLED_RGB_BIAS, MIDFILL_RGB_KNOTS, SizedCupPerception,
                                batch_specs, calibrated_rgb_height, pulse_limit_level, settled_pair, settled_rgb_weight,
                                sha, validate_rgb_calibration)
from pour_wide_motion import late_mouth_offset


class SizedStopTests(unittest.TestCase):
    def test_secondary_neck_model_runs_only_when_primary_heatmap_rejects(self):
        point = np.array([175.,140.])
        saved = dict(thresholds={"source_localization_score":.65},source_bank={},
                     source_heatmap=dict(proposal={"name":"incumbent"},refinement={},
                                         proposal_floor=.70,refinement_floor=.71,max_refine_px=6.),
                     source_heatmap_fallback=dict(proposal={"name":"fallback"},refinement={},
                                                  proposal_floor=.65,refinement_floor=.68,max_refine_px=6.))
        with patch("pour_cup_locator.locate_cup", return_value=(point,.9)) as propose, \
                patch("pour_guard_vision.locate", return_value=(point,.8)):
            self.assertEqual(locate_initial_source(None,saved)[2],"neck_heatmap")
            propose.assert_called_once()
        with patch("pour_cup_locator.locate_cup", side_effect=[(point,.6),(point,.9)]) as propose, \
                patch("pour_guard_vision.locate", return_value=(point,.8)):
            self.assertEqual(locate_initial_source(None,saved)[2],"neck_heatmap_fallback")
            self.assertEqual(propose.call_count,2)

    def test_pulse_limit_requires_three_stable_agreeing_windows(self):
        pair = (.924,.933)
        self.assertIsNone(pulse_limit_level([pair]*7,.035,.012))
        self.assertEqual(pulse_limit_level([pair]*8,.035,.012),.924)
        self.assertIsNone(pulse_limit_level([(.924,.97)]+[pair]*7,.035,.012))
        self.assertIsNone(pulse_limit_level([pair]*7+[(None,None)],.035,.012))

    def test_offscreen_neck_proposal_returns_zero_confidence(self):
        bank = dict(kernels=torch.zeros((1,2,64,48)), mask=torch.ones((1,2,64,48)),
                    offset=(-24,-12), bounds=(24,24,360,360))
        point, score = locate(np.zeros((384,384,3),dtype=np.uint8), bank, device="cpu",
                              expected_uv=[1.,1.])
        self.assertEqual(point.tolist(),[1.,1.])
        self.assertEqual(score,0.)

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

    def test_narrow_presence_check_preserves_a_wide_primary_detection(self):
        checkpoint = dict(source_bank={}, thresholds=dict(source_localization_score=.65),
                          source_initial_fallback=dict(bank=dict(bounds=[90, 120, 170, 170]),
                                                       threshold=.5, presence_floor=.33,
                                                       presence_bounds_only=True))
        outside, inside = [210., 145.], [120., 135.]
        with patch("pour_guard_vision.locate", side_effect=[(outside, .71), (inside, .21)]):
            self.assertEqual(locate_initial_source(None, checkpoint), (outside, .71, "primary"))
        with patch("pour_guard_vision.locate", side_effect=[(inside, .71), (inside, .21)]):
            self.assertEqual(locate_initial_source(None, checkpoint), (inside, 0., "neck_unconfirmed"))

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

    def test_bounded_rgb_height_correction_preserves_high_level_guard(self):
        geometry = {"rim_z_m": .0342}
        calibration = dict(geometry=geometry,knots=[.739,.787,.850,.9],bias=.031)
        validate_rgb_calibration(calibration,geometry)
        for level in (.7,.9,.925):
            self.assertEqual(calibrated_rgb_height(level,calibration),level)
        self.assertAlmostEqual(calibrated_rgb_height(.815,calibration),.784)
        with self.assertRaises(ValueError):
            validate_rgb_calibration(dict(calibration,knots=[.76,.78,.85,.9]),geometry)

    def test_tall_tail_moves_before_the_observed_rebound(self):
        config = dict(tail_setback_m=.009,tail_window_deg=(115.,119.))
        self.assertEqual(late_mouth_offset(114.,.008,**config),0.)
        self.assertLess(late_mouth_offset(119.,.008,**config),0.)
        self.assertAlmostEqual(late_mouth_offset(122.,.008,**config),-.001)

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
