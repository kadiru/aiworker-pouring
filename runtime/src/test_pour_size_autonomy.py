import json
from pathlib import Path
import tempfile
import unittest

from pour_size_autonomy import MAX_SETTLED_RGB_BIAS, batch_specs, settled_pair, settled_rgb_weight, sha


class SizedStopTests(unittest.TestCase):
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
