import unittest

from pour_source_watchdog import TallBottleEmptyGate


class TallBottleEmptyTests(unittest.TestCase):
    evidence = dict(held=True, level_ready=True, present=.99, wrist_empty=.94,
                    scene_empty=.99, tilt=121., flow_ml_s=.3)

    def test_needs_ten_consecutive_frames(self):
        gate = TallBottleEmptyGate()
        for _ in range(9):
            self.assertFalse(gate.update(**self.evidence))
        self.assertTrue(gate.update(**self.evidence))

    def test_loss_of_any_required_evidence_resets_votes(self):
        invalid = dict(held=False, level_ready=False, present=.69, wrist_empty=.919,
                       scene_empty=.979, tilt=119.9, flow_ml_s=1.)
        for key, value in invalid.items():
            with self.subTest(signal=key):
                gate = TallBottleEmptyGate()
                for _ in range(9):
                    gate.update(**self.evidence)
                self.assertFalse(gate.update(**dict(self.evidence, **{key:value})))
                self.assertEqual(gate.votes, 0)
                self.assertFalse(gate.update(**self.evidence))

    def test_nonfinite_flow_cannot_authorize_empty(self):
        gate = TallBottleEmptyGate()
        for _ in range(10):
            self.assertFalse(gate.update(**dict(self.evidence, flow_ml_s=float("nan"))))


if __name__ == "__main__":
    unittest.main()
