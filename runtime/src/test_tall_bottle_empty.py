import unittest

from pour_source_watchdog import TallBottleEmptyGate, TrainedEmptyGate


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

    def test_aligned_confirmation_needs_raw_support_and_consecutive_wrist_evidence(self):
        values = dict(self.evidence, wrist_empty=.91, scene_empty=.7, aligned_empty=.9)
        gate = TallBottleEmptyGate(use_aligned_scene=True)
        for _ in range(9):
            self.assertFalse(gate.update(**values))
        self.assertEqual(gate.scene_view, "cup_aligned")
        self.assertFalse(gate.update(**dict(values, scene_empty=.649)))
        self.assertEqual(gate.votes, 0)
        for _ in range(9):
            self.assertFalse(gate.update(**values))
        self.assertTrue(gate.update(**values))
        self.assertFalse(gate.update(**dict(values, aligned_empty=float("nan"))))
        self.assertFalse(TallBottleEmptyGate().update(**values))

    def test_trained_empty_requires_a_stable_nonfull_cup_and_held_bottle(self):
        gate = TrainedEmptyGate(dict(tilt=120., flow=1.25, probability=.9, votes=8))
        evidence = dict(held=True,present=.99,tilt=122.,flow_ml_s=.6,cup_ready=True,
                        upper_level=.78,probability=.95)
        for change in (dict(held=False),dict(cup_ready=False),dict(upper_level=.85),
                       dict(flow_ml_s=1.25),dict(probability=float("nan"))):
            for _ in range(7):
                self.assertFalse(gate.update(**evidence))
            self.assertFalse(gate.update(**dict(evidence,**change)))
            self.assertEqual(gate.votes,0)
        for _ in range(7):
            self.assertFalse(gate.update(**evidence))
        self.assertTrue(gate.update(**evidence))

    def test_wrist_and_cup_needs_its_own_fourteen_votes(self):
        values = dict(self.evidence, wrist_empty=.95, scene_empty=.1, aligned_empty=0.)
        gate = TallBottleEmptyGate(use_aligned_scene=True, use_wrist_and_cup=True)
        for _ in range(13):
            self.assertFalse(gate.update(**values))
        self.assertFalse(gate.update(**dict(values, level_ready=False)))
        for _ in range(13):
            self.assertFalse(gate.update(**values))
        self.assertTrue(gate.update(**values))
        self.assertEqual(gate.scene_view, "wrist_and_cup")
        self.assertEqual(gate.votes, 0)


if __name__ == "__main__":
    unittest.main()
