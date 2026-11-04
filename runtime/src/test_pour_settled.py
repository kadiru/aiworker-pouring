"""Boundary and data-separation checks for the settled-height controller."""
import json
from pathlib import Path
import unittest
from pour_settled_model import assess,topup_cap,topup_feedback_eligible
from pour_settled_study import make_plan


class SettledTests(unittest.TestCase):
    def test_topup_reaches_last_pouring_angle_but_has_hard_cap(self):
        self.assertEqual(topup_cap(104.),104.5)
        self.assertEqual(topup_cap(94.),95.)
        self.assertEqual(topup_cap(130.),110.)

    def test_image_change_before_pouring_angle_cannot_end_topup(self):
        # Retained failed localization case braked at these two nonflow angles.
        cap=topup_cap(106.)
        for phase in (95.75,97.5,102.75,106.):
            self.assertFalse(topup_feedback_eligible(phase,cap))
        self.assertTrue(topup_feedback_eligible(106.25,cap))

    def test_high_instantaneous_surface_cannot_override_predicted_underfill(self):
        level, stable, full = assess([.895] * 6, .95, .915, .01, .10, 20)
        self.assertTrue(stable); self.assertFalse(full); self.assertLess(level, .90)

    def test_stable_settled_target_passes(self):
        self.assertTrue(assess([.923, .922, .924, .923, .922, .923], .95, .915, .01, .10, 20)[2])

    def test_unstable_missing_or_disagreeing_evidence_cannot_stop_full(self):
        for values, geometry, mass in (([.89, .99] * 3, .95, 20), ([.93] * 5, .95, 20),
                ([.93] * 6, .70, 20), ([.93] * 6, .95, 0), ([float('nan')] * 6, .95, 20)):
            self.assertFalse(assess(values, geometry, .915, .01, .10, mass)[2])

    def test_new_reserved_scenes_are_balanced_and_disjoint(self):
        p=Path('/workspace/learning/A-robust-20261103-v1/frozen-plan.json')
        if not p.exists():p=Path(__file__).resolve().parents[2]/'evidence/robust-A/frozen-plan.json'
        old=json.loads(p.read_text());new=make_plan(old)
        self.assertEqual(new,make_plan(old));self.assertEqual(len(new['test']),30)
        self.assertEqual(sum(x['expected_stop']=='bottle_empty' for x in new['test']),15)
        key=lambda x:(tuple(x['cup_world']),tuple(x['source_xy']),x['initial_ml'],x['cup_initial_ml'])
        self.assertFalse({key(x) for x in old['test']} & {key(x) for x in new['test']})
        for x in new['test']:
            self.assertEqual(x['vessel_appearance'],'A');self.assertEqual(x['spacing'],.0018)
            self.assertEqual(x['dt'],.0025);self.assertFalse(x['expert_recovery'])


if __name__=='__main__':unittest.main()
