"""Source confirmation must be supported by a visible bottle in that view."""
import unittest
from pour_source_alignment import combine_evidence,held_source_presence


class SourceAlignmentTests(unittest.TestCase):
    def test_absent_aligned_bottle_cannot_supply_empty_evidence(self):
        empty,present=combine_evidence(.1,.95,.999,.2)
        self.assertEqual(empty,.1);self.assertEqual(present,.95)

    def test_second_visible_view_can_resolve_translation_failure(self):
        empty,present=combine_evidence(.47,.98,.99,.995)
        self.assertEqual(empty,.99);self.assertEqual(present,.995)

    def test_both_absent_views_cannot_confirm_empty(self):
        empty,present=combine_evidence(.999,.3,.999,.6)
        self.assertEqual(empty,0.);self.assertEqual(present,.6)

    def test_invalid_prediction_cannot_override_valid_view(self):
        self.assertEqual(combine_evidence(float('nan'),.99,.2,.9),(.2,.9))
        self.assertEqual(combine_evidence(1.1,.99,.2,.9),(.2,.9))

    def test_raw_view_preserves_held_bottle_during_early_tilt(self):
        self.assertEqual(held_source_presence(.916,.0066,True),.916)

    def test_visual_presence_cannot_override_lost_encoder_hold(self):
        self.assertEqual(held_source_presence(.999,.999,False),0.)

    def test_presence_ignores_invalid_view(self):
        self.assertEqual(held_source_presence(float('nan'),.9,True),.9)
        self.assertEqual(held_source_presence(2.,-1.,True),0.)


if __name__=='__main__':unittest.main()
