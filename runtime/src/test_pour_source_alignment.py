"""Source confirmation must be supported by a visible bottle in that view."""
import unittest
from pour_source_alignment import combine_evidence


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


if __name__=='__main__':unittest.main()
