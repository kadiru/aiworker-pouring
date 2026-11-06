"""Synthetic boundary cases only; never treated as robot evaluation results."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime/src'))
from pour_wide_report import strict_success,reasons


class ScoringTests(unittest.TestCase):
    def valid(self):
        return dict(grasp_verified=True,visual_grasp_confirmed=True,sustained_grasp_fraction=.99,
            delivered_to_cup_ml=53.,policy_completed=True,completed_return_release_sequence=True,
            expected_stop='cup_near_full',stop_reason='cup_near_full',gates=[],cup_surface_height_fraction=.92,
            source_initial_ml=95.,source_remaining_ml=42.,final_outside_ml=0.,ever_below_worktop_ml=0.,teacher_fallback=False)

    def test_clean_stop_goal_passes(self):
        r=self.valid();self.assertTrue(strict_success(r));self.assertEqual(reasons(r),[])
        r.update(expected_stop='bottle_empty',stop_reason='bottle_empty',source_remaining_ml=0.,cup_surface_height_fraction=.55)
        self.assertTrue(strict_success(r));self.assertEqual(reasons(r),[])

    def test_zero_loss_does_not_credit_missing_task_steps(self):
        for changes in [dict(grasp_verified=False),dict(visual_grasp_confirmed=False),dict(delivered_to_cup_ml=0.),
                        dict(cup_surface_height_fraction=.8999),dict(completed_return_release_sequence=False),
                        dict(sustained_grasp_fraction=.799),dict(gates=['wrong_stop_condition']),dict(stop_reason='bottle_empty')]:
            with self.subTest(changes=changes):
                r=dict(self.valid(),**changes);self.assertFalse(strict_success(r));self.assertTrue(reasons(r))

    def test_even_one_particle_or_returned_table_spill_fails(self):
        for key in ('final_outside_ml','ever_below_worktop_ml'):
            r=self.valid();r[key]=.005832
            self.assertFalse(strict_success(r));self.assertIn('measured_spill',reasons(r))

if __name__=='__main__':unittest.main()
