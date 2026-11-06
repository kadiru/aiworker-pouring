"""Allow explicit evaluation of known full-stop limits without relabeling them."""


def validate_development_outcome(row, passed, failures, limitation_reason=None):
    if passed:
        assert not failures
        return
    assert limitation_reason and len(limitation_reason.strip()) >= 20, (row['name'], failures)
    assert row['expected_stop'] == 'cup_near_full'
    assert failures and set(failures) <= {
        'underfilled', 'fill_unconfirmed', 'level_unconfirmed', 'wrong_stop_condition'}
    # Explicit evaluation can retain conservative full-stop failures only.
    # It cannot excuse a missed grasp, spill, incomplete return or fallback.
    assert row['grasp_verified'] and row['visual_grasp_confirmed']
    assert row['sustained_grasp_fraction'] >= .8 and row['delivered_to_cup_ml'] >= 1
    assert row['policy_completed'] and row['completed_return_release_sequence']
    assert row['final_outside_ml'] == row['ever_below_worktop_ml'] == 0
    assert not row['teacher_fallback']
    assert set(row['gates']) <= {'wrong_stop_condition'}
