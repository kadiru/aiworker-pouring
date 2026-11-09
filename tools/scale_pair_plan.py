"""Validate and append only the declared low/full training supplement."""
import copy
from collections import Counter, defaultdict


def extend(plan, proposal):
    assert plan['version'] == 4 and len(plan['collect']) == 54 and len(plan['holdout']) == 9
    assert proposal['proposal_only'] and not proposal['queued']
    old = {s['name']: s for s in plan['collect']}
    cases = proposal['cases']
    assert len(cases) == 54 and len({s['name'] for s in cases}) == 54
    assert not {s['name'] for s in cases} & (set(old) | {s['name'] for s in plan['holdout']})
    allowed = {'name', 'calibration_initial_height', 'cup_initial_ml', 'proposed_pair_of'}
    for s in cases:
        source = old[s['proposed_pair_of']]
        assert s['study_split'] == source['study_split'] == 'train'
        assert s['calibration_only'] and not s['policy_evaluation']
        assert set(s) == set(source) | {'proposed_pair_of'}
        assert all(s[k] == source[k] for k in source if k not in allowed)
        assert s['calibration_initial_height'] in (.22, 1.)
        assert s['calibration_initial_height'] != source['calibration_initial_height']
        assert s['cup_initial_ml'] > 0
    result = copy.deepcopy(plan)
    result['version'] = 5
    result['collect'] += copy.deepcopy(cases)
    result['collection_rounds'] = {'r0': list(old), 'r1': [s['name'] for s in cases]}
    counts = Counter((s['study_split'], s['cup_height_scale'], s['cup_diameter_scale']) for s in result['collect'])
    fills = defaultdict(set)
    for s in result['collect']:
        if s['study_split'] == 'train':
            fills[(s['cup_height_scale'], s['cup_diameter_scale'], s['layout_reference'])].add(s['calibration_initial_height'])
    assert len(fills) == 36 and all({.22, 1.} <= values for values in fills.values())
    for h in (.9, 1., 1.1):
        for d in (.9, 1., 1.1):
            assert counts['train', h, d] == 10 and counts['validation', h, d] == 2
    assert result['holdout'] == plan['holdout'] and result['component_gates'] == plan['component_gates']
    result['scope'] = '108 authored static perception episodes: 90 train / 18 unchanged validation; missing low/full pairs added at existing training layouts. The nine unused static holdouts and accuracy gates are unchanged. No autonomous size acceptance.'
    result['refit_protocol'] = dict(round='r3', initialization='retained_parent_member_0',
        input_resolution=384, augmentation='integer_translation_and_brightness', epochs=240,
        optimizer='AdamW', learning_rate=3e-5, loss='focal_heatmap_plus_2_smoothL1_offsets',
        expected_training_images=1170, expected_validation_images=234, expected_updates=17760,
        sampling='Equal probability per size, then per episode',
        conditional_level_fit='Only after all-size locator accuracy and temporal confirmation pass',
        conditional_holdout='Only after locator and level gates pass and artifacts are frozen')
    return result
