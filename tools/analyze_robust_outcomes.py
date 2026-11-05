"""Analyze the frozen evaluation from checksum-verified local recordings.

This creates a failure review and candidate data manifest, never fits a model.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence/robust-A'
DATA = ROOT / 'data/material-A-robust'
NATIVE = Path('/workspace/outputs/fountain')


def read(path):
    return json.loads(path.read_text())


def empty_timing(row, samples, decisions, stop, thresholds):
    """Compare a saved visual decision with evaluator-only particle measurements."""
    def terminal_interval(limit):
        if samples[-1]['bottle_ml'] > limit:
            return None
        i = next((i+1 for i in range(len(samples)-1, -1, -1)
                  if samples[i]['bottle_ml'] > limit), 0)
        return samples[i]['t']
    allowance = max(.25, .01*row['source_initial_ml'])
    threshold_t, zero_t = terminal_interval(allowance), terminal_interval(0.)
    actual_stop = stop['t'] if stop and stop['stop_latched'] == 'bottle_empty' else None
    missed = zero_t is not None and stop is None
    after = [d for d in decisions if zero_t is not None and d['t'] >= zero_t-1e-6]
    cap = [d for d in after if d.get('cap_dwell_seconds') is not None and
           d['cap_dwell_seconds'] >= thresholds['empty_cap_dwell_seconds']]
    return dict(name=row['name'], strict_pass=row['surface_goal_zero_spill_success'],
                source_left_nominal_ml=row['source_remaining_ml'], allowance_nominal_ml=allowance,
                terminal_within_allowance_s=threshold_t, terminal_exact_zero_s=zero_t,
                actual_empty_stop_s=actual_stop, stop_reason=row['stop_reason'],
                empty_stop_minus_within_allowance_s=actual_stop-threshold_t
                    if actual_stop is not None and threshold_t is not None else None,
                empty_stop_minus_exact_zero_s=actual_stop-zero_t
                    if actual_stop is not None and zero_t is not None else None,
                observed_empty_without_any_stop_s=samples[-1]['t']-zero_t if missed else None,
                post_drain_decisions=len(after), post_cap_dwell_decisions=len(cap),
                max_post_drain_empty_score=max((d['empty_probability'] for d in after), default=None),
                max_post_cap_dwell_empty_score=max((d['empty_probability'] for d in cap), default=None),
                max_post_drain_slow_votes=max((d['slow_empty_votes'] for d in after), default=None))


def main(evidence=EVIDENCE, data=DATA):
    report = read(evidence / 'final-benchmark.json')
    assert report['status'] == 'complete' and len(report['cases']) == 30
    assert read(data / 'backup-index.json')['all_native_checksums_match']
    full, failures, transient, localization = [], [], [], []
    empty, contacts = [], []
    thresholds = read(evidence / 'model-summary.json')['thresholds'] if evidence.name == 'workspace-A' else {}
    for row in report['cases']:
        folder = data / Path(row['path']).relative_to(NATIVE)
        decisions = read(folder / 'policy-decisions.json')
        samples = read(folder / 'samples.json')
        stop = next((x for x in decisions if x['stop_latched']), None)
        if thresholds:
            tipping = [x for x in samples if abs(x['bottle_tilt_deg']) > 20.]
            both = sum(x['both_fingers'] for x in tipping)
            fraction = both / max(1, len(tipping))
            assert abs(fraction-row['sustained_grasp_fraction']) < 1e-12, row['name']
            stages = ('Returning upright', 'Setting bottle down', 'Releasing bottle',
                      'Retreating', 'Holding completed pose')
            stage_times = [next((d['t'] for d in decisions if d['stop_latched'] and
                                 d['mode'] == stage), None) for stage in stages]
            ordered = all(t is not None for t in stage_times) and all(
                x < y for x,y in zip(stage_times, stage_times[1:]))
            assert bool(row['policy_completed'] and ordered) == row['completed_return_release_sequence']
            contacts.append(dict(name=row['name'], bilateral_samples=both, tipping_samples=len(tipping),
                                 sustained_fraction=fraction, passes_contact_requirement=fraction>=.8,
                                 ordered_return_stage_times_s=dict(zip(stages, stage_times))))
        if thresholds and row['expected_stop'] == 'bottle_empty':
            empty.append(empty_timing(row, samples, decisions, stop, thresholds))
        if stop is not None:
            at = min(samples, key=lambda x: abs(x['t'] - stop['t']))
            assert abs(at['t'] - stop['t']) < .051
        if row['expected_stop'] == 'cup_near_full' and row['stop_reason'] == 'cup_near_full' and row['grasp_verified']:
            assert stop is not None
            full.append(dict(
                name=row['name'], success=row['surface_goal_zero_spill_success'],
                stop_t=stop['t'], measurement_t=at['t'],
                predicted_height_at_stop=stop['cup_predicted_fraction'],
                settled_prediction_at_stop=stop.get('settled_prediction'),
                settled_model_error_points=None if stop.get('settled_prediction') is None else 100 * (stop['settled_prediction'] - row['cup_surface_height_fraction']),
                geometric_estimate_at_stop=stop['surface_geometry_fraction'],
                measured_height_at_stop=at['cup_fill_fraction'],
                measured_final_height=row['cup_surface_height_fraction'],
                prediction_error_at_stop_points=100 * (
                    stop['cup_predicted_fraction'] - at['cup_fill_fraction']),
                height_drop_after_stop_points=100 * (
                    at['cup_fill_fraction'] - row['cup_surface_height_fraction']),
                cup_nominal_ml_at_stop=at['cup_ml'],
                cup_nominal_ml_final=row['cup_final_ml'],
                cup_change_after_stop_nominal_ml=row['cup_final_ml'] - at['cup_ml'],
                settled_prediction_error_points=100 * (
                    stop['cup_predicted_fraction'] - row['cup_surface_height_fraction']),
                relative_episode=str((folder / 'episode.hdf5').relative_to(data))))
        if not row['surface_goal_zero_spill_success']:
            causes = []
            if row['stop_reason'] == 'localization_unconfirmed':
                causes.append('visual_localization_rejected_before_grasp')
                localization.append(dict(name=row['name'],
                    relative_episode=str((folder / 'episode.hdf5').relative_to(data)),
                    first_observation_index=0,
                    note='Inspect saved RGB and replay the unchanged template localizer on CPU; no grasp or pour was attempted.'))
            if not row['grasp_verified']:
                causes.append('grasp_not_verified')
            elif row['sustained_grasp_fraction'] < .8:
                causes.append('grip_not_sustained')
            if row['delivered_to_cup_ml'] < 1:
                causes.append('insufficient_transfer')
            if not row['zero_spill']:
                causes.append('escaped_liquid')
            if not row['completed_return_release_sequence']:
                causes.append('incomplete_return_release')
            if stop is None and row['delivered_to_cup_ml'] >= 1:
                causes.append('no_stop_command_after_transfer')
            poured_full = row['expected_stop'] == 'cup_near_full' and row['delivered_to_cup_ml'] >= 1
            if poured_full and row['cup_surface_height_fraction'] < .9:
                causes.append('settled_cup_below_90_percent')
            if row['expected_stop'] == 'bottle_empty' and row['source_remaining_ml'] > max(.25, .01 * row['source_initial_ml']):
                causes.append('source_not_empty')
            if row['stop_reason'] != row['expected_stop']:
                causes.append('incorrect_stop_reason')
            failures.append(dict(name=row['name'], causes=causes,
                settled_height=row['cup_surface_height_fraction'],
                shortfall_percentage_points=max(0., 100 * (.9 - row['cup_surface_height_fraction']))
                    if poured_full else None))
        outside = [x for x in samples if x['spilled_ml'] > 0]
        if outside:
            transient.append(dict(name=row['name'], sample_count=len(outside),
                first_t=outside[0]['t'], last_t=outside[-1]['t'],
                peak_nominal_ml=max(x['spilled_ml'] for x in outside),
                final_outside_nominal_ml=row['final_outside_ml'],
                ever_below_worktop_nominal_ml=row['ever_below_worktop_ml']))
    assert sum(x['expected_stop'] == 'cup_near_full' for x in report['cases']) == 15
    if failures and len(localization)==len(failures):
        next_task=('Calibrate initial localization using multiple camera frames and positive/negative examples; '
                   'retain the frozen settled-level benchmark and reserve new scenes for any successor evaluation.')
    elif failures:
        next_task='Repair the measured remaining failures using these observed cases as development evidence, followed by new reserved evaluation scenes.'
    else:
        next_task='Retain this frozen result and reserve a new evaluation before expanding workspace or vessel appearance.'
    result = dict(
        checkpoint_sha256=report['checkpoint_sha256'],
        scene_plan_sha256=report['scene_plan_sha256'],
        analysis_performs_training=False, controller_changed_by_analysis=False,
        failures=failures, full_stop_measurements=full,
        localization_failure_candidates=localization,
        transient_outside_classifications=transient,
        runs_without_any_sampled_nonairborne_outside_particles=30 - len(transient),
        spill_definition='Declared success uses zero final outside particles and none ever below cup-frame z=3.5mm (the historical ever_below_worktop_ml field). Transient nonairborne outside classifications are reported separately; zero final spill does not prove no particle was ever outside.',
        inference='Compare the image estimate at the decision with the simultaneous particle surface, then compare that surface with the final settled surface. These measurements distinguish estimation error from subsequent surface settling; they do not prove a unique physical cause.',
        next_training=dict(
            status='post-evaluation candidates identified; this analysis performs no fitting or calibration',
            task=next_task,
            episodes=[x['relative_episode'] for x in full],
            target='Measured final settled height, with decision-to-final cup-volume change retained to avoid conflating settling with additional incoming liquid.',
            evaluation_rule='All30 cases are now observed development evidence for any successor model. Use new reserved seeds for its final evaluation; retain this frozen benchmark unchanged.'))
    if empty:
        assert len(empty) == 15
        assert len(contacts) == 30
        result['grip_and_return_measurements'] = contacts
        result['empty_stop_measurements'] = empty
        result['empty_timing_definition'] = ('Times are first saved samples of the final interval within the given limit. '
            'A negative stop-minus-limit value means the visual stop preceded that measured condition. '
            'A missed empty stop reports only the observed duration until the episode ended.')
    (evidence / 'failure-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(failures=failures,
        transient_outside_runs=len(transient),
        full_case_candidates=len(full), localization_candidates=len(localization),
        analysis_performs_training=False), indent=2))


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--study',choices=('robust','settled','reliable','workspace'),default='robust');a=p.parse_args()
    main(ROOT / f'evidence/{a.study}-A', ROOT / f'data/material-A-{a.study}')
