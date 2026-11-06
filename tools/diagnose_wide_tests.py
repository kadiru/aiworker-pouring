"""Read-only diagnostics after all thirty frozen tests have been audited.

These measurements explain the retained results; they never fit a model,
change a threshold, or replace the predeclared task score.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / 'evidence/wide-A'
DATA = PROJECT / 'data/material-A-wide'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = EVIDENCE / 'test-diagnostics.json'
    assert not destination.exists(), 'Preserve previous diagnostic output'
    report = read(EVIDENCE / 'final-benchmark.json')
    integrity = read(EVIDENCE / 'recording-integrity.json')
    assert report['status'] == 'complete' and len(report['cases']) == 30
    assert integrity['native_checks_passed'] and not integrity['pending_attempts']
    records = {r['name']: r for r in integrity['files']}
    sidecars = {r['relative_path']: r for r in integrity['sidecar_files']}
    cases = []
    for row in report['cases']:
        record = records[row['name']]
        folder = (DATA / record['relative_path']).parent
        content = {}
        for name in ('parameters.json', 'samples.json', 'policy-decisions.json'):
            path = folder / name
            identity = sidecars[str(path.relative_to(DATA))]
            assert path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256']
            content[name] = read(path)
        spec = content['parameters.json']
        samples = content['samples.json']
        decisions = content['policy-decisions.json']
        assert len(samples) == len(decisions)
        accepted = next((d for d in decisions if d.get('source_xy') is not None
                         and d.get('cup_xy') is not None), None)
        localization = None
        if accepted:
            sample = min(samples, key=lambda s: abs(s['t'] - accepted['t']))
            localization = dict(
                decision_time_s=accepted['t'], nearest_sample_time_s=sample['t'],
                source_estimated_xy=accepted['source_xy'],
                source_observed_xy=sample['bottle_pose_xyz'][:2],
                source_xy_error_mm=float(1000*np.linalg.norm(
                    np.array(accepted['source_xy']) - sample['bottle_pose_xyz'][:2])),
                cup_estimated_xy=accepted['cup_xy'], cup_authored_xy=spec['cup_world'][:2],
                cup_xy_error_vs_authored_mm=float(1000*np.linalg.norm(
                    np.array(accepted['cup_xy']) - spec['cup_world'][:2])),
            )
        latched = next((i for i, d in enumerate(decisions) if d['stop_latched']), None)
        stop = None
        if latched is not None:
            d = decisions[latched]
            window = decisions[max(0, latched-5):latched+1]
            predictions = [x['settled_prediction'] for x in window
                           if x.get('settled_prediction') is not None]
            stop = dict(time_s=d['t'], reason=d['stop_latched'], mode=d['mode'],
                pause_count=d['pause_count'], predicted_settled_fraction=d['settled_prediction'],
                prediction_q25_last_six=float(np.quantile(predictions, .25)) if predictions else None,
                sampled_current_rim_fraction=samples[latched]['cup_fill_fraction'],
                sampled_source_nominal_ml=samples[latched]['bottle_ml'])
        table_events = []
        previous = 0.
        for sample in samples:
            value = sample.get('ever_table_ml', 0.)
            if value > previous + 1e-12:
                table_events.append(dict(time_s=sample['t'], phase=sample['phase'],
                    increase_nominal_ml=value-previous, cumulative_nominal_ml=value,
                    bottle_tilt_deg=sample['bottle_tilt_deg'], contacts=sample['contacts']))
            previous = value
        after_release = [i for i, d in enumerate(decisions)
                         if d['mode'] in ('Retreating', 'Holding completed pose')]
        recontacts = [dict(time_s=samples[i]['t'], mode=decisions[i]['mode'],
                          contacts=[c for c in samples[i]['contacts'] if c.startswith('gripper_')])
                      for i in after_release if any(c.startswith('gripper_') for c in samples[i]['contacts'])]
        cases.append(dict(name=row['name'], strict_pass=row['strict_pass'],
            expected_stop=row['expected_stop'], swapped=spec['swapped'],
            failure_reasons=row['failure_reasons'], source_sha256=record['sha256'],
            localization=localization, first_stop=stop,
            final_rim_fraction=row['cup_surface_height_fraction'],
            final_source_nominal_ml=row['source_remaining_ml'],
            final_outside_nominal_ml=row['final_outside_ml'],
            ever_near_worktop_nominal_ml=row['ever_below_worktop_ml'],
            final_height_trace=[{k:s[k] for k in ('t', 'cup_fill_fraction', 'cup_ml', 'phase')}
                                for s in samples[-20:]],
            recorded_table_loss_events=table_events,
            post_release_recontacts=recontacts))
    located = [r['localization'] for r in cases if r['localization'] is not None]
    failures = {}
    for row in cases:
        for reason in row['failure_reasons']:
            failures.setdefault(reason, []).append(row['name'])
    groups = {}
    for swapped in (False, True):
        for goal in ('cup_near_full', 'bottle_empty'):
            rows = [r for r in cases if r['swapped'] == swapped and r['expected_stop'] == goal]
            groups[('swapped' if swapped else 'normal')+'/'+goal] = dict(
                attempts=len(rows), strict_passes=sum(r['strict_pass'] for r in rows))
    result = dict(
        source_report_sha256=sha(EVIDENCE/'final-benchmark.json'),
        source_integrity_sha256=sha(EVIDENCE/'recording-integrity.json'),
        localized_cases=len(located),
        maximum_source_xy_error_mm=max((x['source_xy_error_mm'] for x in located), default=None),
        maximum_cup_xy_error_vs_authored_mm=max((x['cup_xy_error_vs_authored_mm'] for x in located), default=None),
        failure_groups=failures, orientation_and_goal_groups=groups, cases=cases,
        scope=('Post-test diagnostics only; no fitting, threshold or score changes. '
               'Source error uses the nearest 20 Hz native pose to the first accepted localization; '
               'cup error uses the authored fixture center. Contact/loss timing is sampled evidence, '
               'not proof of the cause of a spill or of continuous collision-free motion.'))
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
