"""Read-only timing analysis of audited finish-study outcomes.

Does not change scores, fit, queue, retry, or mutate simulator data. Aggregate
particle counts locate observed loss in time; they do not establish its cause.
"""
import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import sys

from finish_media_source import sha

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / 'data/material-A-finish'
sys.path.insert(0, str(PROJECT / 'runtime/src'))
from pour_wide_report import strict_success, reasons


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--integrity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve earlier diagnostic reports'
    report = read(args.report)
    audit = read(args.integrity)
    assert audit['native_checks_passed'] and not audit['pending_attempts']
    rows = report.get('cases', report.get('rows'))
    assert rows and len({r['name'] for r in rows}) == len(rows)
    indexed = {r['name']: r for r in audit['files']}
    sidecars = {r['relative_path']: r for r in audit['sidecar_files']}
    cases = []
    for row in rows:
        identity = indexed[row['name']]
        relative = PurePosixPath(identity['relative_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.name == 'episode.hdf5' and identity['valid']
        folder = DATA / str(relative.parent)
        hashes = {}
        for name in ('parameters.json', 'metrics.json', 'samples.json', 'policy-decisions.json'):
            source = sidecars[str(relative.parent / name)]
            path = folder / name
            assert path.stat().st_size == source['bytes'] and sha(path) == source['sha256']
            hashes[name] = source['sha256']
        spec = read(folder / 'parameters.json')
        metrics = read(folder / 'metrics.json')
        samples = read(folder / 'samples.json')
        decisions = read(folder / 'policy-decisions.json')
        assert spec['protocol'] == 18 and spec['name'] == row['name']
        assert metrics['policy']['checkpoint_sha256'] == row['checkpoint_sha256']
        assert len(samples) == len(decisions) == identity['transitions']
        # A decision precedes its paired next-state sample by one control step.
        assert all(abs(s['t'] - d['t'] - .05) < .0002
                   for s, d in zip(samples, decisions))
        assert row['strict_pass'] == strict_success(row)
        assert row['failure_reasons'] == reasons(row)
        stop = next((i for i, d in enumerate(decisions) if d['stop_latched']), None)
        stop_time = decisions[stop]['t'] if stop is not None else None

        def event(i):
            if i is None:
                return None
            s, d = samples[i], decisions[i]
            return dict(index=i, sample_time_s=s['t'], preceding_command_time_s=d['t'],
                        mode=d['mode'], fill_mode=d.get('fill_mode'),
                        stop_latched=d['stop_latched'],
                        seconds_relative_to_first_stop_command=(
                            s['t'] - stop_time if stop_time is not None else None),
                        bottle_tilt_deg=s['bottle_tilt_deg'], source_remaining_ml=s['bottle_ml'],
                        cup_surface_height_fraction=s['cup_fill_fraction'],
                        outside_nonairborne_ml=s['spilled_ml'], airborne_ml=s['airborne_ml'],
                        ever_near_worktop_ml=s.get('ever_table_ml', 0.),
                        settled_prediction=d.get('settled_prediction'),
                        coarse_prediction=d.get('cup_predicted_fraction'),
                        surface_geometry_fraction=d.get('surface_geometry_fraction'))

        table = next((i for i, s in enumerate(samples) if s.get('ever_table_ml', 0.) > 0), None)
        outside = next((i for i, s in enumerate(samples) if s['spilled_ml'] > 0), None)
        stages = ('Returning upright', 'Setting bottle down', 'Releasing bottle',
                  'Retreating', 'Holding completed pose')
        starts = {stage: next((d['t'] for d in decisions
                              if d['stop_latched'] and d['mode'] == stage), None)
                  for stage in stages}
        final = samples[-1]
        # Use the final uninterrupted zero-inventory stretch, so a transient
        # empty reading followed by liquid returning to the bottle is excluded.
        empty_start = (1 + max((i for i, s in enumerate(samples) if s['bottle_ml'] > 0),
                               default=-1)) if final['bottle_ml'] == 0 else None
        empty_to_stop = (stop_time - samples[empty_start]['t']
                         if stop_time is not None and empty_start is not None else None)
        assert row['cup_surface_height_fraction'] == final['cup_fill_fraction']
        assert row['source_remaining_ml'] == final['bottle_ml']
        assert row['final_outside_ml'] == final['spilled_ml'] + final['airborne_ml']
        assert row['ever_below_worktop_ml'] == max(s.get('ever_table_ml', 0.) for s in samples)
        cases.append(dict(name=row['name'], expected_stop=row['expected_stop'],
                          strict_pass=row['strict_pass'], failure_reasons=row['failure_reasons'],
                          stop_reason=row['stop_reason'], zero_spill=row['zero_spill'],
                          final_rim_fraction=row['cup_surface_height_fraction'],
                          final_source_ml=row['source_remaining_ml'],
                          final_outside_ml=row['final_outside_ml'],
                          ever_near_worktop_ml=row['ever_below_worktop_ml'],
                          first_stop=event(stop), first_near_worktop_loss=event(table),
                          first_nonairborne_outside_reading=event(outside),
                          final_zero_source_stretch_start=event(empty_start),
                          zero_source_to_first_stop_command_s=empty_to_stop,
                          terminal_stage_command_times_s=starts,
                          source_recording_sha256=identity['sha256'], sidecar_sha256=hashes))
    failures = [r for r in cases if not r['strict_pass']]
    result = dict(report_sha256=sha(args.report), integrity_sha256=sha(args.integrity),
                  analysis_source_sha256=sha(Path(__file__)), attempts=len(cases),
                  strict_passes=sum(r['strict_pass'] for r in cases),
                  zero_measured_loss=sum(r['zero_spill'] for r in cases),
                  failure_reason_counts=dict(Counter(x for r in failures for x in r['failure_reasons'])),
                  cases=cases,
                  scope='Read-only observations from byte/SHA-verified scalar records. Scores are unchanged. '
                        'A decision is paired with the next state 50 ms later. First near-worktop loss '
                        'is the first sampled positive cumulative count, not a continuous contact time. '
                        'The first outside reading may be a transient droplet that returns. No individual '
                        'particle trajectories are recorded, so timing alone does not establish causation. '
                        'Zero-source delay starts at the final uninterrupted sampled zero-inventory stretch; '
                        'it is a diagnostic, not a new pass/fail rule. Negative values mean the stop command '
                        'preceded complete emptying, which may satisfy the declared near-empty tolerance. '
                        'Volumes are nominal particle-count mL, not calibrated physical volume.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('attempts', 'strict_passes', 'zero_measured_loss',
                                           'failure_reason_counts')}))


if __name__ == '__main__':
    main()
