"""Read-only stop-gate diagnosis from verified completed-case scalar copies.

Does not refit, change criteria, retry tests, or claim a physical failure cause.
"""
import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath

import numpy as np
from finish_media_source import sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/material-A-finish'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous diagnostics'
    manifest, model = read(args.sources), read(args.model)
    for name in ('pour_guard_actor.py', 'pour_settled_model.py'):
        assert sha(ROOT / 'runtime/src' / name) == model['runtime_sources'][name]
    for identity in manifest['files']:
        rel = PurePosixPath(identity['relative_path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        path = DATA / str(rel)
        assert path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256']
    verified = {r['relative_path'] for r in manifest['files']}
    cases = []
    for identity in manifest['recording_identities']:
        rel = PurePosixPath(identity['relative_path']).parent
        assert all(str(rel / name) in verified for name in
                   ('parameters.json', 'metrics.json', 'samples.json', 'policy-decisions.json'))
        folder = DATA / str(rel)
        spec, metrics = read(folder / 'parameters.json'), read(folder / 'metrics.json')
        samples, decisions = read(folder / 'samples.json'), read(folder / 'policy-decisions.json')
        assert spec['name'] == identity['name'] and spec['protocol'] == 18
        assert metrics['policy']['checkpoint_sha256'] == model['checkpoint_sha256']
        assert len(samples) == len(decisions) and all(
            abs(s['t'] - d['t'] - .05) < .0002 for s, d in zip(samples, decisions))
        stop = next((i for i, d in enumerate(decisions) if d['stop_latched']), None)
        empty_start = (1 + max((i for i, s in enumerate(samples) if s['bottle_ml'] > 0),
                               default=-1)) if samples[-1]['bottle_ml'] == 0 else None
        empty_gate = None
        if empty_start is not None:
            # Each command sees the preceding state, so start one step after
            # the first sampled zero inventory and stop at the first stop command.
            indices = range(empty_start + 1, stop + 1 if stop is not None else len(decisions))
            eligible = [decisions[i] for i in indices]
            failed, streak, longest = Counter(), 0, 0
            for d in eligible:
                checks = dict(empty_probability=d['empty_probability'] >= model['thresholds']['empty_probability'],
                              source_present=d['source_present_probability'] >= .7,
                              tilt=d['estimated_tilt_from_joints'] >= 95.,
                              flow=d['visual_flow_fraction_s'] * 72.8 < 3.)
                failed.update(k for k, passed in checks.items() if not passed)
                streak = streak + 1 if all(checks.values()) else 0
                longest = max(longest, streak)
            def limits(key):
                return [min(d[key] for d in eligible), max(d[key] for d in eligible)] if eligible else None
            empty_gate = dict(first_final_zero_sample_s=samples[empty_start]['t'],
                end_of_record_s=samples[-1]['t'], decisions_while_known_empty=len(eligible),
                failed_normal_gates=dict(failed), normal_longest_qualifying_streak=longest,
                normal_required_streak=12, empty_probability_range=limits('empty_probability'),
                source_present_probability_range=limits('source_present_probability'),
                source_presence_deadband_count=sum(.05 <= d['source_present_probability'] < .7 for d in eligible),
                encoder_hold_count=sum(bool((d.get('empty_views') or {}).get('encoder_hold')) for d in eligible),
                slow_empty_max_votes=max((d['slow_empty_votes'] for d in eligible), default=0),
                final_cap_dwell_seconds=decisions[-1]['cap_dwell_seconds'])
        full_gate = None
        if stop is not None and decisions[stop]['stop_latched'] in ('fill_unconfirmed', 'level_unconfirmed'):
            window = decisions[stop-5:stop+1]
            assert len(window) == 6 and all(d['fill_mode'] == 'settle' and
                d['pause_count'] == decisions[stop]['pause_count'] for d in window)
            levels = [d['settled_prediction'] for d in window]
            full_gate = dict(command_time_s=decisions[stop]['t'], six_level_predictions=levels,
                q25_level=float(np.quantile(levels, .25)), std=float(np.std(levels)),
                median_geometry=float(np.median([d['surface_geometry_fraction'] for d in window])),
                required_level=model['thresholds']['full_settled_target'],
                below_required_level=bool(np.quantile(levels, .25) < model['thresholds']['full_settled_target']),
                pause_count=decisions[stop]['pause_count'], max_pauses=model['thresholds']['max_pauses'],
                next_sample_height=samples[stop]['cup_fill_fraction'],
                final_settled_height=samples[-1]['cup_fill_fraction'])
        tipping = [s for s in samples if abs(s['bottle_tilt_deg']) > 20.]
        cases.append(dict(name=identity['name'], recording_sha256=identity['sha256'],
            stop_reason=decisions[stop]['stop_latched'] if stop is not None else None,
            final_mode=decisions[-1]['mode'], sustained_two_finger_fraction=
                sum(s['both_fingers'] for s in tipping)/max(1, len(tipping)),
            final_source_ml=samples[-1]['bottle_ml'], final_rim_fraction=samples[-1]['cup_fill_fraction'],
            empty_gate=empty_gate, full_gate=full_gate))
    result = dict(sources_sha256=sha(args.sources), model_summary_sha256=sha(args.model),
        analysis_source_sha256=sha(Path(__file__)), checkpoint_sha256=model['checkpoint_sha256'], cases=cases,
        scope='Recorded-decision diagnosis only. Source presence below 0.7 blocks both empty-stop paths; '
              'presence below 0.05 is required for the separate lost-source abort. Encoder obstruction '
              'is not proof of two-finger contact. Full-stop quantile is reconstructed from six recorded '
              'estimates; a below-target value alone blocks fullness confirmation. No counterfactual '
              'outcome or physical cause of lost contact is established. Full benchmark audit is separate.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(cases, indent=2))


if __name__ == '__main__':
    main()
