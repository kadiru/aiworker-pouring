"""Locate the first changed coarse-brake command in closed development logs.

This is recorded-window analysis, not simulation of counterfactual outcomes.
Never opens a partial HDF5 or changes a controller, recording, or score.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    rows = []
    for path in sorted((args.root / 'outputs/fountain').glob('*/*FN30_A_dev_*/parameters.json')):
        folder = path.parent
        if not (folder / 'episode.hdf5').exists():
            continue
        assert not (folder / 'episode.partial.hdf5').exists()
        spec = read(path)
        if spec['development_round'] not in (0, 1):
            continue
        decisions = read(folder / 'policy-decisions.json')
        samples = read(folder / 'samples.json')
        metrics = read(folder / 'metrics.json')
        assert len(decisions) == len(samples)
        filtered = None
        probes = {}
        timeline = []
        observed_brake = None
        for i, (decision, sample) in enumerate(zip(decisions, samples)):
            mode = decision['mode']
            if not mode.startswith('Pour control:'):
                if filtered is not None:
                    break
                continue
            cnn = decision['cup_predicted_fraction']
            filtered = cnn if filtered is None else .6 * filtered + .4 * cnn
            projected = filtered + .25 * decision['visual_flow_fraction_s']
            assert all(math.isfinite(v) for v in (cnn, filtered, projected))
            current = dict(index=i, decision_t_s=decision['t'], sample_t_s=sample['t'],
                mode=mode, cnn=cnn, filtered_cnn=filtered, projected_cnn=projected,
                settled_prediction=decision['settled_prediction'], phase_deg=decision['own_pour_phase_deg'],
                actual_rim_fraction=sample['cup_fill_fraction'], source_nominal_ml=sample['bottle_ml'],
                outside_nominal_ml=sample['spilled_ml'], ever_table_nominal_ml=sample.get('ever_table_ml', 0.))
            for target in (.87, .85, .82, .80):
                if str(target) not in probes and filtered > .60 and projected >= target:
                    probes[str(target)] = current
            if i % 10 == 0 or mode == 'Pour control: brake':
                timeline.append(current)
            if mode == 'Pour control: brake':
                observed_brake = current
                break
        if observed_brake is not None:
            # All r0/r1 candidates retain the original 0.87 coarse target.
            assert probes['0.87']['index'] == observed_brake['index'], spec['name']
        else:
            assert '0.87' not in probes
        rows.append(dict(name=spec['name'], round=spec['development_round'], expected_stop=spec['expected_stop'],
            checkpoint_sha256=metrics['policy']['checkpoint_sha256'],
            original_stop=metrics['bottle_result']['stop_reason'], original_spill=metrics['spill'],
            observed_first_brake=observed_brake, candidate_first_brakes=probes, original_coarse_timeline=timeline,
            source_hashes={n:sha(folder / n) for n in ('parameters.json', 'samples.json', 'policy-decisions.json', 'metrics.json')}))
    report = dict(cases=rows, candidates=[.87, .85, .82, .80],
        original_trigger_reconstruction_passed=True,
        scope='Closed r0/r1 development scalar traces. Finds the first changed brake command only; does not predict subsequent flow, top-ups, stop votes, spill or success. All original outcomes remain unchanged.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        candidate = row['candidate_first_brakes'].get('0.82')
        print(json.dumps(dict(name=row['name'], original_brake_s=None if row['observed_first_brake'] is None else row['observed_first_brake']['decision_t_s'],
            proposed_brake_s=None if candidate is None else candidate['decision_t_s'],
            recorded_height_at_proposed_command=None if candidate is None else candidate['actual_rim_fraction'],
            original_final_outside_ml=row['original_spill']['final_outside_ml'])))


if __name__ == '__main__':
    main()
