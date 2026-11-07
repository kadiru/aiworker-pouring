"""Read-only scalar diagnosis of closed development pours; never opens active HDF5."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    p = (len(values) - 1) * q
    lo = int(p)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (p - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    assert args.name.startswith('FN30_A_dev_')
    paths = list((args.root / 'outputs/fountain').glob('*/' + '*' + args.name + '-*/parameters.json'))
    assert len(paths) == 1, paths
    folder = paths[0].parent
    assert (folder / 'episode.hdf5').exists() and not (folder / 'episode.partial.hdf5').exists()
    spec = json.loads(paths[0].read_text())
    assert spec['name'] == args.name
    decisions = json.loads((folder / 'policy-decisions.json').read_text())
    samples = json.loads((folder / 'samples.json').read_text())
    assert len(decisions) == len(samples)
    model = json.loads((args.root / 'learning/A-finish-20261106-v1/model-summary.json').read_text())
    metrics = json.loads((folder / 'metrics.json').read_text())
    assert metrics['policy']['checkpoint_sha256'] == model['checkpoint_sha256']
    pauses = []
    i = 0
    while i < len(decisions):
        if decisions[i]['mode'] != 'Pour control: settle':
            i += 1
            continue
        start = i
        while i < len(decisions) and decisions[i]['mode'] == 'Pour control: settle':
            i += 1
        end = i - 1
        window = decisions[max(start, end - 5):end + 1]
        levels = [r['settled_prediction'] for r in window]
        if any(v is None for v in levels):
            continue
        truth = samples[end]['cup_fill_fraction']
        q = quantile(levels, .25)
        pauses.append(dict(start_s=decisions[start]['t'], end_s=decisions[end]['t'],
            predicted_quartile=q, last_six_predictions=levels, prediction_std=statistics.pstdev(levels),
            geometry_median=statistics.median(r['surface_geometry_fraction'] for r in window),
            actual_rim_fraction=truth, quartile_error=q-truth,
            cup_nominal_ml=samples[end]['cup_ml'], pause_count=decisions[end]['pause_count'],
            next_mode=decisions[i]['mode'] if i < len(decisions) else None,
            next_stop=decisions[i]['stop_latched'] if i < len(decisions) else None))
    stop = next((r for r in decisions if r['stop_latched']), None)
    result = dict(name=args.name, checkpoint_sha256=model['checkpoint_sha256'],
        thresholds=model['thresholds'], pauses=pauses, first_stop=stop,
        final_sample=samples[-1],
        source_hashes={n:sha(folder/n) for n in ('parameters.json','metrics.json','samples.json','policy-decisions.json')},
        scope='Closed development scalar diagnosis. Last-six recorded predictions approximate the decision window; all original outcomes remain unchanged. Simulator levels are diagnostic labels, never control inputs.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
