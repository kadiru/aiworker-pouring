"""Audit recorded post-stop particle motion without changing task scores.

Run inside the owned container after the selected split's integrity audit.
Snapshots are sparse (normally 0.25 s); velocity/displacement comparisons
are diagnostics, not proof of a particular solver or readback failure.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path[:0] = ['/workspace/src', '/workspace']
from bottle_pour import BOTTLE, vessel_mask
from pour_perception_geometry import upper_surface
from pour_vessel_geometry import cup_profile
from robot_pour import FRAME

ROOT = Path('/workspace/learning/A-perception-20261108-v1')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    return dict(zip(('p05', 'median', 'p95'),
                    map(float, np.quantile(values, [.05, .5, .95]))))


def analyze(record):
    folder = Path(record['path']).parent
    for name, digest in record['sidecars'].items():
        assert sha(folder / name) == digest
    trajectory = folder / 'trajectory.npz'
    assert sha(trajectory) == record['trajectory_sha256']
    spec = read(folder / 'parameters.json')
    rows = read(folder / 'samples.json')
    decisions = read(folder / 'policy-decisions.json')
    assert len(rows) == len(decisions)
    stop = next((i for i, d in enumerate(decisions) if d['stop_latched']), None)
    report = dict(name=spec['name'], trajectory_sha256=sha(trajectory),
                  samples_sha256=record['sidecars']['samples.json'],
                  final_sample_height=rows[-1]['cup_fill_fraction'],
                  stop_time=None if stop is None else decisions[stop]['t'])
    if stop is None:
        return dict(report, status='no_stop', intervals=[])
    profile = cup_profile(spec)
    times = np.array([r['t'] for r in rows])
    snapshots = []
    with np.load(trajectory, allow_pickle=False) as arrays:
        for i, t in enumerate(arrays['times']):
            if t < decisions[stop]['t']:
                continue
            j = int(np.argmin(abs(times-t)))
            assert abs(times[j]-t) < .0002
            row = rows[j]
            p = arrays[f'positions_{i:03}'].astype(float)
            v = arrays[f'velocities_{i:03}'].astype(float)
            assert p.shape == v.shape == (row['count'], 3)
            assert np.isfinite(p).all() and np.isfinite(v).all()
            local = (p-spec['cup_world']) @ FRAME
            source_local = (p-row['bottle_pose_xyz']) @ Rotation.from_quat(
                row['bottle_quat_xyzw']).as_matrix()
            mask = vessel_mask(local, profile) & ~vessel_mask(source_local, BOTTLE)
            assert abs(mask.sum()*spec['spacing']**3*1e6-row['cup_ml']) < 1e-6
            measured = upper_surface(local[mask], profile, spec['spacing'])
            assert measured['valid']
            assert abs(measured['height_fraction']-row['cup_column_height_fraction']) < 1e-7
            snapshots.append(dict(t=float(t), row=row, p=p, v=v, mask=mask,
                                  mode=decisions[j]['mode']))
    intervals = []
    for a, b in zip(snapshots, snapshots[1:]):
        dt = b['t']-a['t']
        assert dt > 0 and a['p'].shape == b['p'].shape
        common = a['mask'] & b['mask']
        if not common.any():
            continue
        delta = b['p'][common]-a['p'][common]
        displacement_from_velocity = .5*(a['v'][common]+b['v'][common])*dt
        residual = np.linalg.norm(delta-displacement_from_velocity, axis=1)
        intervals.append(dict(start_s=a['t'], end_s=b['t'], end_mode=b['mode'],
            common_particle_count=int(common.sum()),
            changed_membership_count=int(np.count_nonzero(a['mask'] ^ b['mask'])),
            cup_inventory_change_nominal_ml=b['row']['cup_ml']-a['row']['cup_ml'],
            legacy_height_change=b['row']['cup_fill_fraction']-a['row']['cup_fill_fraction'],
            column_height_change=b['row']['cup_column_height_fraction']-a['row']['cup_column_height_fraction'],
            vertical_displacement_mm=summary(delta[:, 2]*1000),
            displacement_mm=summary(np.linalg.norm(delta, axis=1)*1000),
            initial_speed_mm_s=summary(np.linalg.norm(a['v'][common], axis=1)*1000),
            final_speed_mm_s=summary(np.linalg.norm(b['v'][common], axis=1)*1000),
            trapezoid_velocity_residual_mm=summary(residual*1000)))
    worst = min(intervals, key=lambda r: r['column_height_change'], default=None)
    return dict(report, status='analyzed', verified_snapshots=len(snapshots),
                largest_post_stop_column_drop=worst, intervals=intervals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=('development', 'test'), required=True)
    parser.add_argument('--round', type=int, required=True)
    args = parser.parse_args()
    output = ROOT / f'{args.part}-particle-motion-r{args.round}.json'
    assert not output.exists()
    audit = read(ROOT / 'recording-audit.json')
    outcomes = read(ROOT / (args.part+'-outcomes.json'))
    assert not audit['started_incomplete'] and outcomes['status'] == 'complete'
    indexed = {r['name']: r for r in audit['records']}
    records = []
    for result in outcomes['cases']:
        record = indexed[result['name']]
        assert record['sha256'] == result['source_sha256']
        if 'trajectory_sha256' not in record:
            records.append(dict(name=result['name'], status='trajectory_not_recorded'))
        else:
            records.append(analyze(record))
    report = dict(cases=records, audit_sha256=sha(ROOT/'recording-audit.json'),
                  outcomes_sha256=sha(ROOT/(args.part+'-outcomes.json')),
                  helper_sha256=sha(Path(__file__)),
                  scope='Read-only analysis of sparse native particle snapshots. Cup membership and spatial surface are independently recomputed and checked against samples. Largest post-stop column drop is selected diagnostically. No causal solver claim, fitting, control change or replacement of final-sample task scores.')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([dict(name=r['name'], status=r['status'],
                          largest_drop=r.get('largest_post_stop_column_drop')) for r in records]))


if __name__ == '__main__':
    main()
