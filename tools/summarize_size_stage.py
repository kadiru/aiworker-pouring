"""Close the size pilot from verified native records, without promoting a policy."""
import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
E = PROJECT / 'evidence/size-A'
DATA = PROJECT / 'data/material-A-size'
sys.path.insert(0, str(PROJECT / 'runtime/src'))
from pour_wide_report import strict_success, reasons


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = read(E / 'stage-results.json')
    candidate = read(E / 'candidate.json')
    plan = read(E / 'plan.json')
    backup = read(E / 'scalar-backup.json')
    review = read(E / 'visual-review-progress.json')
    probe = read(E / 'low-fill-temporal-perception-probe.json')
    diagnosis = read(E / 'recovery-diagnostics.json')
    parent = read(PROJECT / 'evidence/finish-A/final-benchmark.json')
    assert report['development_closed'] and not report['pending'] and not report['unstarted']
    assert len(report['integrity']) == 24 and len(report['development']) == 6
    assert report['parent_files_unchanged'] and report['learned_weights_and_original_thresholds_unchanged']
    assert not candidate['trained'] and not candidate['full_prediction_repair_complete']
    checkpoint = sha(E / 'best.pt')
    assert checkpoint == candidate['checkpoint_sha256'] == report['candidate_checkpoint_sha256'] == probe['checkpoint_sha256']
    assert sha(E / 'plan.json') == candidate['plan_sha256']
    assert sha(E / 'stage-results.json') == backup['report_sha256']
    for artifact, helper in (('stage-results.json', 'check_size_stage.py'),
                             ('recovery-keyframes.json', 'review_size_recovery.py'),
                             ('recovery-diagnostics.json', 'diagnose_size_recovery.py'),
                             ('localization-diagnostics.json', 'diagnose_size_localization.py'),
                             ('camera-resolution.json', 'measure_size_camera_resolution.py'),
                             ('low-fill-temporal-perception-probe.json', 'probe_size_perception.py')):
        assert read(E / artifact)['helper_sha256'] == sha(PROJECT / 'tools' / helper)
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    committed = subprocess.check_output(['git', 'archive', source_commit, 'runtime/src'], cwd=PROJECT)
    with tarfile.open(fileobj=io.BytesIO(committed)) as archive:
        source_hashes = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                         for m in archive.getmembers() if m.isfile()}
    for name, digest in candidate['runtime_sources'].items():
        assert sha(PROJECT / 'runtime/src' / name) == digest, name
        assert source_hashes['runtime/src/' + name] == digest, name
    for item in backup['entries']:
        p = DATA / item['relative_path']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
    indexed = {r['name']: r for r in report['integrity']}
    expected = {s['name']: s for part in plan.values() if isinstance(part, list) for s in part}
    assert set(indexed) == set(expected)
    previous = {r['name']: r for r in parent['cases']}
    diagnosed = {r['name']: r for r in diagnosis['cases']}
    assert len(diagnosed) == 6 and diagnosis['checkpoint_sha256'] == checkpoint
    rows = []
    for r in report['development']:
        identity = indexed[r['name']]
        diagnostic = diagnosed[r['name']]
        assert all(digest == identity['sidecars'][name] for name, digest in diagnostic['hashes'].items())
        assert diagnostic['strict_pass'] == r['strict_pass']
        path = Path(identity['path'])
        folder = DATA / path.parent.parent.name / path.parent.name
        spec = read(folder / 'parameters.json')
        assert all(spec[k] == v for k, v in expected[r['name']].items())
        samples = read(folder / 'samples.json')
        decisions = read(folder / 'policy-decisions.json')
        metrics = read(folder / 'metrics.json')
        last = samples[-1]
        assert metrics['bottle_result'] == last
        assert metrics['policy']['checkpoint_sha256'] == checkpoint
        assert r['cup_surface_height_fraction'] == last['cup_fill_fraction']
        assert r['source_remaining_ml'] == last['bottle_ml']
        quantum = spec['spacing'] ** 3 * 1e6
        baseline = round(spec['cup_initial_ml'] / quantum) * quantum
        assert r['delivered_to_cup_ml'] == max(0., last['cup_ml'] - baseline)
        assert r['stop_reason'] == last['stop_reason'] and r['expected_stop'] == spec['expected_stop']
        assert r['policy_completed'] == metrics['policy']['done']
        assert not r['teacher_fallback'] and not metrics['policy']['teacher_fallback']
        assert r['final_outside_ml'] == last['spilled_ml'] + last['airborne_ml']
        assert r['ever_below_worktop_ml'] == max(s['ever_table_ml'] for s in samples)
        assert r['zero_spill'] == (r['final_outside_ml'] == r['ever_below_worktop_ml'] == 0)
        tipping = [s for s in samples if abs(s['bottle_tilt_deg']) > 20]
        assert r['sustained_grasp_fraction'] == sum(s['both_fingers'] for s in tipping) / max(1, len(tipping))
        assert r['grasp_verified'] == any(s['grasp_verified'] for s in samples)
        assert r['visual_grasp_confirmed'] == any(d.get('visual_grasp_confirmed', False) for d in decisions)
        modes = ('Returning upright', 'Setting bottle down', 'Releasing bottle', 'Retreating', 'Holding completed pose')
        times = [next((d['t'] for d in decisions if d.get('stop_latched') and d['mode'] == m), None) for m in modes]
        ordered = all(t is not None for t in times) and all(a < b for a, b in zip(times, times[1:]))
        assert r['completed_return_release_sequence'] == bool(ordered and metrics['policy']['done'])
        assert r['strict_pass'] == strict_success(r) == (not reasons(r))
        before = previous[spec['former_test_case']]
        rows.append(dict(name=r['name'], former_test_case=spec['former_test_case'],
                         former_strict_pass=before['strict_pass'], strict_pass=r['strict_pass'],
                         expected_stop=r['expected_stop'], actual_stop=r['stop_reason'],
                         final_height=r['cup_surface_height_fraction'], source_remaining_nominal_ml=r['source_remaining_ml'],
                         loss_nominal_ml=max(r['final_outside_ml'], r['ever_below_worktop_ml']),
                         return_release_complete=r['completed_return_release_sequence'],
                         sampled_over_cup_dwell_s=diagnostic['withdrawal']['longest_sampled_upright_stationary_over_cup_s'],
                         post_release_gripper_contact_frames=diagnostic['withdrawal']['post_release_gripper_contact_frames'],
                         failure_reasons=r['failure_reasons']))
    inspected = []
    for group in review['calibration_groups']:
        assert group['manual_review'] == 'inspected' and sha(E / group['sheet']) == group['sheet_sha256']
        for frame in group['frames']:
            assert frame['source_sha256'] == indexed[frame['name']]['sha256']
            inspected.append((frame['name'], frame['camera'], frame['index']))
    for r in review['recovery']:
        assert r['manual_review'] == 'inspected' and sha(E / (r['name'] + '.jpg')) == r['sheet_sha256']
        assert r['source_sha256'] == indexed[r['name']]['sha256']
        for f in r['frames']:
            inspected.append((r['name'], f['camera'], f['index']))
    assert len(review['recovery']) == 6 and len(inspected) == 72
    snapshots = {n: read(E / (n + '.json')) for n in ('runtime-before-recovery', 'runtime-final')}
    before, after = snapshots.values()
    assert before['container'] == after['container'] and before['limits'] == after['limits']
    assert before['guard'] and after['guard']
    health = read(E / 'runtime-final-health.json')
    assert not health['queue'] and not health['recording']['active']
    assert not health.get('error') and not health.get('learning_error')
    summary = dict(
        scope='Completed stationary size pilot and six reused-layout development regressions. Not fresh held-out evaluation, not size-generalizing training.',
        checkpoint_sha256=checkpoint, parent_checkpoint_sha256=candidate['parent_checkpoint_sha256'],
        runtime_source_commit=source_commit,
        no_new_weight_fitting=True, resized_autonomy_ready=False,
        counts=dict(grid_contained=sum(r['stable_contained'] for r in report['calibration']), grid_cases=9,
                    seed_contained=sum(r['stable_contained'] for r in report['seed_calibration']), seed_cases=9,
                    low_fill_localizations=probe['confirmed_cups'], low_fill_views=9,
                    regressions=6, strict_passes=sum(r['strict_pass'] for r in rows),
                    zero_measured_loss=sum(r['loss_nominal_ml'] == 0 for r in rows),
                    return_release_sequences=sum(r['return_release_complete'] for r in rows)),
        cases=rows, native_recordings=24, verified_local_sidecars=len(backup['entries']),
        independently_recomputed_task_scores=6, selected_native_frames_reviewed=72,
        review_scope='Selected frames, not continuous playback. Original HDF5 recordings remain on Ubuntu.',
        resource_scope='Owned container, 4070 binding, caps and priority guard preserved. No foreign job signaled or reconfigured; snapshots cannot prove zero shared-host overhead.',
        artifacts={n: sha(E / n) for n in ('candidate.json', 'plan.json', 'stage-results.json', 'scalar-backup.json',
                                         'visual-review-progress.json', 'recovery-keyframes.json', 'low-fill-temporal-perception-probe.json',
                                         'localization-diagnostics.json', 'camera-resolution.json', 'recovery-diagnostics.json',
                                         'runtime-before-recovery.json', 'runtime-final.json', 'runtime-final-health.json')},
        helper_sha256=sha(Path(__file__)),
    )
    (E / 'stage-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (E / 'recovery-results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows({**r, 'failure_reasons': '; '.join(r['failure_reasons'])} for r in rows)
    print(json.dumps({k: v for k, v in summary.items() if k in ('counts', 'verified_local_sidecars', 'native_recordings')}))


if __name__ == '__main__':
    main()
