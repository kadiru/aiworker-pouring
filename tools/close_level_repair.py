"""Verify the failed repair boundary without selecting models or consuming tests."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '/tmp')
import watch_wide_collection as w

ROOT = w.HOST_ROOT / 'learning/A-scale-dynamic-20261109-v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host(path):
    return w.HOST_ROOT / Path(path).relative_to('/workspace')


def main():
    witness = w.isolation()
    health = w.health()
    assert not health['queue'] and not (health.get('recording') or {}).get('active')
    assert not health.get('error') and not health.get('learning_error')
    candidate = read(ROOT / 'candidate.json')
    assert not candidate['autonomy_accepted']
    assert sha(ROOT / 'best.pt') == candidate['checkpoint_sha256']
    assert all(sha(host(n)) == v for n, v in candidate['protected_parent_files'].items())
    assert all(sha(w.HOST_ROOT / 'src' / n) == v for n, v in candidate['runtime_sources'].items())
    assert sha(ROOT / 'tracker-t1.pt') == 'f378b47db56a39dd085a7a78874c5b75e1819fdf666ca1357eb54d1a5239dcbe'
    assert sha(ROOT / 'dynamic_cup_tracker.py') == '2157808b4926f39d57e7e80e4490e016a2fd073962ed2331f2b6b2643c6da8ff'
    protocol = read(ROOT / 'protocol.json')
    original = read(ROOT / 'protocol-v3-before-level-repair.json')
    assert all(protocol[k] == original[k] for k in ('validation', 'holdout', 'gates'))
    assert all(s in protocol['train'] for s in original['train'])
    absent = ('component-selection.json', 'autonomy-selection.json', 'pair-r7-protocol.json',
              'pair-r7.json', 'level-cache-v3.json', 'level-cache-v3.npz')
    assert all(not (ROOT / n).exists() for n in absent)
    assert not list((w.HOST_ROOT / 'outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    processes = subprocess.check_output(['docker', 'top', w.CID, '-eo', 'pid,args'], text=True)
    assert not any('/workspace/learning/A-scale-dynamic-20261109-v1/' in line and '/isaac-sim' in line
                   for line in processes.splitlines())

    audit = read(ROOT / 'train-r2-audit.json')
    review = read(ROOT / 'train-r2-manual-review.json')
    admission = read(ROOT / 'static-repair-tracker-admission.json')
    assert audit['complete'] and len(audit['records']) == 36 and review['accepted']
    assert review['audit_sha256'] == sha(ROOT / 'train-r2-audit.json')
    assert admission['protocol_sha256'] == sha(ROOT / 'protocol.json') and not admission['passed']
    assert admission['tracker_sha256'] == sha(ROOT / 'tracker-t1.pt')
    records = {r['name']: r for r in audit['records']}
    specs = {s['name']: s for s in protocol['train']}
    assert set(records) == {r['name'] for r in admission['cases']}
    for record in records.values():
        assert sha(host(record['path'])) == record['sha256']
        assert record['recording_validation']['valid'] and record['label']['accepted']
        assert all(v == 0 for v in record['containment'].values())
    for batch in review['blocks']:
        index = batch['batch']
        manifest = ROOT / f'train-r2-early-review-b{index}.json'
        assert sha(manifest) == batch['manifest_sha256']
        assert sha(ROOT / f'train-r2-review-b{index}.json') == batch['review_sha256']
        assert all(sha(host(page['path'])) == page['sha256'] for page in read(manifest)['pages'])

    retired = read(ROOT / 'retired-moving-r1-audit.json')
    for record in retired['cases']:
        if 'directory' in record:
            for name, item in record['files'].items():
                assert sha(host(record['directory']) / name) == item['sha256']
    failures = []
    for case in admission['cases']:
        if not case['passed']:
            s = specs[case['name']]
            failures.append(dict(name=case['name'], cup_height_scale=s['cup_height_scale'],
                                 cup_diameter_scale=s['cup_diameter_scale'],
                                 measured_level=records[case['name']]['label']['target'],
                                 tracked_error_mm=case['error_mm'],
                                 raw_max_error_mm=case['raw_max_error_mm']))
    assert len(failures) == 6
    assert all(c['confirmed'] and len(c['frames']) == 11 and all(f['valid'] for f in c['frames'])
               for c in admission['cases'])
    for round_name in ('r5', 'r6'):
        assert not read(ROOT / f'pair-{round_name}.json')['passed']
        cfg = read(ROOT / f'pair-{round_name}-protocol.json')
        archive = ROOT / f'pair-{round_name}-frozen-inputs'
        assert all(sha(archive / n) == v for n, v in cfg['frozen_files'].items())
        for kind in ('rgb', 'color'):
            n = f'pair-{round_name}-{kind}.pt'
            assert sha(archive / n) == sha(ROOT / n)
    job = read(ROOT / 'static-repair-cache-v3/completion.json')
    assert job['exit_code'] == 1
    names = ('train-r2-audit.json', 'train-r2-manual-review.json', 'static-repair-tracker-admission.json',
             'pair-r5.json', 'pair-r6.json', 'retired-moving-r1-audit.json', 'level-repair-cpu-checks.json',
             'level-repair-r7-pipeline.log', 'run_repair_fit_pipeline.py',
             'static-repair-cache-v3/run.log', 'static-repair-cache-v3/completion.json')
    report = dict(
        status='repair_not_accepted_fixed_tracker_blocks_added_data',
        runtime=witness, health={k: health.get(k) for k in ('run', 'phase', 'queue', 'recording', 'error', 'learning_error')},
        active_checkpoint_sha256=candidate['checkpoint_sha256'],
        protected_parent_files_verified=len(candidate['protected_parent_files']),
        runtime_sources_verified=len(candidate['runtime_sources']),
        tracker_checkpoint_sha256=sha(ROOT / 'tracker-t1.pt'),
        tracker_module_sha256=sha(ROOT / 'dynamic_cup_tracker.py'),
        protocol_sha256=sha(ROOT / 'protocol.json'),
        original_validation_holdout_gates_unchanged=True,
        completed_fits=['r5', 'r6'], accepted_new_pairs=0,
        r7_fit_started=False, r7_skip_reason='All-case frozen-tracker admission failed before cache extension and fit preparation.',
        static_records=36, static_native_audit_passes=36, static_visual_frames_reviewed=144,
        static_measured_level_range=review['level_range'], static_measured_loss_zero=True,
        static_tracker_passes=30, static_tracker_failures=failures,
        all_396_static_tail_frames_tracker_valid=True,
        note='Failures are inaccurate stable anchors/raw detections, not a temporal validity rejection. No failed cases were silently dropped.',
        retired_moving_counts=retired['counts'],
        reserved_scenes_started=0, reserved_scenes_preserved=len(protocol['holdout']),
        autonomy_accepted=False, no_active_controller_change=True, component_jobs_running=False,
        disk_free_gib=shutil.disk_usage(w.HOST_ROOT).free / 1024**3,
        reports={n: sha(ROOT / n) for n in names},
        helper_sha256=sha(Path(__file__)),
        scope='Native calibration and supervised model repair only. No resized-bottle task training or eight-hour session launched. No foreign process signaled or reconfigured; snapshots cannot prove uninterrupted other workloads.')
    with (ROOT / 'level-repair-closure.json').open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'protected_parent_files_verified', 'runtime_sources_verified',
                                             'static_records', 'static_tracker_passes', 'reserved_scenes_started', 'disk_free_gib')}))


if __name__ == '__main__':
    main()
