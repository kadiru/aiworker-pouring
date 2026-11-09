"""Verify a completed, manually reviewed reserved delivery; never runs simulation."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
E = PROJECT / 'evidence/side-return-A'
STEM = 'test-r1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = [f'SR_A_test_r0_{i:02d}' for i in range(30)]
    paths = {key: E / filename for key, filename in {
        'freeze': 'final-selection.json',
        'outcomes': STEM + '-outcomes.json',
        'audit': STEM + '-audit.json',
        'backup': 'scalar-backup-' + STEM + '.json',
        'manual': STEM + '-manual-review.json',
        'delivery': STEM + '-delivery.json',
        'runtime_before': STEM + '-runtime-before.json',
        'runtime_after': STEM + '-runtime-after.json',
    }.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print(json.dumps({'status': 'pending', 'missing': missing}))
        return
    d = {key: read(path) for key, path in paths.items()}
    frozen, report, audit, backup, manual = (d[k] for k in
                                            ('freeze', 'outcomes', 'audit', 'backup', 'manual'))
    assert report['status'] == 'complete' and report['part'] == 'test'
    cases = report['cases']
    assert [c['name'] for c in cases] == names
    assert report['counts']['completed'] == report['counts']['planned'] == 30
    for key, observed in {
        'accepted': sum(c['accepted'] for c in cases),
        'strict_passes': sum(c['strict_pass'] for c in cases),
        'zero_measured_loss': sum(c['zero_spill'] for c in cases),
        'side_returns': sum(c['side_return']['passed'] for c in cases),
        'stopping_margins': sum(c['stopping_margin_passed'] for c in cases),
        'completed_returns': sum(c['withdrawal']['passed'] for c in cases),
    }.items():
        assert report['counts'][key] == observed, key
    for obj in (report, d['runtime_before'], d['runtime_after']):
        for key in ('checkpoint_sha256', 'plan_sha256', 'runtime_sources'):
            assert obj[key] == frozen[key], key
    assert d['delivery']['counts'] == report['counts']
    assert not audit['started_incomplete'] and audit['protected_parent_files_unchanged']
    assert report['audit_sha256'] == backup['audit_sha256'] == manual['audit_sha256'] == sha(paths['audit'])
    records = {r['name']: r for r in audit['records']}
    assert len(records) == len(audit['records']) and set(names) <= records.keys()
    for c in cases:
        assert c['source_sha256'] == records[c['name']]['sha256']
    assert backup['verified_local_sidecars'] == len(backup['entries'])
    assert backup['recordings'] == len(records)
    for entry in backup['entries']:
        path = PROJECT / 'data/material-A-side-return' / entry['relative_path']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
    assert manual['manual_review'] == 'inspected' and manual['review_complete'] is True
    assert manual['checkpoint_sha256'] == frozen['checkpoint_sha256']
    assert [c['name'] for c in manual['cases']] == names
    for reviewed, c in zip(manual['cases'], cases):
        assert reviewed['manual_review'] == 'inspected'
        assert reviewed['source_sha256'] == c['source_sha256']
        assert reviewed['accepted'] == c['accepted']
        for native_path, digest in reviewed['reviewed_files'].items():
            assert sha(E / (STEM + '-native-review') / Path(native_path).name) == digest
    for path, digest in manual['encoded_contact_sheets_and_cards'].items():
        assert sha(Path(path)) == digest
    videos = []
    for batch in (1, 2, 3):
        stem = f'{STEM}-part{batch}'
        manifest_path, qa_path = E / (stem + '-video.json'), E / (stem + '-video-review.json')
        manifest, qa = read(manifest_path), read(qa_path)
        assert [c['name'] for c in manifest['clips']] == names[(batch-1)*10:batch*10]
        assert manifest['report_sha256'] == sha(paths['outcomes'])
        assert manifest['audit_sha256'] == sha(paths['audit'])
        assert manifest['checkpoint_sha256'] == frozen['checkpoint_sha256']
        assert sha(Path(manifest['video'])) == manifest['sha256'] == qa['video_sha256']
        assert qa['full_decode_passed'] and qa['all_clips_have_initial_middle_final']
        assert len(qa['camera_comparisons']) == 90 and len(qa['encoded_review_frames']) == 32
        assert qa['maximum_mean_absolute_pixel_error'] < 2.5
        assert manual['video_review_sha256'][stem] == sha(qa_path)
        assert manual['video_sha256'][stem] == manifest['sha256']
        for clip, c in zip(manifest['clips'], cases[(batch-1)*10:batch*10]):
            assert clip['source_sha256'] == c['source_sha256']
            assert clip['accepted'] == c['accepted']
            assert clip['playback_speed'] == (2 if c['accepted'] else 1)
        videos.append(dict(part=batch, path=manifest['video'], sha256=manifest['sha256'],
                           duration_s=manifest['duration_s'], manifest_sha256=sha(manifest_path),
                           review_sha256=sha(qa_path), max_pixel_error=qa['maximum_mean_absolute_pixel_error']))
    assert manual['selected_native_frames'] == sum(len(c['frames']) + len(c['return_frames']) for c in manual['cases'])
    assert manual['selected_encoded_frames'] == 96
    full = [c for c in cases if c['expected_stop'] == 'cup_near_full']
    empty = [c for c in cases if c['expected_stop'] == 'bottle_empty']
    assert len(full) == len(empty) == 15
    side = [c['side_return'] for c in cases if c['side_return']['planned']]
    result = dict(
        closed_utc=datetime.now(timezone.utc).isoformat(), checkpoint_sha256=frozen['checkpoint_sha256'],
        plan_sha256=frozen['plan_sha256'], counts=report['counts'],
        full_final_height_range=[min(c['cup_surface_height_fraction'] for c in full), max(c['cup_surface_height_fraction'] for c in full)],
        empty_max_source_nominal_ml=max(c['source_remaining_ml'] for c in empty),
        side_planned_cases=len(side),
        peak_sampled_height_in_side_planned_cases=max((s['peak_surface_height_fraction'] for s in side), default=None),
        side_counts=dict(Counter(c['side_return'].get('side', 'unplanned') for c in cases)),
        max_placement_error_mm=max((1000*s['final_placement_error_m'] for s in side), default=None),
        minimum_rotation_translation_overlap=min((s['simultaneous_translation_rotation_fraction'] for s in side), default=None),
        upright_over_cup_samples=sum(s['upright_over_cup_samples'] for s in side) if side else None,
        failures=[dict(name=c['name'], reasons=c['acceptance_failures'], final_outside_nominal_ml=c['final_outside_ml'],
                       ever_below_worktop_nominal_ml=c['ever_below_worktop_ml']) for c in cases if not c['accepted']],
        native_records=len(records), scalar_sidecars=backup['verified_local_sidecars'],
        selected_native_frames=manual['selected_native_frames'], selected_encoded_frames=96,
        videos=videos, input_sha256={key: sha(path) for key, path in paths.items()}, helper_sha256=sha(Path(__file__)),
        scope='Thirty once-only frozen simulation tests, all outcomes retained. Selected-frame manual review plus full video decoding, not continuous manual playback. No new model fitting or RL; existing supervised vision and motion BC with programmed control. Zero measured loss is sampled final outside / ever near table, not a physical-water guarantee.')
    destination = E / (STEM + '-closure.json')
    with destination.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(status='closed', counts=result['counts'], report=str(destination))))


if __name__ == '__main__':
    main()
