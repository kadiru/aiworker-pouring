"""Prepare an archived, opt-in late-drip candidate after reviewed r2.

Run only while the owned simulator is idle. Then deploy the two staged runtime
files and restart only that container before queuing r3. No weight fitting,
physics change, original-outcome relabeling, or reserved-test use occurs.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path
import shutil
import urllib.request
import torch
from calibrate_finish_target import ROOT, OUTPUTS, equal, read, sha
from pour_robust_report import completed_row
from pour_wide_report import strict_success, reasons


def main():
    p = argparse.ArgumentParser()
    for name in ('review', 'withdrawal', 'diagnostic', 'geometry', 'candidate-dir', 'source-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    previous = 'bc9dc0e2d0afd2d090ff36f283be2276a9090f5fc48d3bf8290291dfcf2ee81e'
    assert not (ROOT / 'final-selection.json').exists()
    assert not list(OUTPUTS.glob('*/*FN30_A_test*/parameters.json'))
    with urllib.request.urlopen('http://192.168.68.60:8212/health', timeout=10) as response:
        h = json.load(response)
    assert not h['queue'] and not (h.get('recording') or {}).get('active')
    assert not h.get('error') and not h.get('learning_error')
    model = read(ROOT / 'model-summary.json')
    assert sha(ROOT / 'best.pt') == model['checkpoint_sha256'] == previous
    assert model['terminal_drain_hold_s'] == 0.
    assert model['thresholds']['coarse_brake_target'] == .82 and model['thresholds']['max_pauses'] == 13
    assert 'late_mouth_shift_m' not in model
    for n, digest in model['runtime_sources'].items():
        assert sha(Path('/workspace/src') / n) == digest, n
    manifest = read(a.source_manifest)
    geometry = read(a.geometry)
    assert manifest['candidate_for_parent_checkpoint_sha256'] == previous
    assert geometry['parent_checkpoint_sha256'] == previous and geometry['complete']
    assert geometry['candidate_runtime_sources'] == manifest['files']
    assert geometry['late_mouth_shift_m'] == .008 and geometry['shift_angle_range_deg'] == [116., 122.]
    assert geometry['cuda_initialized'] is False
    assert geometry['minimum_carried_bottle_vertical_collision_margin_m'] > 0.
    assert geometry['helper_sha256'] == sha(a.candidate_dir / 'audit_finish_late_drip.py')
    plan = read(ROOT / 'frozen-plan.json')
    assert {r['name'] for r in geometry['cases']} == {r['name'] for r in plan['development'] + plan['test']}
    assert len(geometry['cases']) == 38 and all(r['ready'] for r in geometry['cases'])
    changed = {n for n, digest in manifest['files'].items() if digest != model['runtime_sources'][n]}
    assert changed == {'pour_guard_actor.py', 'pour_wide_motion.py'}
    for n in changed:
        assert sha(a.candidate_dir / n) == manifest['files'][n], n
    for helper in geometry['geometry_helper_sources'].values():
        assert sha(Path(helper['path'])) == helper['sha256']
    review, motion = read(a.review), read(a.withdrawal)
    assert review['checkpoint_sha256'] == previous and review['manual_review'] == 'inspected'
    expected = {s['name'] + '_r2' for s in plan['development']}
    assert len(review['cases']) == 8 and {r['name'] for r in review['cases']} == expected
    assert all(r['manual_review'] == 'inspected' for r in review['cases'])
    integrity = read(ROOT / 'development-recording-integrity.json')
    assert integrity['native_checks_passed'] and not integrity['pending_attempts']
    indexed = {r['name']: r for r in integrity['files']}
    for r in review['cases']:
        assert r['source_sha256'] == indexed[r['name']]['sha256']
    assert {r['name'] for r in motion['cases']} == expected
    for r in motion['cases']:
        assert r['source_sha256'] == indexed[r['name']]['sha256']
        assert r['post_release_gripper_contact_frames'] == 0
        assert r['longest_sampled_upright_near_stationary_over_cup_s'] <= .35
        starts = r['stage_starts_s']
        assert abs(starts['Setting bottle down'] - starts['Returning upright'] - 3.1) <= .051
        assert abs(r['final_tilt_deg']) < 8 and abs(r['final_base_height_m']) <= .005
    folders = {}
    for path in OUTPUTS.glob('*/*FN30_A_dev_*/parameters.json'):
        spec = read(path)
        if spec['name'] in expected:
            assert spec['name'] not in folders
            assert (path.parent / 'episode.hdf5').exists() and not (path.parent / 'episode.partial.hdf5').exists()
            folders[spec['name']] = path.parent
    assert set(folders) == expected
    rows = []
    for name in sorted(expected):
        r = completed_row(folders[name], previous)
        failed = reasons(r)
        assert r['grasp_verified'] and r['visual_grasp_confirmed'] and r['sustained_grasp_fraction'] >= .8
        assert r['delivered_to_cup_ml'] >= 1 and r['completed_return_release_sequence'] and r['policy_completed']
        assert not r['teacher_fallback']
        if not strict_success(r):
            assert name == 'FN30_A_dev_03_r2' and set(failed) == {'measured_spill'}
            assert r['source_remaining_ml'] == 0. and r['stop_reason'] == 'bottle_empty'
            assert abs(r['final_outside_ml'] - .011664) < 1e-10
            assert abs(r['ever_below_worktop_ml'] - .011664) < 1e-10
        rows.append(dict(r, strict_pass=strict_success(r), failure_reasons=failed))
    assert sum(r['strict_pass'] for r in rows) == 7
    diagnostic = read(a.diagnostic)
    assert diagnostic['name'] == 'FN30_A_dev_03_r2'
    folder = folders[diagnostic['name']]
    for n, digest in diagnostic['source_hashes'].items():
        assert sha(folder / n) == digest
    samples, decisions = read(folder / 'samples.json'), read(folder / 'policy-decisions.json')
    loss = next(i for i, r in enumerate(samples) if r['ever_table_ml'] > 0.)
    stop = next(i for i, r in enumerate(decisions) if r['stop_latched'])
    assert (loss, stop) == (diagnostic['first_loss_index'], diagnostic['first_stop_index'])
    assert loss < stop and decisions[loss]['mode'] == 'Pour control: coarse'
    assert decisions[loss]['pause_count'] == 0 and 124. < samples[loss]['bottle_tilt_deg'] < 127.

    archive = ROOT / 'candidates/v23-r2'
    archive.mkdir(parents=True, exist_ok=False)
    (archive / 'src').mkdir()
    for n in model['runtime_sources']:
        shutil.copy2(Path('/workspace/src') / n, archive / 'src' / n)
    for n in ('best.pt', 'model-summary.json', 'data-manifest.json', 'frozen-plan.json',
              'development-recording-integrity.json', 'recording-repair.json', 'full-stop-calibration.json',
              'initial-pour-calibration.json', 'coarse-brake-window-probe.json',
              'infrastructure-aborts.json', 'development-resume-plan-r0.json'):
        shutil.copy2(ROOT / n, archive / n)
    for path, name in ((a.review, 'development-review.json'), (a.withdrawal, 'development-withdrawal.json'),
                       (a.diagnostic, 'late-loss-diagnostic.json'), (a.geometry, 'candidate-late-drip-clearance.json')):
        shutil.copy2(path, archive / name)
    saved = torch.load(ROOT / 'best.pt', map_location='cpu', weights_only=False)
    original = copy.deepcopy(saved)
    assert 'late_mouth_shift_m' not in saved
    saved['late_mouth_shift_m'] = .008
    restored = copy.deepcopy(saved)
    restored.pop('late_mouth_shift_m')
    assert equal(original, restored) and not torch.cuda.is_initialized()
    torch.save(saved, ROOT / 'best.drip.partial.pt')
    assert equal(saved, torch.load(ROOT / 'best.drip.partial.pt', map_location='cpu', weights_only=False))
    checkpoint = sha(ROOT / 'best.drip.partial.pt')
    calibration = dict(calibrated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_checkpoint_sha256=previous, checkpoint_sha256=checkpoint, late_mouth_shift_m=.008,
        shift_angle_range_deg=[116., 122.], changed_checkpoint_fields=['late_mouth_shift_m'],
        learned_weights_unchanged=True, thresholds_unchanged=True, physics_unchanged=True,
        terminal_timing_unchanged=True, runtime_source_commit=manifest['source_commit'],
        previous_runtime_sources=model['runtime_sources'], runtime_sources=manifest['files'],
        consumed_development=rows, diagnostic_sha256=sha(a.diagnostic), geometry_sha256=sha(a.geometry),
        development_review_sha256=sha(a.review), development_withdrawal_sha256=sha(a.withdrawal),
        helper_sha256=sha(Path(__file__)), comparison_helper_sha256=sha(Path(__file__).with_name('calibrate_finish_target.py')),
        scope='Development-selected programmed lip alignment for late emptying. Geometry supports more cup margin but does not prove droplet behavior. Full-pour/top-up poses through 116 degrees and continuous 3.1-second return timing are unchanged. Native r3 remains required; no original outcome is relabeled.')
    updated = copy.deepcopy(model)
    updated.update(checkpoint_sha256=checkpoint, parent_candidate_checkpoint_sha256=previous,
                   late_mouth_shift_m=.008, late_drip_alignment=calibration, runtime_sources=manifest['files'])
    for n, obj in (('late-drip-calibration.json', calibration), ('model-summary.drip.partial.json', updated)):
        (ROOT / n).write_text(json.dumps(obj, indent=2) + '\n')
    shutil.copy2(a.diagnostic, ROOT / 'late-loss-diagnostic-r2.json')
    if a.geometry.resolve() != (ROOT / 'late-drip-clearance-r3.json').resolve():
        shutil.copy2(a.geometry, ROOT / 'late-drip-clearance-r3.json')
    for n in changed:
        shutil.copy2(a.candidate_dir / n, ROOT / 'staged' / n)
    release = copy.deepcopy(manifest)
    release.update(checkpoint_sha256=checkpoint, late_drip_calibration_sha256=sha(ROOT / 'late-drip-calibration.json'))
    release['changed_fitted_source_files'] = sorted(n for n, digest in model['sources'].items() if manifest['files'][n] != digest)
    (ROOT / 'runtime-source-manifest-drip-r3.json').write_text(json.dumps(release, indent=2) + '\n')
    (ROOT / 'best.drip.partial.pt').replace(ROOT / 'best.pt')
    (ROOT / 'model-summary.drip.partial.json').replace(ROOT / 'model-summary.json')
    print(json.dumps(dict(checkpoint_sha256=checkpoint, learned_weights_unchanged=True,
        next_action='Deploy the two staged runtime sources and restart only the owned container, then verify isolation/source identity before explicitly queuing r3.')))


if __name__ == '__main__':
    main()
