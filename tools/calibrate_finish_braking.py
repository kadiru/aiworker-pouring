"""Apply an explicit two-scalar braking repair after reviewed development r1.

Native CPU operation while idle. Learned weights, runtime sources, goal and
timeout stay unchanged. Old failed outcomes remain archived and unchanged.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import torch

from calibrate_finish_target import ROOT, OUTPUTS, equal, read, sha
sys.path.insert(0, '/workspace/src')
from pour_robust_report import completed_row
from pour_wide_report import strict_success, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--withdrawal', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    args = parser.parse_args()
    assert not (ROOT / 'final-selection.json').exists()
    assert not list(OUTPUTS.glob('*/*FN30_A_test*/parameters.json'))
    with urllib.request.urlopen('http://192.168.68.60:8212/health', timeout=10) as response:
        health = json.load(response)
    assert not health['queue'] and not (health.get('recording') or {}).get('active')
    assert not health.get('error') and not health.get('learning_error')
    model = read(ROOT / 'model-summary.json')
    previous = sha(ROOT / 'best.pt')
    assert previous == model['checkpoint_sha256'] == 'f9ffbce349aa5b3a0720205f28c5fc51499a5a0b72762bc5470aa41b2930bb27'
    assert model['terminal_drain_hold_s'] == 0.
    assert model['thresholds']['coarse_brake_target'] == .87 and model['thresholds']['max_pauses'] == 10
    assert not (ROOT / 'development-resume-plan.json').exists()
    sources = model['runtime_sources']
    for name, expected in sources.items():
        assert sha(Path('/workspace/src') / name) == expected, name
    review = read(args.review)
    assert review['checkpoint_sha256'] == previous and review['manual_review'] == 'inspected'
    assert len(review['cases']) == 8 and all(r['manual_review'] == 'inspected' for r in review['cases'])
    expected = {r['name'] for r in review['cases']}
    assert all(n.endswith('_r1') for n in expected)
    integrity = read(ROOT / 'development-recording-integrity.json')
    assert integrity['native_checks_passed'] and not integrity['pending_attempts']
    indexed = {r['name']: r for r in integrity['files']}
    for row in review['cases']:
        assert row['source_sha256'] == indexed[row['name']]['sha256']
    motion = read(args.withdrawal)
    assert {r['name'] for r in motion['cases']} == expected
    for row in motion['cases']:
        assert row['source_sha256'] == indexed[row['name']]['sha256']
        assert row['post_release_gripper_contact_frames'] == 0
        assert row['longest_sampled_upright_near_stationary_over_cup_s'] <= .35
        starts = row['stage_starts_s']
        assert abs(starts['Setting bottle down'] - starts['Returning upright'] - 3.1) <= .051
        assert abs(row['final_tilt_deg']) < 8 and abs(row['final_base_height_m']) <= .005
    folders = {}
    for path in OUTPUTS.glob('*/*FN30_A_dev_*/parameters.json'):
        spec = read(path)
        if (path.parent / 'episode.hdf5').exists():
            assert spec['name'] not in folders
            folders[spec['name']] = path.parent
    probe = read(args.probe)
    assert probe['original_trigger_reconstruction_passed'] and len(probe['cases']) == 16
    assert sum(r['round'] == 0 for r in probe['cases']) == sum(r['round'] == 1 for r in probe['cases']) == 8
    assert {r['name'] for r in probe['cases'] if r['round'] == 1} == expected
    for row in probe['cases']:
        for name, digest in row['source_hashes'].items():
            assert sha(folders[row['name']] / name) == digest
        proposed = row['candidate_first_brakes'].get('0.82')
        if row['expected_stop'] == 'cup_near_full':
            assert proposed is not None and proposed['actual_rim_fraction'] < .90
            assert proposed['outside_nominal_ml'] == proposed['ever_table_nominal_ml'] == 0
            assert proposed['decision_t_s'] <= row['observed_first_brake']['decision_t_s']
        else:
            assert proposed is None, 'Inspect any newly introduced empty-case brake before applying'
    rows = []
    for name in sorted(expected):
        row = completed_row(folders[name], previous)
        failures = reasons(row)
        assert row['grasp_verified'] and row['visual_grasp_confirmed'] and row['sustained_grasp_fraction'] >= .8
        assert row['delivered_to_cup_ml'] >= 1 and row['completed_return_release_sequence'] and row['policy_completed']
        assert not row['teacher_fallback']
        if not strict_success(row):
            assert name == 'FN30_A_dev_05_r1'
            assert set(failures) <= {'level_unconfirmed', 'fill_unconfirmed', 'measured_spill', 'wrong_stop_condition', 'excess_outside'}
            assert row['cup_surface_height_fraction'] >= .9
        rows.append(dict(row, strict_pass=strict_success(row), failure_reasons=failures))
    assert sum(r['strict_pass'] for r in rows) == 7
    archive = ROOT / 'candidates/v23-r1'
    archive.mkdir(parents=True, exist_ok=False)
    (archive / 'src').mkdir()
    for name in sources:
        shutil.copy2(Path('/workspace/src') / name, archive / 'src' / name)
    for name in ('best.pt', 'model-summary.json', 'data-manifest.json', 'frozen-plan.json',
                 'development-recording-integrity.json', 'recording-repair.json', 'full-stop-calibration.json',
                 'infrastructure-aborts.json', 'development-resume-plan-r0.json'):
        shutil.copy2(ROOT / name, archive / name)
    for source, name in ((args.review, 'development-review.json'), (args.withdrawal, 'development-withdrawal.json'),
                         (args.probe, 'coarse-brake-window-probe.json')):
        shutil.copy2(source, archive / name)
    saved = torch.load(ROOT / 'best.pt', map_location='cpu', weights_only=False)
    original = copy.deepcopy(saved)
    changed = dict(coarse_brake_target=.82, max_pauses=13)
    old_values = {k:saved['thresholds'][k] for k in changed}
    assert old_values == dict(coarse_brake_target=.87, max_pauses=10)
    saved['thresholds'].update(changed)
    restored = copy.deepcopy(saved)
    restored['thresholds'].update(old_values)
    assert equal(original, restored), 'Only the two declared thresholds may change'
    assert not torch.cuda.is_initialized()
    torch.save(saved, ROOT / 'best.braking.partial.pt')
    assert equal(saved, torch.load(ROOT / 'best.braking.partial.pt', map_location='cpu', weights_only=False))
    checkpoint = sha(ROOT / 'best.braking.partial.pt')
    calibration = dict(calibrated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_checkpoint_sha256=previous, checkpoint_sha256=checkpoint, old_values=old_values, new_values=changed,
        changed_checkpoint_fields=['thresholds.' + k for k in changed], learned_weights_unchanged=True,
        runtime_sources_unchanged=True, full_stop_target_unchanged=saved['thresholds']['full_settled_target'],
        consumed_development=rows, coarse_window_probe_sha256=sha(args.probe),
        development_review_sha256=sha(args.review), development_withdrawal_sha256=sha(args.withdrawal),
        helper_sha256=sha(Path(__file__)), comparison_helper_sha256=sha(Path(__file__).with_name('calibrate_finish_target.py')),
        scope='Development-selected initial brake at 82% projected CNN level, then at most thirteen pause/top-up cycles. Full-stop target, 60-second limit, physics, motion, learned weights and scoring are unchanged. Native r2 remains required; no previous outcome is relabeled.')
    model = copy.deepcopy(model)
    model.update(checkpoint_sha256=checkpoint, parent_candidate_checkpoint_sha256=previous,
                 initial_pour_calibration=calibration)
    model['thresholds'].update(changed)
    for name, obj in (('initial-pour-calibration.json', calibration), ('model-summary.braking.partial.json', model)):
        (ROOT / name).write_text(json.dumps(obj, indent=2) + '\n')
    shutil.copy2(args.probe, ROOT / 'coarse-brake-window-probe.json')
    (ROOT / 'best.braking.partial.pt').replace(ROOT / 'best.pt')
    (ROOT / 'model-summary.braking.partial.json').replace(ROOT / 'model-summary.json')
    print(json.dumps(dict(checkpoint_sha256=checkpoint, changed_thresholds=changed,
                         learned_weights_unchanged=True, next_action='Explicitly queue development round 2')))


if __name__ == '__main__':
    main()
