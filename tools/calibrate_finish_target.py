"""Calibrate only the full-stop margin after reviewed, closed development.

Run explicitly inside the owned container while idle. No fitting, runtime
source changes, reserved evaluation, or relabeling of old outcomes occurs.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

import numpy as np
import torch

sys.path.insert(0, '/workspace/src')
from pour_robust_report import completed_row
from pour_wide_report import strict_success, reasons

ROOT = Path('/workspace/learning/A-finish-20261106-v1')
OUTPUTS = Path('/workspace/outputs/fountain')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--withdrawal', type=Path, required=True)
    parser.add_argument('--validation-margin', type=float, required=True)
    args = parser.parse_args()
    assert .002 <= args.validation_margin < .005
    assert not (ROOT / 'final-selection.json').exists()
    assert not list(OUTPUTS.glob('*/*FN30_A_test*/parameters.json'))
    with urllib.request.urlopen('http://192.168.68.60:8212/health', timeout=10) as response:
        health = json.load(response)
    assert not health['queue'] and not (health.get('recording') or {}).get('active')
    assert not health.get('error') and not health.get('learning_error')
    model = read(ROOT / 'model-summary.json')
    previous = sha(ROOT / 'best.pt')
    assert previous == model['checkpoint_sha256']
    assert model['terminal_drain_hold_s'] == 0.
    sources = model.get('runtime_sources', model['sources'])
    for name, expected in sources.items():
        assert sha(Path('/workspace/src') / name) == expected, name
    review = read(args.review)
    assert review['checkpoint_sha256'] == previous and review['manual_review'] == 'inspected'
    assert len(review['cases']) == 8 and all(c['manual_review'] == 'inspected' for c in review['cases'])
    integrity = read(ROOT / 'development-recording-integrity.json')
    assert integrity['native_checks_passed'] and not integrity['pending_attempts']
    indexed = {r['name']: r for r in integrity['files']}
    expected = {r['name'] for r in review['cases']}
    assert all(n.endswith('_r0') for n in expected)
    motion = read(args.withdrawal)
    assert {r['name'] for r in motion['cases']} == expected
    for row in motion['cases']:
        assert row['source_sha256'] == indexed[row['name']]['sha256']
        assert row['post_release_gripper_contact_frames'] == 0
        assert row['longest_sampled_upright_near_stationary_over_cup_s'] <= .35
    for row in review['cases']:
        assert row['source_sha256'] == indexed[row['name']]['sha256']
    over = max(.01, model['chosen']['metrics']['wide_near_overestimate'])
    target = .90 + over + args.validation_margin
    before = model['thresholds']['full_settled_target']
    assert before - .005 <= target < before
    rows = []
    probes = []
    for path in sorted(OUTPUTS.glob('*/*FN30_A_dev_*/parameters.json')):
        spec = read(path)
        if spec['name'] not in expected:
            continue
        row = completed_row(path.parent, previous)
        failures = reasons(row)
        assert row['zero_spill'] and row['completed_return_release_sequence']
        assert strict_success(row) or set(failures) <= {'fill_unconfirmed', 'wrong_stop_condition'}, failures
        rows.append(dict(row, strict_pass=strict_success(row), failure_reasons=failures))
        if strict_success(row):
            continue
        assert row['expected_stop'] == 'cup_near_full' and row['cup_surface_height_fraction'] >= .90
        decisions = read(path.parent / 'policy-decisions.json')
        settle = [d for d in decisions if d['mode'] == 'Pour control: settle'][-6:]
        assert len(settle) == 6
        levels = [d['settled_prediction'] for d in settle]
        q = float(np.quantile(levels, .25))
        geometry = float(np.median([d['surface_geometry_fraction'] for d in settle]))
        deviation = float(np.std(levels))
        assert q >= target and deviation < model['thresholds']['settled_stability_std']
        assert abs(geometry - q) < model['thresholds']['settled_geometry_tolerance']
        probes.append(dict(name=spec['name'], prediction_quartile=q, prediction_std=deviation,
            geometry_median=geometry, actual_final_rim_fraction=row['cup_surface_height_fraction'],
            scope='Recorded prediction window clears proposed threshold; future three-vote confirmation still requires native development.'))
    assert len(rows) == 8 and len(probes) > 0
    archive = ROOT / 'candidates/v23-r0'
    archive.mkdir(parents=True, exist_ok=False)
    (archive / 'src').mkdir()
    for name in sources:
        shutil.copy2(Path('/workspace/src') / name, archive / 'src' / name)
    for name in ('best.pt', 'model-summary.json', 'data-manifest.json', 'frozen-plan.json',
                 'development-recording-integrity.json', 'recording-repair.json',
                 'infrastructure-aborts.json', 'development-resume-plan.json'):
        shutil.copy2(ROOT / name, archive / name)
    shutil.copy2(args.review, archive / 'development-review.json')
    shutil.copy2(args.withdrawal, archive / 'development-withdrawal.json')
    saved = torch.load(ROOT / 'best.pt', map_location='cpu', weights_only=False)
    original = copy.deepcopy(saved)
    assert saved['thresholds']['full_settled_target'] == before
    saved['thresholds']['full_settled_target'] = target
    restored = copy.deepcopy(saved)
    restored['thresholds']['full_settled_target'] = before
    assert equal(original, restored), 'Any change beyond the declared scalar is forbidden'
    assert not torch.cuda.is_initialized(), 'Calibration is a CPU scalar update, not a new GPU fit'
    torch.save(saved, ROOT / 'best.calibrated.partial.pt')
    loaded = torch.load(ROOT / 'best.calibrated.partial.pt', map_location='cpu', weights_only=False)
    assert equal(saved, loaded)
    changed = sha(ROOT / 'best.calibrated.partial.pt')
    calibration = dict(calibrated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_checkpoint_sha256=previous, checkpoint_sha256=changed,
        old_target=before, new_target=target, consumed_validation_max_overestimate=over,
        validation_margin=args.validation_margin, changed_checkpoint_fields=['thresholds.full_settled_target'],
        learned_weights_unchanged=True, runtime_sources_unchanged=True,
        consumed_development=rows, recorded_window_probes=probes,
        development_review_sha256=sha(args.review), development_withdrawal_sha256=sha(args.withdrawal),
        helper_sha256=sha(Path(__file__)),
        scope='Development-selected scalar calibration. Validation margin is empirical, not a physical guarantee. Old outcomes are retained. All eight layouts must run again before reserved evaluation.')
    model = copy.deepcopy(model)
    model.update(checkpoint_sha256=changed, parent_candidate_checkpoint_sha256=previous,
                 target=target, full_stop_calibration=calibration, development_aliases={})
    model['thresholds']['full_settled_target'] = target
    for filename, obj in (('full-stop-calibration.json', calibration), ('model-summary.calibrated.partial.json', model)):
        (ROOT / filename).write_text(json.dumps(obj, indent=2) + '\n')
    (ROOT / 'best.calibrated.partial.pt').replace(ROOT / 'best.pt')
    (ROOT / 'model-summary.calibrated.partial.json').replace(ROOT / 'model-summary.json')
    (ROOT / 'development-resume-plan.json').rename(ROOT / 'development-resume-plan-r0.json')
    print(json.dumps(dict(checkpoint_sha256=changed, old_target=before, new_target=target,
                         learned_weights_unchanged=True, next_action='Explicitly queue development round 1')))


if __name__ == '__main__':
    main()
