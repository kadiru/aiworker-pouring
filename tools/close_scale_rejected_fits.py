"""Close three rejected static fits without consuming holdouts or changing runtime."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, '/tmp')
import watch_wide_collection as w

ROOT = w.HOST_ROOT / 'learning/A-scale-20261109-v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / 'static-fit-closure.json'
    assert not output.exists()
    m = read(ROOT / 'candidate.json')
    plan = read(ROOT / 'plan.json')
    runtime = w.isolation()
    health = w.health()
    assert not health['queue'] and not health['recording']['active'] and not health.get('error')
    assert not m['trained']
    assert sha(ROOT / 'best.pt') == m['checkpoint_sha256']
    assert sha(ROOT / 'plan.json') == m['plan_sha256']
    assert all(sha(w.HOST_ROOT / 'src' / n) == d for n, d in m['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT / Path(p).relative_to('/workspace')) == d
               for p, d in m['protected_parent_files'].items())
    for name in ('cup-neural-localization.pt', 'scope-selection.json', 'static-level-fit.json',
                 'component-selection.json', 'static-holdout-evaluation.json'):
        assert not (ROOT / name).exists(), name
    started = {read(p)['name'] for p in (w.HOST_ROOT / 'outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in plan['holdout']}
    rounds = []
    for name in ('r0', 'r1', 'r2'):
        directory = ROOT if name == 'r2' else ROOT / 'locator-rounds' / name
        report_path = directory / 'cup-neural-localization.json'
        report = read(report_path)
        completion_path = ROOT / ('locator-fit-' + name) / 'completion.json'
        completion = read(completion_path)
        assert completion['exit_code'] == 0 and completion['passed'] is False and report['passed'] is False
        assert completion['report_sha256'] == sha(report_path)
        assert report['attempt_checkpoint_sha256'] == sha(directory / 'cup-neural-localization-attempt.pt')
        assert report['source_sha256'] == sha(directory / 'fit_scale_locator.py')
        rounds.append(dict(round=name, passed=False, best_epoch=report['best_attempt']['epoch'],
            best_validation_max_error_mm=report['best_attempt']['max_error_mm'],
            epochs=report['epochs'], optimizer_steps=report['optimizer_steps'],
            input_resolution=report.get('input_resolution', 384), fit_seconds=report['fit_seconds'],
            report_path=str(report_path), report_sha256=sha(report_path), completion_sha256=sha(completion_path)))
    report = dict(completed_fits=3, accepted_locator=False, rounds=rounds,
        accepted_level=False, independent_holdouts_run=0, unused_holdouts=len(plan['holdout']),
        active_checkpoint_retained=True, checkpoint_sha256=m['checkpoint_sha256'],
        plan_sha256=m['plan_sha256'], runtime=runtime, queue_empty=True, recording_inactive=True,
        protected_parent_files_unchanged=True, runtime_sources_unchanged=True,
        collection_audit_sha256=sha(ROOT / 'collect-r0-audit.json'),
        camera_review_sha256=sha(ROOT / 'collect-r0-manual-review.json'),
        next_data_proposal_sha256=sha(ROOT / 'fill-pair-supplement-proposal.json'),
        next_action='Declare and collect the missing paired low/near-full training views; retain current validation and unused holdouts. No resized autonomy is accepted.',
        helper_sha256=sha(Path(__file__)))
    with output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    m.update(stage='static_localization_gated', completed_locator_fits=3, accepted_locator=False,
             static_fit_closure_sha256=sha(output), active_checkpoint_retained=True)
    temp = ROOT / 'candidate.tmp'
    temp.write_text(json.dumps(m, indent=2) + '\n')
    temp.replace(ROOT / 'candidate.json')
    print(json.dumps(dict(closed=True, accepted_locator=False, unused_holdouts=len(plan['holdout']),
                         active_checkpoint_retained=True, closure_sha256=sha(output))))


if __name__ == '__main__':
    main()
