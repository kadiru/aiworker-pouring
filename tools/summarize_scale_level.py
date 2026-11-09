"""Verify local evidence bindings and summarize the completed size-level study.

Does not collect, fit, change gates, activate a model or infer a manual review.
Native model/HDF5 hashes are attested by the guarded native completion reports;
these large artifacts remain external to Git.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'evidence/scale-A'


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def accuracy(result, gate):
    return len(result['sizes']) == 9 and all(
        row['mae'] < gate['level_validation_mae']
        and row['near_max'] < gate['level_validation_near_max_error']
        and row['near_overestimate'] < gate['level_validation_near_overestimate']
        for row in result['sizes'].values())


def main():
    old = read('plan-collection-r1.json')
    plan = read('plan-collection-r2.json')
    assert plan['collect'][:108] == old['collect']
    assert plan['holdout'] == old['holdout'] and plan['component_gates'] == old['component_gates']
    assert len(plan['collect']) == 144 and len(plan['holdout']) == 9
    audit = read('collect-r2-audit.json')
    review = read('collect-r2-manual-review.json')
    names = {s['name'] for s in plan['collect']}
    assert audit['complete'] and audit['labels_accepted'] == 144
    assert audit['plan_sha256'] == sha('plan-collection-r2.json')
    assert len(audit['records']) == 144 and {r['name'] for r in audit['records']} == names
    assert all(r['recording_validation']['valid'] and r['label']['accepted'] for r in audit['records'])
    assert review['accepted'] and review['audit_sha256'] == sha('collect-r2-audit.json')
    assert len(review['cases']) == 36 and len(review['previously_reviewed']) == 108
    assert set(review['cases']) | set(review['previously_reviewed']) == names
    progress = read('collect-r2-progress.json')
    assert progress['completed'] == progress['planned'] == 36
    assert {r['name'] for r in progress['cases']} == set(plan['collection_rounds']['r2'])
    zero_loss = sum(r['ever_table_ml'] == 0 and r['final_outside_ml'] == 0 for r in progress['cases'])
    loc = read('locator-revalidation-r10.json')
    level = read('level-r10.json')
    finish = read('level-pipeline-r10-completion.json')
    assert loc['passed'] and loc['revalidation_only'] and loc['learned_parameters'] == 0
    assert level['localization_checkpoint_sha256'] == loc['checkpoint_sha256']
    assert loc['checkpoint_sha256'] == 'a19e1ff32746e6541ecfad3e8b6e1ab88ab04a3d41a78c5fa274c98ea1717570'
    for report in (loc, level):
        assert report['plan_sha256'] == audit['plan_sha256']
        assert report['audit_sha256'] == sha('collect-r2-audit.json')
        assert report['manual_review_sha256'] == sha('collect-r2-manual-review.json')
    assert finish['reports']['static-level-fit.json'] == sha('level-r10.json')
    assert finish['reports']['cup-neural-localization.json'] == sha('locator-revalidation-r10.json')
    assert finish['active_controller_unchanged'] and not finish['autonomous_size_accepted']
    assert finish['queue_empty'] and finish['recording_inactive']
    assert level['parent_checkpoint_sha256'] == finish['checkpoint_sha256'] == audit['checkpoint_sha256']
    assert level['epochs'] == 240 and level['optimizer_steps'] == 5280
    assert level['counts']['train']['episodes'] == 126 and level['counts']['validation']['episodes'] == 18
    assert level['refit_protocol_sha256'] == sha('level-refit-r10-plan.json')
    evaluation = None
    test_zero_loss = None
    if level['passed']:
        assert all(accuracy(r, plan['component_gates']) for r in (
            level['selected']['training'], level['selected']['validation'], level['cpu_validation']))
        assert level['cpu_gpu_max_difference'] < 1e-4 and level['inference_path_max_difference'] < 1e-4
        assert finish['reserved_scenes_started'] == 9
        selection = read('component-selection.json')
        evaluation = read('static-holdout-evaluation.json')
        test_review = read('holdout-r0-manual-review.json')
        test_progress = read('holdout-r0-progress.json')
        test_audit = read('holdout-r0-audit.json')
        assert test_progress['completed'] == test_progress['planned'] == 9
        assert test_audit['complete'] and test_audit['labels_accepted'] == 9
        assert all(r['recording_validation']['valid'] for r in test_audit['records'])
        test_zero_loss = sum(r['ever_table_ml'] == 0 and r['final_outside_ml'] == 0 for r in test_progress['cases'])
        assert selection['all_size_component_gates_passed'] and not selection['model_activated']
        assert selection['components']['static-level-candidate.pt']['sha256'] == level['checkpoint_sha256']
        assert evaluation['component_selection_sha256'] == sha('component-selection.json')
        assert evaluation['audit_sha256'] == sha('holdout-r0-audit.json') == test_review['audit_sha256']
        assert test_review['accepted'] and len(test_review['cases']) == 9
        assert {r['name'] for r in evaluation['cases']} == {s['name'] for s in plan['holdout']}
        assert evaluation['metrics']['passed'] == sum(r['passed'] for r in evaluation['cases'])
        assert evaluation['passed'] == all(r['passed'] for r in evaluation['cases'])
        assert finish['independent_static_passed'] == evaluation['passed']
        assert not evaluation['autonomous_use_accepted']
    else:
        assert finish['stage'] == 'level_gate_failed' and finish['reserved_scenes_started'] == 0
    rejected = []
    for revision in ('r7', 'r8', 'r9'):
        report = read(f'level-{revision}.json')
        closure = read(f'level-pipeline-{revision}-completion.json')
        assert not report['passed'] and closure['reserved_scenes_started'] == 0
        assert closure['reports']['static-level-fit.json'] == sha(f'level-{revision}.json')
        rejected.append(dict(revision=revision,report_sha256=sha(f'level-{revision}.json')))
    result = dict(
        development_passed=level['passed'], new_training_recordings=36,
        new_recordings_zero_measured_loss=zero_loss, training_episodes=126, validation_episodes=18,
        known_size_combinations=9, method='Supervised size-aware RGB regression; no RL',
        trainable_parameters=level['newly_fitted_parameters'], optimizer_steps=level['optimizer_steps'],
        selected_epoch=level['selected']['epoch'], fit_seconds=level['fit_seconds'],
        validation=level['selected']['validation'], cpu_gpu_max_difference=level['cpu_gpu_max_difference'],
        reserved_scenes_started=finish['reserved_scenes_started'],
        independent_static_metrics=None if evaluation is None else evaluation['metrics'],
        independent_static_passed=None if evaluation is None else evaluation['passed'],
        independent_static_zero_measured_loss=test_zero_loss,
        active_controller_unchanged=True, autonomous_size_accepted=False,
        rejected_development_revisions=rejected, runtime=finish['runtime'],
        plan_sha256=audit['plan_sha256'], parent_checkpoint_sha256=finish['checkpoint_sha256'],
        locator_checkpoint_sha256=loc['checkpoint_sha256'], level_checkpoint_sha256=level['checkpoint_sha256'],
        audit_sha256=sha('collect-r2-audit.json'), manual_review_sha256=sha('collect-r2-manual-review.json'),
        completion_sha256=sha('level-pipeline-r10-completion.json'),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Static known-size calibration only. Development validation was reused. Reserved tests are near-full only, not autonomous pours; higher-fill, independent low-fill and dynamic/held-bottle checks remain.')
    (ROOT/'level-size-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('development_passed','reserved_scenes_started','independent_static_passed')}))


if __name__ == '__main__':
    main()
