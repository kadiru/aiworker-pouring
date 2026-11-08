"""Declare a learned stop-level readout and preserve the rejected raw CNN fit."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    plan=read(ROOT/'plan.json');manifest=read(ROOT/'candidate.json');failed=read(ROOT/'cup-component.json')
    assert plan['version']==12 and plan['perception_round']==3
    assert not manifest['trained'] and not failed['passed'] and failed['localization_passed']
    assert not (ROOT/'component-selection.json').exists() and not (ROOT/'final-selection.json').exists()
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert all(sha(Path(p))==d for p,d in manifest['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==d for n,d in manifest['runtime_sources'].items())
    archive=ROOT/'rounds/task-r3-level-attempt2';archive.mkdir()
    for name in ('plan.json','candidate.json','cup-component.json','cup-level-validation.json','cup-level-attempt.pt',
        'cup-level-predictions.json','cup-data-manifest.json','cup-localization-validation.json',
        'fit_perception_cup.py','perception_task_labels.py','perception_fit_data.py','evaluate_perception_holdout.py','apply_perception_components.py'):
        shutil.copy2(ROOT/name,archive/name)
    (archive/'archive-manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in archive.iterdir()}),indent=2)+'\n')
    plan.update(version=13,level_prediction_kind='cnn_plus_supervised_stop_readout',
        stop_readout_fit=dict(ridge=[.1,1.,10.,100.],ridge_features=['first60','first7_plus_spatial_shape'],
            residual_features='first7_plus_color_quantiles_and_spatial_shape',residual_widths=[32,16],
            residual_steps=1200,task_weight=16.,seed=26110833),
        stop_readout_scope='Evaluate the final existing-runtime stop-level readout against unchanged level gates; retain raw CNN errors separately. Supervised training features are RGB-derived and image-localized XY only. No bias chosen by hand, threshold relaxation, changed split or recorded task score.')
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest['protected_parent_files'].update({str(p):sha(p) for p in archive.iterdir()})
    manifest['plan_sha256']=sha(ROOT/'plan.json');(ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('cup-component.json','cup-level-validation.json','cup-level-attempt.pt','cup-level-predictions.json','cup-localization-validation.json'):
        assert sha(ROOT/name)==sha(archive/name);(ROOT/name).unlink()
    report=dict(plan_sha256=manifest['plan_sha256'],archive=str(archive),helper_sha256=sha(Path(__file__)),
        settings=plan['stop_readout_fit'],scope=plan['stop_readout_scope'])
    (ROOT/'stop-readout-preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
