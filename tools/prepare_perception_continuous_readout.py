"""Preserve numerical failure and declare continuous readout inputs."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    plan=read(ROOT/'plan.json');manifest=read(ROOT/'candidate.json');failed=read(ROOT/'cup-component.json')
    report=read(ROOT/'cup-level-validation.json')
    assert plan['version']==13 and plan['perception_round']==3
    assert not manifest['trained'] and not failed['passed'] and not report['numerical_check_passed']
    assert not (ROOT/'component-selection.json').exists() and not (ROOT/'final-selection.json').exists()
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert all(sha(Path(p))==d for p,d in manifest['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==d for n,d in manifest['runtime_sources'].items())
    archive=ROOT/'rounds/task-r3-readout-attempt2';archive.mkdir()
    for name in ('plan.json','candidate.json','cup-component.json','cup-level-validation.json','cup-level-attempt.pt',
        'cup-data-manifest.json','cup-replay-manifest.json','cup-localization-validation.json',
        'fit_perception_settled.py','fit_perception_cup.py','perception_task_labels.py','perception_fit_data.py'):
        shutil.copy2(ROOT/name,archive/name)
    (archive/'archive-manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in archive.iterdir()}),indent=2)+'\n')
    plan['version']=14;plan['stop_readout_fit']['residual_features']='first60_plus_continuous_spatial_shape'
    plan['readout_numerical_repair']='Remove hard quantile-bin features from the residual network. Preserve existing continuous RGB/CNN/position features, fixed widths/updates, train/validation splits and all accuracy/numerical limits. Maximum CPU/GPU feature discrepancy in removed quantile channels includes a one-row 1/95 jump.'
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest['protected_parent_files'].update({str(p):sha(p) for p in archive.iterdir()})
    manifest['plan_sha256']=sha(ROOT/'plan.json');(ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('cup-component.json','cup-level-validation.json','cup-level-attempt.pt','cup-localization-validation.json'):
        assert sha(ROOT/name)==sha(archive/name);(ROOT/name).unlink()
    report=dict(plan_sha256=manifest['plan_sha256'],archive=str(archive),helper_sha256=sha(Path(__file__)),
        settings=plan['stop_readout_fit'],scope=plan['readout_numerical_repair'])
    (ROOT/'continuous-readout-preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
