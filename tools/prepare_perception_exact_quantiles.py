"""Preserve the continuous-feature failure and stage exact-count quantiles."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    plan=read(ROOT/'plan.json');manifest=read(ROOT/'candidate.json');failed=read(ROOT/'cup-component.json')
    assert plan['version']==14 and plan['perception_round']==3
    assert not manifest['trained'] and not failed['passed']
    assert not (ROOT/'component-selection.json').exists() and not (ROOT/'final-selection.json').exists()
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert all(sha(Path(p))==d for p,d in manifest['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==d for n,d in manifest['runtime_sources'].items())
    archive=ROOT/'rounds/task-r3-readout-attempt3';archive.mkdir()
    for name in ('plan.json','candidate.json','cup-component.json','cup-level-validation.json','cup-level-attempt.pt',
        'cup-data-manifest.json','cup-replay-manifest.json','cup-localization-validation.json',
        'fit_perception_settled.py','fit_perception_cup.py','perception_task_labels.py','perception_fit_data.py'):
        shutil.copy2(ROOT/name,archive/name)
    shutil.copy2('/workspace/src/pour_settled_model.py',archive/'runtime-pour_settled_model.py')
    if (ROOT/'staged/pour_settled_model.py').exists():shutil.copy2(ROOT/'staged/pour_settled_model.py',archive/'staged-pour_settled_model.py')
    (archive/'archive-manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in archive.iterdir()}),indent=2)+'\n')
    plan['version']=15;plan['stop_readout_fit']['residual_features']='first7_plus_exact_count_quantiles_and_spatial_shape'
    plan['exact_quantile_repair']='Recover integer red-pixel counts from the 38-pixel histogram; select quantiles by integer cumulative-count/percent comparisons. Preserve 709 feature columns, fixed readout widths/updates, existing splits and all accuracy/numerical limits. Stage this arithmetic repair for fitting; active runtime remains unchanged until accepted deployment.'
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest['protected_parent_files'].update({str(p):sha(p) for p in archive.iterdir()})
    manifest['plan_sha256']=sha(ROOT/'plan.json');(ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('cup-component.json','cup-level-validation.json','cup-level-attempt.pt','cup-localization-validation.json'):
        assert sha(ROOT/name)==sha(archive/name);(ROOT/name).unlink()
    report=dict(plan_sha256=manifest['plan_sha256'],archive=str(archive),helper_sha256=sha(Path(__file__)),settings=plan['stop_readout_fit'],scope=plan['exact_quantile_repair'])
    (ROOT/'exact-quantile-preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
