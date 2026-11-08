"""Preserve failed level fitting and declare pre-stop-only task supervision."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    plan=read(ROOT/'plan.json');manifest=read(ROOT/'candidate.json');failed=read(ROOT/'cup-component.json')
    assert plan['version']==11 and plan['perception_round']==3
    assert not manifest['trained'] and not failed['passed'] and failed['localization_passed']
    assert not (ROOT/'component-selection.json').exists() and not (ROOT/'final-selection.json').exists()
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert all(sha(Path(p))==d for p,d in manifest['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==d for n,d in manifest['runtime_sources'].items())
    archive=ROOT/'rounds/task-r3-level-attempt1';archive.mkdir()
    names=('plan.json','candidate.json','cup-component.json','cup-level-validation.json','cup-level-attempt.pt',
        'cup-level-predictions.json','cup-data-manifest.json','cup-localization-validation.json',
        'fit_perception_cup.py','perception_task_labels.py','perception_fit_data.py')
    for name in names:shutil.copy2(ROOT/name,archive/name)
    (archive/'archive-manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in archive.iterdir()}),indent=2)+'\n')
    plan.update(version=12,task_level_frame_scope='Pre-stop settling decisions only. Require observed grasp, known image-localized cup center and causal stable half-second label. Exclude all stop-latched return frames, regardless of inventory stability. Original train/validation episode splits and task scores unchanged.')
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest['protected_parent_files'].update({str(p):sha(p) for p in archive.iterdir()})
    manifest['plan_sha256']=sha(ROOT/'plan.json');(ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('cup-component.json','cup-level-validation.json','cup-level-attempt.pt','cup-level-predictions.json','cup-localization-validation.json'):
        assert sha(ROOT/name)==sha(archive/name);(ROOT/name).unlink()
    report=dict(plan_sha256=manifest['plan_sha256'],archive=str(archive),helper_sha256=sha(Path(__file__)),
        frame_scope=plan['task_level_frame_scope'],scope='Offline supervision repair only. Same recorded data, fixed validation episodes, unchanged controller/runtime, component gates, scoring, physics and material. Failed original fit remains archived.')
    (ROOT/'decision-label-preparation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
