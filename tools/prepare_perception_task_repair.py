"""Preserve failed task development, declare training reuse and fresh attempts."""
import copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
sys.path[:0]=['/workspace/src','/workspace']
from pour_perception_geometry import calibration_seed_points
from pour_vessel_geometry import cup_geometry,cup_profile
from pour_guard_vision import project
from bottle_pour import vessel_mask
from perception_fit_data import admitted_scenes,admitted_tasks
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    assert not (ROOT/'final-selection.json').exists()
    manifest=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json')
    report=read(ROOT/'development-outcomes.json');holdout=read(ROOT/'perception-holdout.json')
    audit=read(ROOT/'recording-audit.json');review=read(ROOT/'development-manual-review.json')
    assert plan['perception_round']==2 and report['counts']['completed']==6 and report['counts']['strict_passes']<6
    assert holdout['passed'] and review['reviewed'] and not audit['started_incomplete']
    assert review['outcomes_sha256']==sha(ROOT/'development-outcomes.json')
    assert report['checkpoint_sha256']==holdout['checkpoint_sha256']==manifest['checkpoint_sha256']==sha(ROOT/'best.pt')
    assert report['plan_sha256']==manifest['plan_sha256']==sha(ROOT/'plan.json')
    assert all(sha(Path(p))==d for p,d in manifest['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==d for n,d in manifest['runtime_sources'].items())
    started={read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*PV_A_*/parameters.json')}
    assert {s['name'] for s in plan['test']}.isdisjoint(started)
    assert {s['name'] for s in plan['development']}=={r['name'] for r in report['cases']}
    old_tasks=copy.deepcopy(plan['development']);old_holdout=copy.deepcopy(plan['perception_holdout'])
    rng=np.random.default_rng(26110803);cal=read(ROOT/'camera-frustum-preflight.json')['calibration']
    scenes=[];checks=[]
    for i,(original,cut) in enumerate(zip(old_holdout,(.14,.48,.78,.93,.99,.89,.12,1.,.96))):
        s=copy.deepcopy(original);s['name']=f'PV_A_baseline_perception_r3_{i:02d}'
        offset=rng.uniform(-.004,.004,2)
        if np.linalg.norm(offset)<.003:offset=offset/np.linalg.norm(offset)*.003
        s['cup_world']=(np.asarray(original['cup_world'])+[*offset,0.]).tolist()
        s['source_xy']=(np.asarray(original['source_xy'])+rng.uniform(-.003,.003,2)).tolist()
        s.update(calibration_initial_height=cut,perception_round=3)
        s['declared_cup_geometry']=cup_geometry(s)
        with np.load(s['calibration_particle_seed']) as f:points=calibration_seed_points(s,f['cup_local_points'])
        assert vessel_mask(points,cup_profile(s)).all();s['cup_initial_ml']=len(points)*s['spacing']**3*1e6
        g=s['declared_cup_geometry'];radius=g['rim_radius_m']+.004;c=np.asarray(s['cup_world'])
        uv=np.array([project(c+[x,y,z],cal) for x in (-radius,radius) for y in (-radius,radius) for z in (0.,g['rim_z_m']+.003)])
        assert (uv>8).all() and (uv<1016).all() and s['name'] not in started
        scenes.append(s);checks.append(dict(name=s['name'],seed_cut=cut,particles=len(points),positions_sha256=hashlib.sha256(points.tobytes()).hexdigest()))
    tasks=[]
    for i,original in enumerate(old_tasks):
        s=copy.deepcopy(original);s.update(name=f'PV_A_development_r1_{i:02d}',development_round=1)
        offset=rng.uniform(-.003,.003,2)
        if np.linalg.norm(offset)<.002:offset=offset/np.linalg.norm(offset)*.002
        s['cup_world']=(np.array(s['cup_world'])+[*offset,0.]).tolist()
        s['source_xy']=(np.array(s['source_xy'])+rng.uniform(-.003,.003,2)).tolist()
        assert s['name'] not in started;tasks.append(s)
    archive=ROOT/'rounds/baseline-r2';archive.mkdir();(archive/'src').mkdir()
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.json','.pt','.py'):shutil.copy2(p,archive/p.name)
    for n in manifest['runtime_sources']:shutil.copy2(Path('/workspace/src')/n,archive/'src'/n)
    for n in ('development-review',):
        if (ROOT/n).exists():shutil.copytree(ROOT/n,archive/n)
    identities={str(p.relative_to(archive)):sha(p) for p in archive.rglob('*') if p.is_file()}
    (archive/'archive-manifest.json').write_text(json.dumps(dict(files=identities,scope='Passed static perception and failed autonomous task development retained together.'),indent=2)+'\n')
    history=dict(round=0,checkpoint_sha256=report['checkpoint_sha256'],report_path=str(archive/'development-outcomes.json'),
        report_sha256=sha(archive/'development-outcomes.json'),fit_splits={s['name']:('validation' if i in (2,3) else 'train') for i,s in enumerate(old_tasks)},
        scope='Explicit consumed development tuning only. Indices 00/01/04/05 train; 02/03 validation, declared before fitting. Prior failures retain their original scores and recorded split. Passed round-2 static holdout remains excluded from fitting.')
    plan.setdefault('task_repair_history',[]).append(history)
    plan.setdefault('consumed_task_development',[]).extend(old_tasks)
    plan.setdefault('archived_perception_holdout',[]).extend(old_holdout)
    plan.update(version=plan['version']+1,perception_round=3,perception_holdout=scenes,development=tasks)
    admitted_scenes(plan);admitted_tasks(plan)
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest['protected_parent_files'].update({str(p):sha(p) for p in archive.rglob('*') if p.is_file()})
    manifest.update(trained=False,component_gates_passed=False,plan_sha256=sha(ROOT/'plan.json'),perception_round=3,
        scope='Task-domain perception repair initialized from round-2 weights. Consumed development only; independent reserved tests remain unstarted. Original-size scope only.')
    (ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for n in ('component-selection.json','perception-holdout.json','baseline-holdout-geometry.json','cup-component.json','cup-component.pt',
        'cup-level-validation.json','cup-localization-validation.json','cup-level-attempt.pt','cup-level-predictions.json',
        'cup-neural-localization.json','cup-neural-localization.pt','cup-neural-localization-attempt.pt','scope-selection.json',
        'development-outcomes.json','development-visual-review.json','development-manual-review.json'):
        p=ROOT/n
        if p.exists():assert sha(p)==sha(archive/n);p.unlink()
    if (ROOT/'development-review').exists():shutil.rmtree(ROOT/'development-review')
    (ROOT/'task-repair-preparation-r3.json').write_text(json.dumps(dict(history=history,static_cases=checks,
        fresh_tasks=[s['name'] for s in tasks],plan_sha256=sha(ROOT/'plan.json'),helper_sha256=sha(Path(__file__))),indent=2)+'\n')
    print(json.dumps(dict(round=3,consumed_tasks=len(old_tasks),new_holdouts=len(scenes),new_development=len(tasks),archive=str(archive))))
if __name__=='__main__':main()
