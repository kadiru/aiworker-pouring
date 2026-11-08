"""Archive a failed locator fit and declare eight additional training fixtures."""
import copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
sys.path[:0]=['/workspace/src','/workspace',str(ROOT)]
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
    plan=read(ROOT/'plan.json');manifest=read(ROOT/'candidate.json');failed=read(ROOT/'cup-neural-localization.json')
    assert plan['version']==10 and plan['perception_round']==3 and len(plan['level_supplement'])==12
    assert not failed['passed'] and not manifest['trained'] and not (ROOT/'component-selection.json').exists()
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert all(sha(Path(p))==d for p,d in manifest['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==d for n,d in manifest['runtime_sources'].items())
    started={read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*PV_A_*/parameters.json')}
    reserved={s['name'] for k in ('test','development','perception_holdout','deferred_size_holdout') for s in plan[k]}
    assert not reserved.intersection(started)
    old=copy.deepcopy(plan);rng=np.random.default_rng(26110831)
    cal=read(ROOT/'camera-frustum-preflight.json')['calibration'];scenes=[];checks=[]
    for i,cut in enumerate((.30,.48,.65,.78,.91,.56,None,None)):
        if i<6:
            s=copy.deepcopy(plan['level_supplement'][2*i])
            s.update(calibration_initial_height=cut,initial_ml=20.)
            with np.load(s['calibration_particle_seed']) as f:points=calibration_seed_points(s,f['cup_local_points'])
            assert vessel_mask(points,cup_profile(s)).all()
            s['cup_initial_ml']=len(points)*s['spacing']**3*1e6
        else:
            # Only declared training development layouts; validation layouts stay excluded.
            source=plan['consumed_task_development'][0 if i==6 else 5]
            assert admitted_tasks(plan)[0 if i==6 else 5][1]=='train'
            s=copy.deepcopy(source)
            s.update(calibration_only=True,policy_evaluation=False,max_seconds=3.5,
                initial_ml=96. if i==6 else 29.,cup_initial_ml=3. if i==6 else 7.)
            s.pop('checkpoint',None)
        s.update(name=f'PV_A_task_supplement_{i:02d}',study_split='train',
            save_particle_trajectory=True,record_particle_diagnostics=True,record_surface_diagnostics=True)
        offset=rng.uniform(-.006,.006,2)
        s['cup_world']=(np.asarray(s['cup_world'])+[*offset,0.]).tolist()
        s['source_xy']=(np.asarray(s['source_xy'])+rng.uniform(-.004,.004,2)).tolist()
        s['declared_cup_geometry']=cup_geometry(s)
        assert s['name'] not in started
        g=s['declared_cup_geometry'];radius=g['rim_radius_m']+.004;c=np.asarray(s['cup_world'])
        uv=np.array([project(c+[x,y,z],cal) for x in (-radius,radius) for y in (-radius,radius) for z in (0.,g['rim_z_m']+.003)])
        assert (uv>8).all() and (uv<1016).all()
        scenes.append(s);checks.append(dict(name=s['name'],seed_cut=cut,frustum_visible=True,
            initialization='settled_seed_subset' if i<6 else 'autonomous_initial_grid',
            scope='New training-only stationary view; no autonomous success claim.'))
    archive=ROOT/'rounds/task-r3-locator-attempt1';archive.mkdir()
    for name in ('plan.json','candidate.json','cup-neural-localization.json','cup-neural-localization-attempt.pt',
        'cup-locator-data-manifest.json','fit_perception_locator.py','perception_fit_data.py','recording-audit.json'):
        shutil.copy2(ROOT/name,archive/name)
    identities={p.name:sha(p) for p in archive.iterdir()}
    (archive/'archive-manifest.json').write_text(json.dumps(dict(files=identities),indent=2)+'\n')
    plan['level_supplement'].extend(scenes);plan['version']+=1
    for key in old:
        if key not in ('level_supplement','version'):assert plan[key]==old[key]
    admitted_scenes(plan);admitted_tasks(plan)
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest['protected_parent_files'].update({str(p):sha(p) for p in archive.iterdir()})
    manifest.update(plan_sha256=sha(ROOT/'plan.json'))
    (ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('cup-neural-localization.json','cup-neural-localization-attempt.pt'):
        assert sha(ROOT/name)==sha(archive/name);(ROOT/name).unlink()
    report=dict(cases=checks,plan_sha256=sha(ROOT/'plan.json'),helper_sha256=sha(Path(__file__)),
        failed_fit=failed['best_attempt']['max_error_mm'],reserved_and_validation_unchanged=True,
        scope='Eight new training-only views: six intermediate fills at jittered training layouts plus two low initial views. Baseline geometry, fluid, rendering, component gates and task thresholds unchanged. No validation or reserved recordings reused.')
    (ROOT/'task-supplement-preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
