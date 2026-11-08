"""Archive a failed independent round, explicitly admit it, declare fresh checks."""
import argparse,copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
sys.path[:0]=['/workspace/src','/workspace']
from pour_perception_geometry import calibration_seed_points
from pour_vessel_geometry import cup_geometry,cup_profile
from pour_guard_vision import project
from bottle_pour import vessel_mask
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--expected-current-round',type=int,choices=(0,1),default=0);args=parser.parse_args()
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    assert not (ROOT/'final-selection.json').exists()
    manifest=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');report=read(ROOT/'perception-holdout.json');audit=read(ROOT/'recording-audit.json')
    assert manifest['trained'] and not report['passed'] and not audit['started_incomplete']
    assert report['checkpoint_sha256']==manifest['checkpoint_sha256']==sha(ROOT/'best.pt')
    assert report['plan_sha256']==manifest['plan_sha256']==sha(ROOT/'plan.json')
    assert all(sha(Path(p))==d for p,d in manifest['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==d for n,d in manifest['runtime_sources'].items())
    old=plan['perception_holdout'];assert {s['name'] for s in old}=={r['name'] for r in report['cases']}
    started={read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*PV_A_*/parameters.json')}
    assert {s['name'] for k in ('development','test') for s in plan[k]}.isdisjoint(started)
    assert all(r['measured_label']['accepted'] for r in report['cases'])
    previous_round=plan.get('perception_round',0);new_round=previous_round+1
    assert previous_round==args.expected_current_round,'Unexpected study round; do not silently consume another evaluation'
    cuts={1:(.12,.45,.75,.91,.98,.85,.12,1.,.96),2:(.13,.50,.77,.94,.99,.88,.15,1.,.97)}[new_round]
    rng=np.random.default_rng(26110800+new_round);cal=read(ROOT/'camera-frustum-preflight.json')['calibration'];scenes=[];checks=[]
    for i,(original,cut) in enumerate(zip(old,cuts)):
        s=copy.deepcopy(original);s['name']=f'PV_A_baseline_perception_r{new_round}_{i:02d}'
        offset=rng.uniform(-.004,.004,2)
        if np.linalg.norm(offset)<.003:offset=offset/np.linalg.norm(offset)*.003
        s['cup_world']=(np.asarray(original['cup_world'])+[*offset,0.]).tolist()
        s['source_xy']=(np.asarray(original['source_xy'])+rng.uniform(-.003,.003,2)).tolist()
        s.update(calibration_initial_height=cut,perception_round=new_round)
        s['declared_cup_geometry']=cup_geometry(s)
        with np.load(s['calibration_particle_seed']) as f:points=calibration_seed_points(s,f['cup_local_points'])
        assert vessel_mask(points,cup_profile(s)).all();s['cup_initial_ml']=len(points)*s['spacing']**3*1e6
        g=s['declared_cup_geometry'];radius=g['rim_radius_m']+.004;c=np.asarray(s['cup_world'])
        uv=np.array([project(c+[x,y,z],cal) for x in (-radius,radius) for y in (-radius,radius) for z in (0.,g['rim_z_m']+.003)])
        assert (uv>8).all() and (uv<1016).all() and s['name'] not in started
        scenes.append(s);checks.append(dict(name=s['name'],cup_offset_m=offset.tolist(),seed_cut=cut,particles=len(points),
            positions_sha256=hashlib.sha256(points.tobytes()).hexdigest(),frustum_visible=True))
    archive=ROOT/'rounds'/f'baseline-r{previous_round}';archive.mkdir(parents=True);(archive/'src').mkdir()
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.json','.pt','.py'):shutil.copy2(p,archive/p.name)
    for n in manifest['runtime_sources']:shutil.copy2(Path('/workspace/src')/n,archive/'src'/n)
    identities={str(p.relative_to(archive)):sha(p) for p in archive.rglob('*') if p.is_file()}
    (archive/'archive-manifest.json').write_text(json.dumps(dict(files=identities,failed_holdout_sha256=sha(archive/'perception-holdout.json')),indent=2)+'\n')
    history=dict(round=previous_round,checkpoint_sha256=report['checkpoint_sha256'],report_path=str(archive/'perception-holdout.json'),
        report_sha256=sha(archive/'perception-holdout.json'),passed=False,consumed_training_names=[s['name'] for s in old],
        scope='Failed independent evaluation preserved. These nine stationary recordings are now explicitly admitted as training only; their original recorded splits and old failure result do not change.')
    plan.setdefault('perception_history',[]).append(history);plan.setdefault('consumed_perception_training',[]).extend(copy.deepcopy(old))
    plan.update(version=plan['version']+1,perception_round=new_round,perception_holdout=scenes)
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest['protected_parent_files'].update({str(p):sha(p) for p in archive.rglob('*') if p.is_file()})
    manifest.update(trained=False,component_gates_passed=False,plan_sha256=sha(ROOT/'plan.json'),perception_round=new_round,
        scope=f'Repair initialized from trained rejected round-{previous_round} weights; current-round component acceptance pending. Prior independent failures remain frozen in rounds/.')
    (ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for n in ('component-selection.json','perception-holdout.json','baseline-holdout-geometry.json','cup-component.json','cup-component.pt',
              'cup-level-validation.json','cup-localization-validation.json','cup-level-attempt.pt','cup-level-predictions.json',
              'cup-neural-localization.json','cup-neural-localization.pt','cup-neural-localization-attempt.pt','scope-selection.json','cup-locator-ensemble.json'):
        p=ROOT/n
        if p.exists():assert sha(p)==sha(archive/n);p.unlink()
    (ROOT/f'repair-round-{new_round}-preparation.json').write_text(json.dumps(dict(history=history,new_cases=checks,plan_sha256=sha(ROOT/'plan.json'),helper_sha256=sha(Path(__file__))),indent=2)+'\n')
    print(json.dumps(dict(round=new_round,consumed_failed_holdouts=len(old),new_unstarted_holdouts=len(scenes),archive=str(archive),plan_sha256=sha(ROOT/'plan.json'))))
if __name__=='__main__':main()
