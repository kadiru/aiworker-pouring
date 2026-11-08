"""Keep the failed grid pilots; prepare all nine settled-seed calibration cases."""
import copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/learning/A-size-20261108-v1')
PARENT=Path('/workspace/learning/A-finish-20261106-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    previous=read(ROOT/'candidate.json');report=read(ROOT/'stage-results.json')
    assert report['calibration_closed'] and len(report['calibration'])==9
    assert not report['development']
    assert sha(ROOT/'best.pt')==previous['checkpoint_sha256']
    assert all(sha(Path('/workspace/src')/n)==v for n,v in previous['runtime_sources'].items())
    assert all(sha(PARENT/n)==v for n,v in previous['protected_parent_files'].items())
    archive=ROOT/'grid-candidate-r0';archive.mkdir()
    for name in ('candidate.json','plan.json','best.pt','stage-results.json'):shutil.copy2(ROOT/name,archive/name)
    sys.path.insert(0,str(ROOT/'staged'))
    from pour_source_watchdog import DEFAULT
    from pour_vessel_geometry import settled_cup_points,cup_profile
    from bottle_pour import vessel_mask
    seed=PARENT/'calibration-particle-seed.npz'
    assert sha(seed)=='1c21c07fe9659e8688cb3bc0365c3c96e5a0efd0dcb57ce60233dea66e7680a8'
    with np.load(seed,allow_pickle=False) as f:points=f['cup_local_points'].copy()
    plan=read(ROOT/'plan.json');cases=[];checks=[]
    for base in plan['calibration']:
        s=dict(base,name=base['name'].replace('_static_','_seed_'),calibration_version=3,
            calibration_particle_seed=str(seed),calibration_particle_seed_sha256=sha(seed),
            record_particle_diagnostics=True,initialization_revision='unscaled_low_layer_from_native_settled_seed',
            grid_pilot_name=base['name'],requested_cup_initial_ml=base['cup_initial_ml'])
        q=s['spacing']**3*1e6;s['cup_initial_ml']=round(base['cup_initial_ml']/q)*q
        selected=settled_cup_points(s,points)
        assert vessel_mask(selected,cup_profile(s)).all()
        assert abs(len(selected)*q-s['cup_initial_ml'])<1e-8
        checks.append(dict(name=s['name'],particles=len(selected),positions_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
            min_z=float(selected[:,2].min()),max_z=float(selected[:,2].max()),nominal_ml=len(selected)*q))
        cases.append(s)
    plan['calibration_seed']=cases
    plan['scope']+=' The nine failed/accepted original grid pilots remain retained; a new nine-case seed-initialization revision changes no liquid coefficients or collision parameters.'
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    changed=('pour_source_watchdog.py','pour_vessel_geometry.py','pour_finish_calibration.py','robot_pour.py','run_fountain.py')
    candidate=copy.deepcopy(previous)
    for name in changed:candidate['runtime_sources'][name]=sha(ROOT/'staged'/name)
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    saved.update(source_watchdog=DEFAULT,sources=candidate['runtime_sources'],model_version='visual_size_stage_v25_seed_preparation')
    torch.save(saved,ROOT/'best.pt')
    candidate.update(checkpoint_sha256=sha(ROOT/'best.pt'),source_watchdog=DEFAULT,
        plan_sha256=sha(ROOT/'plan.json'),previous_candidate_sha256=previous['checkpoint_sha256'])
    (ROOT/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
    (ROOT/'seed-preparation.json').write_text(json.dumps(dict(seed_sha256=sha(seed),cases=checks,
        failed_grid_cases=[r['name'] for r in report['calibration'] if not r['stable_contained']],
        scope='Changes initial particle arrangement only. Original particle locations are subset-selected, never rescaled. Single-particle loss in the first grid round is retained; cause is not established by scalar timing.'),indent=2)+'\n')
    print(json.dumps(dict(prepared=9,checkpoint_sha256=candidate['checkpoint_sha256'])))


if __name__=='__main__':main()
