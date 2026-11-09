"""Archive the failed/interrupted grid batch; declare three gradual-fill pilots."""
import copy
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/src','/workspace']
from pour_vessel_geometry import settled_cup_points
ROOT=Path('/workspace/learning/A-scale-20261109-v1')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')


def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error')
    old=read(ROOT/'plan.json');m=read(ROOT/'candidate.json')
    assert old['version']==1 and not m['trained']
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    interruption=read(ROOT/'pilot-r0-interruption.json')
    assert [r['status'] for r in interruption['cases']]==['complete','interrupted_partial']+['unstarted']*7
    seed=Path('/workspace/learning/A-finish-20261106-v1/calibration-particle-seed.npz')
    assert sha(seed)=='1c21c07fe9659e8688cb3bc0365c3c96e5a0efd0dcb57ce60233dea66e7680a8'
    with np.load(seed,allow_pickle=False) as f:points=f['cup_local_points'].copy()
    archive=ROOT/'seed-r0';archive.mkdir()
    for name in ('plan.json','candidate.json','pilot-geometry.json','seed-preparation.json','pilot-r0-interruption.json'):
        shutil.copy2(ROOT/name,archive/name)
    cases=[];checks=[]
    for i,index in enumerate((0,4,8)):
        s=copy.deepcopy(old['pilot'][index]);s.pop('policy_family',None)
        s.pop('calibration_initial_height',None)
        s.update(name=f'SC_A_jet_r1_{i:02d}',study_split='seed_jet_pilot',
                 calibration_version=3,calibration_particle_seed=str(seed),calibration_particle_seed_sha256=sha(seed),
                 cup_initial_ml=12*s['cup_height_scale']*s['cup_diameter_scale']**2,
                 max_seconds=19.,initialization_revision='low_measured_seed_then_native_gradual_jet',
                 calibration_fill=dict(flow_nominal_ml_s=8.,target_height_fraction=.90,
                                       start_s=.5,max_feed_s=15.,settle_s=3.))
        selected=settled_cup_points(s,points)
        s['cup_initial_ml']=len(selected)*s['spacing']**3*1e6
        cases.append(s);checks.append(dict(name=s['name'],initial_particles=len(selected),initial_nominal_ml=s['cup_initial_ml']))
    plan=dict(old,version=2,pilot=cases,pilot_r0=old['pilot'],pilot_round='r1',
              scope='Three gradual-fill calibration pilots at .9/.9,1/1,1.1/1.1 cup scales. Parked arm, original bottle, unchanged fluid coefficients and particle resolution. Not autonomous pouring.',
              excluded_initialization='Dense r0 grid rejected for immediate fluid ejection. One completed, one interrupted by Docker daemon restart, seven unstarted; all retained.')
    write(ROOT/'plan.json',plan)
    m.update(plan_sha256=sha(ROOT/'plan.json'),stage='gradual_seed_calibration',pilot_round='r1')
    for name in ('run_fountain.py','pour_scale_fill.py'):
        m['runtime_sources'][name]=sha(ROOT/'staged'/name)
    write(ROOT/'candidate.json',m)
    write(ROOT/'gradual-seed-preparation.json',dict(cases=checks,source_seed=str(seed),source_seed_sha256=sha(seed),
        source_seed_had_historical_spill=True,scope='Only its physically retained low layer initializes each cup; fresh jet filling and independent settling verification are required.',
        helper_sha256=sha(Path(__file__))))
    print(json.dumps(dict(pilots=len(cases),plan_sha256=m['plan_sha256'],trained=False)))


if __name__=='__main__':main()
