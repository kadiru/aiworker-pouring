"""After reviewed bounding pilots, declare the six remaining cup-shape seeds."""
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
    plan=read(ROOT/'plan.json');m=read(ROOT/'candidate.json');audit=read(ROOT/'pilot-r1-audit.json')
    review=read(ROOT/'pilot-r1-manual-review.json')
    assert plan['version']==2 and plan['pilot_round']=='r1' and not m['trained']
    assert audit['complete'] and audit['labels_accepted']==3 and all(r['high_fill_coverage'] for r in audit['records'])
    assert review['accepted'] and review['audit_sha256']==sha(ROOT/'pilot-r1-audit.json')
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    seed=Path(plan['pilot'][0]['calibration_particle_seed'])
    assert sha(seed)==plan['pilot'][0]['calibration_particle_seed_sha256']
    with np.load(seed,allow_pickle=False) as f:points=f['cup_local_points'].copy()
    archive=ROOT/'seed-r1';archive.mkdir()
    for name in ('plan.json','candidate.json','pilot-geometry.json','pilot-r1-audit.json','pilot-r1-manual-review.json'):
        shutil.copy2(ROOT/name,archive/name)
    cases=[]
    for i,index in enumerate((1,2,3,5,6,7)):
        s=copy.deepcopy(plan['pilot'][0]);old=plan['pilot_r0'][index]
        for key in ('cup_height_scale','cup_diameter_scale','declared_cup_geometry'):s[key]=copy.deepcopy(old[key])
        s.update(name=f'SC_A_jet_r2_{i:02d}',cup_initial_ml=12*s['cup_height_scale']*s['cup_diameter_scale']**2)
        selected=settled_cup_points(s,points);s['cup_initial_ml']=len(selected)*s['spacing']**3*1e6;cases.append(s)
    plan.update(version=3,pilot_round='r2',pilot_r1=plan['pilot'],pilot=cases,
                scope='Six remaining stationary cup-shape seeds, using the unchanged physically reviewed gradual-fill protocol. All nine shape combinations will require separate seed/camera acceptance before training collection.')
    write(ROOT/'plan.json',plan);m.update(plan_sha256=sha(ROOT/'plan.json'),pilot_round='r2')
    write(ROOT/'candidate.json',m)
    print(json.dumps(dict(pilots=6,plan_sha256=m['plan_sha256'],trained=False)))


if __name__=='__main__':main()
