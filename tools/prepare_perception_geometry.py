"""Archive pilot state; add particle-backed label diagnostics before fitting."""
import copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
import torch
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    audit=read(ROOT/'recording-audit.json');assert len(audit['records'])==3
    assert all(r['label']['accepted'] for r in audit['records'])
    candidate=read(ROOT/'candidate.json');assert sha(ROOT/'best.pt')==candidate['checkpoint_sha256']
    archive=ROOT/'pilot-r0';archive.mkdir()
    for n in ('plan.json','candidate.json','best.pt','recording-audit.json'):shutil.copy2(ROOT/n,archive/n)
    plan=read(ROOT/'plan.json');plan['version']=2;plan['geometry']=[]
    for i in (0,1,2):
        s=copy.deepcopy(plan['pilot'][i]);s.update(name=f'PV_A_geometry_{i:02d}',study_split='label_diagnostic',
            record_surface_diagnostics=True,save_particle_trajectory=True)
        plan['geometry'].append(s)
    for i in range(4):
        s=copy.deepcopy(plan['pilot'][2]);s.pop('calibration_hold_angle_deg',None)
        s.update(name=f'PV_A_presence_absent_{i:02d}',study_split='presence_train' if i<2 else 'presence_validation',
            calibration_arm_angle_deg=80. if i%2==0 else 65.,source_xy=[.115,.045+i*.003],
            record_surface_diagnostics=True,save_particle_trajectory=True)
        plan['geometry'].append(s)
    for s in plan['collect']:s.update(record_surface_diagnostics=True,save_particle_trajectory=True)
    plan['label_diagnostic']='Spatial column tops are recorded alongside the unchanged old height. Compare retained inventory, particle trajectories and native images before accepting a revised training target.'
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for name in ('pour_perception_geometry.py','robot_pour.py','run_fountain.py'):
        candidate['runtime_sources'][name]=sha(ROOT/'staged'/name)
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False);saved['sources']=candidate['runtime_sources']
    torch.save(saved,ROOT/'best.pt');candidate.update(checkpoint_sha256=sha(ROOT/'best.pt'),plan_sha256=sha(ROOT/'plan.json'))
    (ROOT/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
    print(json.dumps(dict(geometry_cases=7,learned_weights_unchanged=True)))
if __name__=='__main__':main()
