"""Use recorded native held-finger angles to initialize authored calibration."""
import copy,hashlib,json,shutil,urllib.request
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    candidate=read(ROOT/'candidate.json');assert not candidate['trained'] and sha(ROOT/'best.pt')==candidate['checkpoint_sha256']
    assert all(sha(Path(p))==v for p,v in candidate['protected_parent_files'].items())
    paths=list(Path('/workspace/outputs/fountain').glob('*/*SZ_A_recovery_23*/samples.json'));assert len(paths)==1
    rows=read(paths[0]);samples=[r for r in rows if r['grasp_verified'] and 77<r['bottle_tilt_deg']<83 and r.get('both_fingers')]
    assert len(samples)>50
    angles=np.median([r['joint_positions_deg'][-4:] for r in samples],axis=0).tolist()
    archive=ROOT/'camera-r2';archive.mkdir()
    for n in ('plan.json','candidate.json','best.pt','recording-audit.json'):shutil.copy2(ROOT/n,archive/n)
    plan=read(ROOT/'plan.json');plan['version']=4
    for part in ('collect','perception_holdout'):
        for s in plan[part]:
            if 'calibration_hold_angle_deg' in s:s['calibration_finger_initial_deg']=angles
    pilot=copy.deepcopy(plan['pilot'][1]);pilot.update(name='PV_A_gripper_ready_00',study_split='camera_pilot',
        cup_detail_resolution=1024,cup_detail_focal_m=.030,record_surface_diagnostics=True,save_particle_trajectory=True,
        calibration_finger_initial_deg=angles)
    plan['camera_pilot'].append(pilot)
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    candidate['runtime_sources']['robot_pour.py']=sha(ROOT/'staged/robot_pour.py')
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False);saved['sources']=candidate['runtime_sources'];torch.save(saved,ROOT/'best.pt')
    candidate.update(checkpoint_sha256=sha(ROOT/'best.pt'),plan_sha256=sha(ROOT/'plan.json'))
    (ROOT/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
    (ROOT/'gripper-initialization.json').write_text(json.dumps(dict(source=str(paths[0]),sha256=sha(paths[0]),frames=len(samples),angles_deg=angles,
        scope='Consumed native held-grasp angles initialize only authored static calibration. Prior static held views remain diagnostic only; no autonomous trajectory or learned motor parameter is changed.'),indent=2)+'\n')
    print(json.dumps(dict(angles_deg=angles,source_frames=len(samples),new_pilots=1)))
if __name__=='__main__':main()
