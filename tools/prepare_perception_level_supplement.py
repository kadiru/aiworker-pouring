"""Add low/near-rim training views without consuming validation or reserved data."""
import copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
sys.path[:0]=[str(ROOT/'staged'),'/workspace/src','/workspace']
from pour_perception_geometry import calibration_seed_points
from pour_vessel_geometry import cup_geometry,cup_profile
from pour_guard_vision import project
from bottle_pour import vessel_mask
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not h['recording']['active']
    manifest=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');assert not manifest['trained'] and 'level_supplement' not in plan
    assert not (ROOT/'component-selection.json').exists() and sha(ROOT/'best.pt')==manifest['checkpoint_sha256']
    assert all(sha(Path(p))==d for p,d in manifest['protected_parent_files'].items())
    refs=[s for s in read(Path('/workspace/learning/A-wide-20261105-v1/frozen-plan.json'))['demonstrations'] if s['study_split']=='train']
    base=next(s for s in plan['collect'] if s['study_split']=='train' and s['cup_height_scale']==s['cup_diameter_scale']==1.)
    fingers=read(ROOT/'gripper-initialization.json')['angles_deg'];cal=read(ROOT/'camera-frustum-preflight.json')['calibration'];scenes=[];checks=[]
    with np.load(base['calibration_particle_seed']) as f:seed=f['cup_local_points']
    for i,index in enumerate((0,4,8,12,16,20)):
        for j,cut in enumerate((.10,1.)):
            s=copy.deepcopy(base);s.update(name=f'PV_A_level_supplement_{2*i+j:02d}',study_split='train',
                calibration_initial_height=cut,cup_world=refs[index]['cup_world'].copy(),source_xy=refs[index]['source_xy'].copy(),calibration_layout_source=refs[index]['name'])
            s.pop('calibration_hold_angle_deg',None);s.pop('calibration_finger_initial_deg',None)
            if (i+j)%2:s.update(calibration_hold_angle_deg=80.,calibration_finger_initial_deg=fingers)
            s['declared_cup_geometry']=cup_geometry(s);points=calibration_seed_points(s,seed);assert vessel_mask(points,cup_profile(s)).all()
            s['cup_initial_ml']=len(points)*s['spacing']**3*1e6
            g=s['declared_cup_geometry'];radius=g['rim_radius_m']+.004;c=np.asarray(s['cup_world'])
            uv=np.array([project(c+[x,y,z],cal) for x in (-radius,radius) for y in (-radius,radius) for z in (0.,g['rim_z_m']+.003)])
            assert (uv>8).all() and (uv<1016).all()
            scenes.append(s);checks.append(dict(name=s['name'],particles=len(points),positions_sha256=hashlib.sha256(points.tobytes()).hexdigest(),frustum_visible=True,seed_cut=cut))
    archive=ROOT/'before-level-supplement';archive.mkdir()
    for n in ('plan.json','candidate.json','best.pt','scope-selection.json'):shutil.copy2(ROOT/n,archive/n)
    plan.update(version=plan['version']+1,level_supplement=scenes,
        perception_training_pilots=[s['name'] for s in plan['camera_pilot']],
        training_pilot_reuse_scope='Four previously inspected camera/grip calibration records may be reused for perception training only. Original recorded splits and evidence kinds remain unchanged.')
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for n in ('pour_perception_geometry.py','pour_perception_study.py'):manifest['runtime_sources'][n]=sha(ROOT/'staged'/n)
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False);saved['sources']=manifest['runtime_sources'];saved['model_version']='visual_perception_v26_calibration_supplement'
    torch.save(saved,ROOT/'best.pt');manifest.update(checkpoint_sha256=sha(ROOT/'best.pt'),plan_sha256=sha(ROOT/'plan.json'))
    (ROOT/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'level-supplement-preparation.json').write_text(json.dumps(dict(cases=checks,plan_sha256=manifest['plan_sha256'],helper_sha256=sha(Path(__file__)),
        scope='Twelve new stationary training cases at six existing training layouts, low and near-rim cuts. No validation or reserved images reused. Seed positions are subsetted without scaling; all fluid coefficients and vessel geometry stay unchanged. Native containment/label checks still required.'),indent=2)+'\n')
    print(json.dumps(dict(new_training_cases=12,reused_calibration_records=4,plan_sha256=manifest['plan_sha256'])))
if __name__=='__main__':main()
