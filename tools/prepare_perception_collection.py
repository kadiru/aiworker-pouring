"""Freeze wider camera/data coverage before any level/localization fitting."""
import copy,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    sys.path.insert(0,str(ROOT/'staged'))
    from pour_guard_vision import project
    from pour_vessel_geometry import cup_geometry
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    report=read(ROOT/'recording-audit.json');diagnostic=[r for r in report['records'] if r['name'].startswith('PV_A_geometry_')]
    assert len(diagnostic)==3 and all(r['column_label']['accepted'] for r in diagnostic)
    selection=read(ROOT/'surface-label-validation.json');assert selection['passed']
    candidate=read(ROOT/'candidate.json');assert sha(ROOT/'best.pt')==candidate['checkpoint_sha256']
    archive=ROOT/'geometry-r1';archive.mkdir()
    for n in ('plan.json','candidate.json','best.pt','recording-audit.json'):shutil.copy2(ROOT/n,archive/n)
    plan=read(ROOT/'plan.json');plan['version']=3
    refs=read(Path('/workspace/learning/A-wide-20261105-v1/frozen-plan.json'))['demonstrations']
    train=[s for s in refs if s['study_split']=='train'];val=[s for s in refs if s['study_split']=='validation']
    counts={};holdout=[];collection=[]
    for s in plan['collect']:
        split=s['study_split'];i=counts.get(split,0);counts[split]=i+1
        if split=='perception_test':
            layout=plan['test'][i%6]
        else:layout=(train if split=='train' else val)[i%(len(train) if split=='train' else len(val))]
        s.update(cup_world=layout['cup_world'].copy(),source_xy=layout['source_xy'].copy(),
            cup_detail_resolution=1024,cup_detail_focal_m=.030,calibration_layout_source=layout['name'])
        (holdout if split=='perception_test' else collection).append(s)
    plan['collect']=collection;plan['perception_holdout']=holdout;plan['camera_pilot']=[]
    for i,index in enumerate((3,8,19)):
        s=copy.deepcopy(next(s for s in collection if s['cup_height_scale']==1. and s['cup_diameter_scale']==1.))
        s.update(name=f'PV_A_wide_camera_{i:02d}',study_split='camera_pilot',cup_world=train[index]['cup_world'].copy(),source_xy=train[index]['source_xy'].copy())
        s.pop('calibration_hold_angle_deg',None);plan['camera_pilot'].append(s)
    for s in plan['development']+plan['test']:
        s.update(cup_detail_resolution=1024,cup_detail_focal_m=.030,record_surface_diagnostics=True)
    # Near-full baseline validation is declared before its images exist.
    from pour_perception_geometry import calibration_seed_points
    for i in range(2):
        s=copy.deepcopy(next(s for s in collection if s['cup_height_scale']==1. and s['cup_diameter_scale']==1.))
        s.update(name=f'PV_A_validation_full_{i:02d}',study_split='validation',calibration_initial_height=1.,
            cup_world=val[2*i+1]['cup_world'].copy(),source_xy=val[2*i+1]['source_xy'].copy())
        s.pop('calibration_hold_angle_deg',None)
        if i:s['calibration_hold_angle_deg']=80.
        with np.load(s['calibration_particle_seed']) as f:p=calibration_seed_points(s,f['cup_local_points'])
        s['cup_initial_ml']=len(p)*s['spacing']**3*1e6;plan['collect'].append(s)
    plan['training_label']='Final-second median of each recording\'s measured cup_fill_fraction; spatial column-top diagnostic retained separately. No fixed inventory-to-height target mapping. Legacy instantaneous scores unchanged.'
    plan['component_gates']=dict(localization_max_error_mm=3.,localization_negative_margin=.015,
        level_validation_mae=.02,level_validation_near_max_error=.035,level_validation_near_overestimate=.025,
        geometry_validation_max_error=.06,stop_target_surface_fraction=.95)
    cal=copy.deepcopy(report['records'][0]['cameras']['cup_detail']['calibration'])
    K=np.asarray(cal['K']);K[0,0]*=1024/768*.030/.045;K[1,1]*=1024/768*.030/.045;K[0,2]=K[1,2]=512
    cal['K']=K.tolist();cal['resolution_wh']=[1024,1024]
    checks=[]
    for part in ('camera_pilot','collect','perception_holdout','development','test'):
        for s in plan[part]:
            g=cup_geometry(s);c=np.asarray(s['cup_world']);radius=g['rim_radius_m']+.004
            uv=np.array([project(c+[dx,dy,z],cal) for dx in (-radius,radius) for dy in (-radius,radius) for z in (0.,g['rim_z_m']+.003)])
            checks.append(dict(name=s['name'],min_uv=uv.min(0).tolist(),max_uv=uv.max(0).tolist(),visible=bool((uv>8).all() and (uv<1016).all())))
    assert all(r['visible'] for r in checks),[r for r in checks if not r['visible']]
    (ROOT/'camera-frustum-preflight.json').write_text(json.dumps(dict(calibration=cal,cases=checks,scope='Ideal projection screen; native view/occlusion checks still required.'),indent=2)+'\n')
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for name in ('pour_recording.py','validate_pour_recording.py','run_fountain.py','pour_perception_study.py'):
        candidate['runtime_sources'][name]=sha(ROOT/'staged'/name)
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False);saved['sources']=candidate['runtime_sources']
    torch.save(saved,ROOT/'best.pt');candidate.update(checkpoint_sha256=sha(ROOT/'best.pt'),plan_sha256=sha(ROOT/'plan.json'))
    (ROOT/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
    print(json.dumps(dict(collection=len(plan['collect']),holdout=len(holdout),camera_pilots=3,frustum_passed=len(checks))))
if __name__=='__main__':main()
