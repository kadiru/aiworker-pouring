"""Native evidence audit for stationary calibration and consumed-layout regressions."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/src','/workspace']
from PIL import Image,ImageDraw
from validate_pour_recording import h5py,decode,validate
from pour_size_study import ROOT,OUTPUTS
from pour_robust_report import completed_row
from pour_wide_report import strict_success,reasons

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,type(a)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    plan=read(ROOT/'plan.json');candidate=read(ROOT/'candidate.json')
    parent=Path('/workspace/learning/A-finish-20261106-v1')
    assert all(sha(parent/n)==v for n,v in candidate['protected_parent_files'].items())
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    previous=torch.load(parent/'best.pt',map_location='cpu',weights_only=False)
    allowed={'source_watchdog','model_version','parent_checkpoint_sha256','method','sources'}
    assert equal({k:v for k,v in saved.items() if k not in allowed},{k:v for k,v in previous.items() if k not in allowed})
    assert sha(ROOT/'best.pt')==candidate['checkpoint_sha256']
    for n,h in candidate['runtime_sources'].items():assert sha(Path('/workspace/src')/n)==h,n
    expected={s['name']:s for part in plan.values() if isinstance(part,list) for s in part}
    calibration=[];seed_calibration=[];development=[];images={};pending=[];seen=set();integrity=[]
    out=ROOT/'review';out.mkdir(exist_ok=True)
    for path in sorted(OUTPUTS.glob('*/*SZ_A_*/parameters.json')):
        spec=read(path);name=spec['name'];assert name in expected and name not in seen
        seen.add(name);assert all(spec[k]==v for k,v in expected[name].items())
        folder=path.parent;h5=folder/'episode.hdf5'
        if not h5.exists() or not (folder/'metrics.json').exists():pending.append(name);continue
        valid=validate(h5);samples=read(folder/'samples.json');m=read(folder/'metrics.json');last=samples[-1]
        integrity.append(dict(name=name,path=str(h5),sha256=sha(h5),bytes=h5.stat().st_size,
            transitions=valid['transitions'],max_camera_skew_s=valid['max_camera_state_skew_s'],
            sidecars={p.name:sha(p) for p in folder.glob('*.json')}))
        with h5py.File(h5) as f:
            d=f['data/demo_0'];rgb=decode(d['next_obs/task_image'][-1],'jpeg95')
            k_error=float(np.max(abs(f['calibration/task/intrinsics'][:]-np.asarray(saved['calibration']['K']))))
            poses=np.concatenate([d[g+'/task_world_T_camera'][:] for g in ('obs','next_obs')])
            t_error=float(np.max(abs(poses-np.asarray(saved['calibration']['T']))))
            camera_motion=float(np.max(abs(poses-poses[0])))
            assert k_error<1e-3 and t_error<1e-6 and camera_motion<1e-9
            integrity[-1]['fixed_task_camera_check']=dict(intrinsics_max_abs_error=k_error,
                extrinsics_max_abs_error=t_error,camera_pose_change=camera_motion)
            Image.fromarray(rgb).save(out/(name+'.jpg'))
            if spec.get('calibration_only'):
                tail=[r for r in samples if r['t']>=last['t']-.5]
                commands=float(np.ptp(d['actions'][:],axis=0).max())
                motion=float(np.ptp(np.array([r['joint_positions_deg'][:7] for r in tail]),axis=0).max())
                record=dict(name=name,height_scale=spec['cup_height_scale'],diameter_scale=spec['cup_diameter_scale'],
                    geometry=spec['declared_cup_geometry'],initial_cup_nominal_ml=spec['cup_initial_ml'],
                    final_height=last['cup_fill_fraction'],final_cup_nominal_ml=last['cup_ml'],
                    final_outside_nominal_ml=last['spilled_ml']+last['airborne_ml'],ever_table_nominal_ml=last['ever_table_ml'],
                    command_range_rad=commands,tail_arm_motion_deg=motion,
                    tail_height_std=float(np.std([r['cup_fill_fraction'] for r in tail])),
                    tail_inventory_range_ml=float(np.ptp([r['cup_ml'] for r in tail])))
                record['stable_contained']=bool(commands<1e-7 and motion<.3 and record['tail_height_std']<.01 and
                    record['tail_inventory_range_ml']<.05 and record['final_outside_nominal_ml']==record['ever_table_nominal_ml']==0.)
                seeded=spec.get('calibration_version')==3
                (seed_calibration if seeded else calibration).append(record)
                images.setdefault('seed' if seeded else 'grid',[]).append((record,rgb))
            else:
                assert m['policy']['checkpoint_sha256']==candidate['checkpoint_sha256']
                r=completed_row(folder,m['policy']['checkpoint_sha256'])
                r.update(strict_pass=strict_success(r),failure_reasons=reasons(r),former_test_case=spec['former_test_case'])
                development.append(r)
    for group,frames in images.items():
        canvas=Image.new('RGB',(3*384,3*424),(245,245,245));draw=ImageDraw.Draw(canvas)
        for i,(r,rgb) in enumerate(frames):
            x=(i%3)*384;y=(i//3)*424
            canvas.paste(Image.fromarray(rgb),(x,y+40))
            draw.text((x+8,y+7),f"H {r['height_scale']:.1f} / D {r['diameter_scale']:.1f} | {'contained' if r['stable_contained'] else 'CHECK'}",fill=(20,20,20))
            draw.text((x+8,y+22),f"settled level {100*r['final_height']:.1f}%",fill=(20,20,20))
        canvas.save(out/('cup-size-seed-contact-sheet.jpg' if group=='seed' else 'cup-size-contact-sheet.jpg'))
    report=dict(scope='Stationary cup geometry/containment pilots and development on consumed V24 layouts, not fresh held-out tests. Learned weights unchanged.',
        parent_files_unchanged=True,learned_weights_and_original_thresholds_unchanged=True,
        expected_calibration=9,expected_development=6,
        calibration=calibration,seed_calibration=seed_calibration,development=development,integrity=integrity,pending=pending,
        unstarted=sorted(set(expected)-seen),calibration_closed=len(calibration)==9,
        all_calibration_contained=len(calibration)==9 and all(r['stable_contained'] for r in calibration),
        all_seed_calibration_contained=len(seed_calibration)==9 and all(r['stable_contained'] for r in seed_calibration),
        development_closed=len(development)==6,
        candidate_checkpoint_sha256=candidate['checkpoint_sha256'],helper_sha256=sha(Path(__file__)))
    (ROOT/'stage-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(calibration=len(calibration),contained=sum(r['stable_contained'] for r in calibration),
        seed_calibration=len(seed_calibration),seed_contained=sum(r['stable_contained'] for r in seed_calibration),
        development=len(development),passes=sum(r['strict_pass'] for r in development),pending=pending,unstarted=report['unstarted'])))
    if len(seed_calibration)==9 and not (ROOT/'low-fill-temporal-perception-probe.json').exists():
        from probe_size_perception import main as probe
        probe()


if __name__=='__main__':main()
