"""Check native stationary pilots/calibration, never report task success."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from PIL import Image
from pour_finish_study import ROOT,OUTPUTS
p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--pilot-start',type=int,default=0);a=p.parse_args()
plan=json.loads((ROOT/'calibration-plan.json').read_text())['cases'];expected={s['name']:s for s in (plan[a.pilot_start:a.pilot_start+3] if a.pilot else plan)}
rows=[];out=ROOT/(f'calibration-pilot-review-{a.pilot_start}' if a.pilot else 'calibration-review');out.mkdir(exist_ok=True)
for path in sorted(OUTPUTS.glob('*/*FCAL_A_*/episode.hdf5')):
    spec=json.loads((path.parent/'parameters.json').read_text());name=spec['name']
    if name not in expected:continue
    assert spec==expected[name]
    samples=json.loads((path.parent/'samples.json').read_text());tail=[r for r in samples if r['t']>=samples[-1]['t']-.5]
    with h5py.File(path) as f:
        assert f.attrs['status']=='complete';d=f['data/demo_0'];command_range=float(np.ptp(d['actions'][:],axis=0).max())
        assert command_range<1e-7,(name,'Moving command during calibration',command_range)
        Image.fromarray(decode(d['next_obs/task_image'][-1],'jpeg95')).save(out/(name+'.jpg'))
    motion=float(np.ptp(np.array([r['joint_positions_deg'][:7] for r in tail]),axis=0).max())
    height_std=float(np.std([r['cup_fill_fraction'] for r in tail]));inventory_range=float(np.ptp([r['cup_ml'] for r in tail]))
    row=dict(name=name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),initial_height=spec['calibration_initial_height'],initial_nominal_ml=spec['cup_initial_ml'],final_height=samples[-1]['cup_fill_fraction'],command_range_rad=command_range,tail_arm_motion_deg=motion,tail_height_std=height_std,tail_inventory_range_ml=inventory_range,final_outside_ml=samples[-1]['spilled_ml']+samples[-1]['airborne_ml'],image=str(out/(name+'.jpg')))
    row['stable']=motion<.3 and height_std<.01 and inventory_range<.05;rows.append(row)
result=dict(scope='Stationary supervised camera calibration, not autonomous task success',expected=len(expected),closed=len(rows),all_stable=all(r['stable'] for r in rows),rows=rows)
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
assert len(rows)==len(expected) and result['all_stable']
