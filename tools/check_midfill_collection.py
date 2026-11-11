"""Audit size-calibration records and extract measured seeds; no task claims."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path[:0] = ['/workspace/src', '/workspace']
from validate_pour_recording import h5py, decode, validate
from pour_perception_geometry import settled_label
from pour_vessel_geometry import cup_profile
from bottle_pour import vessel_mask
from robot_pour import FRAME
ROOT = Path('/workspace/learning/A-scale-dynamic-20261109-v1')


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=('seed_pilot','seed_remaining','low_train','train','validation','holdout'),required=True)
    parser.add_argument('--round',choices=('r3',),default='r3');a=parser.parse_args()
    m=read(ROOT/'candidate.json');plan=read(ROOT/'protocol.json');declared=plan.get(a.part)
    if a.part.startswith('seed_'):
        pilot=set(plan['seed_pilot_names']);declared=[s for s in plan['seed'] if (s['name'] in pilot)==(a.part=='seed_pilot')]
    assert a.part=='train' and plan['version']==6
    wanted=set(plan['midfill_train_names'])
    expected={s['name']:s for s in declared if s['name'] in wanted}
    assert expected.keys()==wanted
    assert sha(ROOT/'plan.json')==m['plan_sha256']
    assert all(sha(Path(p))==h for p,h in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==h for n,h in m['runtime_sources'].items())
    if a.part=='pilot':assert plan.get('pilot_round','r0')==a.round
    out=ROOT/(a.part+'-'+a.round+'-review');out.mkdir(exist_ok=True);records=[]
    for p in sorted(Path('/workspace/outputs/fountain').glob('*/*SD_A_*/parameters.json')):
        spec=read(p);name=spec['name']
        if name not in expected:continue
        assert all(spec[k]==v for k,v in expected[name].items())
        folder=p.parent;path=folder/'episode.hdf5';validation=validate(path);samples=read(folder/'samples.json')
        label=settled_label(samples);column=settled_label(samples,field='cup_column_height_fraction')
        row=dict(name=name,path=str(path),sha256=sha(path),bytes=path.stat().st_size,split=spec['study_split'],
                 label=label,column_label=column,recording_validation=validation,
                 sidecars={f.name:sha(f) for f in folder.glob('*.json')},trajectory_sha256=sha(folder/'trajectory.npz'))
        row['containment']=dict(final_outside_ml=samples[-1]['spilled_ml'],final_airborne_ml=samples[-1]['airborne_ml'],
            ever_table_ml=max(r['ever_table_ml'] for r in samples))
        if spec.get('calibration_motion'):
            xyz=np.asarray([r['bottle_pose_xyz'] for r in samples]);tilt=np.asarray([r['bottle_tilt_deg'] for r in samples])
            row['motion']=dict(maximum_translation_m=float(np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1).max()),
                rotation_range_deg=float(np.ptp(tilt)),cup_inventory_range_ml=float(np.ptp([r['cup_ml'] for r in samples])),
                bottle_inventory_range_ml=float(np.ptp([r['bottle_ml'] for r in samples])))
            row['motion']['passed']=row['motion']['maximum_translation_m']>.02 and row['motion']['rotation_range_deg']>15.
        with h5py.File(path) as f:
            d=f['data/demo_0'];row['cameras']={};canvas=Image.new('RGB',(1536,408),(245,245,245));draw=ImageDraw.Draw(canvas)
            for i,camera in enumerate(('task','wrist','cup_detail','overview')):
                image=decode(d['next_obs/'+camera+'_image'][-1],'jpeg95')
                cal=dict(K=f['calibration/'+camera+'/intrinsics'][:].tolist(),T=d['next_obs/'+camera+'_world_T_camera'][-1].tolist(),table_z=1.04)
                row['cameras'][camera]=dict(calibration=cal,resolution=[image.shape[1],image.shape[0]])
                canvas.paste(Image.fromarray(image).resize((384,384)),(i*384,24));draw.text((i*384+6,6),name+' | '+camera,fill=(20,20,20))
            view=out/(name+'.jpg');canvas.save(view);row['review_path']=str(view);row['review_sha256']=sha(view)
            if spec.get('calibration_motion'):
                canvas=Image.new('RGB',(1152,816),(245,245,245));draw=ImageDraw.Draw(canvas)
                for j,t in enumerate((.5,.9,1.3,1.7,2.1,2.5)):
                    index=int(np.abs(d['next_time'][:]-t).argmin());rgb=decode(d['next_obs/cup_detail_image'][index],'jpeg95')
                    x=j%3*384;y=j//3*408;canvas.paste(Image.fromarray(rgb).resize((384,384)),(x,y+24))
                    draw.text((x+6,y+6),name+' | t='+str(round(float(d['next_time'][index]),3)),fill=(20,20,20))
                view=out/(name+'-motion.jpg');canvas.save(view);row.update(motion_review_path=str(view),motion_review_sha256=sha(view))
        if a.part.startswith('seed_'):
            (ROOT/'seeds').mkdir(exist_ok=True)
            trajectory=np.load(folder/'trajectory.npz');last=len(trajectory['times'])-1
            points=trajectory[f'positions_{last:03}'];local=(points-np.asarray(spec['cup_world']))@FRAME
            inside=vessel_mask(local,cup_profile(spec));cup=local[inside].astype(np.float32)
            destination=ROOT/'seeds'/(name+'-settled.npz')
            assert not destination.exists()
            np.savez_compressed(destination,cup_local_points=cup)
            row.update(settled_seed_path=str(destination),settled_seed_sha256=sha(destination),settled_seed_particles=len(cup),
                       high_fill_coverage=.94<=label['target']<=.98)
        records.append(row);print(json.dumps(dict(name=name,label=label,coverage=row.get('high_fill_coverage'))),flush=True)
    assert len(records)==len(expected) and len({r['name'] for r in records})==len(expected)
    result=dict(part=a.part,records=records,complete=True,protected_parent_files_unchanged=True,
                labels_accepted=sum(r['label']['accepted'] for r in records),manual_review='pending',
                protocol_sha256=sha(ROOT/'protocol.json'),plan_sha256=m['plan_sha256'],checkpoint_sha256=m['checkpoint_sha256'],helper_sha256=sha(Path(__file__)),
                scope='Authored perception calibration and physical settling; bottle/hand motion is measured where declared. Not autonomous pouring or a visual-size estimator.')
    destination=ROOT/(a.part+'-'+a.round+'-audit.json')
    with destination.open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()
