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
ROOT = Path('/workspace/learning/A-scale-20261109-v1')


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=('pilot','collect','holdout'),required=True)
    parser.add_argument('--round',choices=('r0','r1','r2'),default='r0');a=parser.parse_args()
    m=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');expected={s['name']:s for s in plan[a.part]}
    assert sha(ROOT/'plan.json')==m['plan_sha256']
    assert all(sha(Path(p))==h for p,h in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==h for n,h in m['runtime_sources'].items())
    if a.part=='pilot':assert plan.get('pilot_round','r0')==a.round
    out=ROOT/(a.part+'-'+a.round+'-review');out.mkdir(exist_ok=True);records=[]
    for p in sorted(Path('/workspace/outputs/fountain').glob('*/*SC_A_*/parameters.json')):
        spec=read(p);name=spec['name']
        if name not in expected:continue
        assert all(spec[k]==v for k,v in expected[name].items())
        folder=p.parent;path=folder/'episode.hdf5';validation=validate(path);samples=read(folder/'samples.json')
        label=settled_label(samples);column=settled_label(samples,field='cup_column_height_fraction')
        row=dict(name=name,path=str(path),sha256=sha(path),bytes=path.stat().st_size,split=spec['study_split'],
                 label=label,column_label=column,recording_validation=validation,
                 sidecars={f.name:sha(f) for f in folder.glob('*.json')},trajectory_sha256=sha(folder/'trajectory.npz'))
        with h5py.File(path) as f:
            d=f['data/demo_0'];row['cameras']={};canvas=Image.new('RGB',(1536,408),(245,245,245));draw=ImageDraw.Draw(canvas)
            for i,camera in enumerate(('task','wrist','cup_detail','overview')):
                image=decode(d['next_obs/'+camera+'_image'][-1],'jpeg95')
                cal=dict(K=f['calibration/'+camera+'/intrinsics'][:].tolist(),T=d['next_obs/'+camera+'_world_T_camera'][-1].tolist(),table_z=1.04)
                row['cameras'][camera]=dict(calibration=cal,resolution=[image.shape[1],image.shape[0]])
                canvas.paste(Image.fromarray(image).resize((384,384)),(i*384,24));draw.text((i*384+6,6),name+' | '+camera,fill=(20,20,20))
            view=out/(name+'.jpg');canvas.save(view);row['review_path']=str(view);row['review_sha256']=sha(view)
        if a.part=='pilot':
            trajectory=np.load(folder/'trajectory.npz');last=len(trajectory['times'])-1
            points=trajectory[f'positions_{last:03}'];local=(points-np.asarray(spec['cup_world']))@FRAME
            inside=vessel_mask(local,cup_profile(spec));cup=local[inside].astype(np.float32)
            destination=ROOT/'seeds'/(name+'-settled.npz')
            assert not destination.exists()
            np.savez_compressed(destination,cup_local_points=cup)
            row.update(settled_seed_path=str(destination),settled_seed_sha256=sha(destination),settled_seed_particles=len(cup),
                       high_fill_coverage=.85<=label['target']<=.98)
        records.append(row);print(json.dumps(dict(name=name,label=label,coverage=row.get('high_fill_coverage'))),flush=True)
    assert len(records)==len(expected) and len({r['name'] for r in records})==len(expected)
    result=dict(part=a.part,records=records,complete=True,protected_parent_files_unchanged=True,
                labels_accepted=sum(r['label']['accepted'] for r in records),manual_review='pending',
                plan_sha256=m['plan_sha256'],checkpoint_sha256=m['checkpoint_sha256'],helper_sha256=sha(Path(__file__)),
                scope='Stationary authored calibration and physical settling, not autonomous pouring or a visual-size estimator.')
    destination=ROOT/(a.part+'-'+a.round+'-audit.json')
    with destination.open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()
