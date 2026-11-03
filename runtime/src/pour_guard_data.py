"""Offline RGB labels/calibration; reserved G_A scenes are never fitted."""
import json
from pathlib import Path
import numpy as np
import torch
from pour_stop_data import h5py
from validate_pour_recording import decode
from pour_guard_study import ROOT
from pour_guard_vision import project,cup_crop


def build_data():
    ROOT.mkdir(parents=True,exist_ok=True);cache=ROOT/'cup-cache.pt'
    if cache.exists():return torch.load(cache,map_location='cpu',weights_only=False)
    arrays={s:{k:[] for k in ('image','fill','calm','good')} for s in ('train','validation')}
    initial={s:[] for s in arrays};manifest=[];cal=None
    for p in sorted(Path('/workspace/outputs/fountain').glob('*/*/episode.hdf5')):
        with h5py.File(p,'r') as f:
            spec=json.loads(f.attrs['spec']);name=spec.get('name','');split=spec.get('study_split')
            if (any(prefix in name for prefix in ('G_A_','H_A_')) or split not in ('train','validation','test') or
                spec.get('spacing')!=.0018 or spec.get('dt')!=.0025 or f.attrs.get('status')!='complete' or
                f.attrs.get('camera_rig')!='A_task_wrist_v4' or f.attrs.get('image_encoding')!='jpeg95'):continue
            dest='validation' if split=='validation' else 'train';d=f['data/demo_0'];n=len(d['actions'])
            T=d['obs/task_world_T_camera'][0];K=f['calibration/task/intrinsics'][:]
            if cal is None:cal=dict(T=T.tolist(),K=K.tolist(),table_z=1.04)
            assert np.allclose(T,cal['T'],atol=1e-5) and np.allclose(K,cal['K'],atol=1e-5)
            image=decode(d['obs/task_image'][0],'jpeg95');base=d['ground_truth/bottle_pose'][0,:3]
            cup=np.array(spec['cup_world']);initial[dest].append(dict(name=name,image=image,
                source_uv=project(base+[0,0,.135],cal),cup_uv=project(cup+[0,0,.038],cal),
                source_xy=base[:2],cup_xy=cup[:2]))
            rows=json.loads((p.parent/'samples.json').read_text());m=json.loads(f.attrs['metrics'])
            good=bool(m.get('zero_spill'));used=0
            for i in range(0,n,3):
                row=rows[i];height=float(row['cup_fill_fraction'])
                if not good and height>=.65:continue
                a=cup_crop(decode(d['next_obs/task_image'][i],'jpeg95'),cup[:2],cal)
                raw=a.astype(float);red=(raw[:,:,0]>1.4*raw[:,:,1]+6)&(raw[:,:,0]>1.2*raw[:,:,2]+5)
                if height>.15 and red.sum()<12:continue
                calm=row['t']<3.2 or (row['stop_time'] is not None and row['t']>row['stop_time']+1.25)
                for key,value in zip(arrays[dest],(a,np.clip(height,0,1),float(calm),float(good))):arrays[dest][key].append(value)
                used+=1
            manifest.append(dict(name=name,path=str(p),original_split=split,fit_split=dest,frames=used,
                zero_spill=good,used_consumed_test=split=='test'))
    data={s:{k:torch.from_numpy(np.asarray(v,dtype=np.uint8 if k=='image' else np.float32)) for k,v in a.items()} for s,a in arrays.items()}
    result=dict(data=data,initial=initial,calibration=cal,manifest=manifest)
    torch.save(result,cache);(ROOT/'data-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return result
