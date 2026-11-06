"""CPU-only diagnosis of closed reference/development recordings; never tests."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,'/workspace/src')
import numpy as np
import torch
from validate_pour_recording import h5py,decode
from pour_guard_vision import locate,plane_point,project
from pour_source_alignment import aligned_crops
from pour_stop_data import crops
from pour_stop_models import LiquidDetector
from pour_guard_actor import held_gripper

ROOT=Path('/workspace/learning/A-wide-20261105-v1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    assert not (ROOT/'final-selection.json').exists();torch.set_num_threads(2);started=time.monotonic()
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    model=LiquidDetector().eval();model.load_state_dict(saved['source_detector'])
    accepted=set(json.loads((ROOT/'reference-acceptance.json').read_text())['accepted']);episodes=[]
    paths=sorted(Path('/workspace/outputs/fountain').glob('*/*T30_A_demo_*/episode.hdf5'))+sorted(Path('/workspace/outputs/fountain').glob('*/*XW30_A_dev_*_r0*/episode.hdf5'))
    with torch.inference_mode():
        for path in paths:
            rows=json.loads((path.parent/'samples.json').read_text())
            with h5py.File(path) as f:
                spec=json.loads(f.attrs['spec']);name=spec['name']
                if spec.get('protocol')!=17 or (name.startswith('T30') and name not in accepted):continue
                assert spec['study_split']!='test' and f.attrs['status']=='complete'
                d=f['data/demo_0'];image=decode(d['obs/task_image'][0],'jpeg95');cal=saved['calibration']
                uv,score=locate(image,saved['cup_bank'],device='cpu');xy=plane_point(uv,1.078,cal)[:2]
                true=project(np.array(spec['cup_world'])+[0,0,.038],cal);x,y=np.rint(true).astype(int)
                erased=image.copy();erased[max(0,y-18):min(384,y+22),max(0,x-44):min(384,x+44)]=180
                _,negative=locate(erased,saved['cup_bank'],device='cpu')
                result=dict(name=name,split=spec['study_split'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    cup_localization=dict(uv=uv.tolist(),score=score,error_mm=float(np.linalg.norm(xy-np.asarray(spec['cup_world'])[:2])*1000),synthetic_occlusion_score=negative),presence=[])
                ids=[i for i,r in enumerate(rows) if i%10==0 and r['grasp_verified'] and r['both_fingers']]
                for begin in range(0,len(ids),32):
                    indices=ids[begin:begin+32];raw=[];aligned=[]
                    for i in indices:
                        task=decode(d['obs/task_image'][i],'jpeg95');wrist=decode(d['obs/wrist_image'][i],'jpeg95')
                        raw.append(crops(task,wrist));aligned.append(aligned_crops(task,wrist,xy,cal))
                    if not indices:continue
                    rp=model(*[torch.as_tensor(np.stack(v)) for v in zip(*raw)])['present_logit'].sigmoid()
                    ap=model(*[torch.as_tensor(np.stack(v)) for v in zip(*aligned)])['present_logit'].sigmoid()
                    for j,i in enumerate(indices):
                        q=d['obs/joint_pos'][i]
                        result['presence'].append(dict(index=i,t=rows[i]['t'],tilt=rows[i]['bottle_tilt_deg'],raw=float(rp[j]),aligned=float(ap[j]),fused=max(float(rp[j]),float(ap[j])),encoder_hold=held_gripper(q,saved['pour_primitive']['grasp_closure']['min_deficit_rad'])))
                episodes.append(result);print(name,'cup_error',round(result['cup_localization']['error_mm'],3),'cup_score',round(score,3),'presence_samples',len(ids),flush=True)
    report=dict(checkpoint_sha256=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest(),device='cpu',cuda_initialized=torch.cuda.is_initialized(),episodes=episodes,elapsed_s=time.monotonic()-started,scope='Consumed reference/development diagnostics only. Fixed model, no fitting/deployment, no reserved test data. Synthetic cup erasure is not real missing-object validation.')
    assert not report['cuda_initialized'];a.output.write_text(json.dumps(report,indent=2)+'\n');print('CPU_DIAGNOSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
