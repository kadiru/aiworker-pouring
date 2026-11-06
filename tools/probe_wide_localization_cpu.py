"""Partial-reference diagnostic on CPU; never deploys a bank or uses tests."""
import argparse, hashlib, json, sys
from pathlib import Path

STUDY=Path('/workspace/learning/A-wide-20261105-v1')
sys.path[:0]=[str(STUDY/'staged'),'/workspace/src']
import numpy as np
import torch
from validate_pour_recording import h5py,decode
from pour_guard_vision import make_bank,locate,project,plane_point,bottle_base_from_neck


def main(output):
    assert not output.exists()
    torch.set_num_threads(1);entries=[]
    for path in sorted(Path('/workspace/outputs/fountain').glob('*/*T30_A_demo_*/episode.hdf5')):
        with h5py.File(path) as f:
            spec=json.loads(f.attrs['spec']);m=json.loads(f.attrs['metrics'])
            if spec.get('protocol')!=17 or not(m['grasp_verified'] and m['zero_spill'] and not m['gates']):continue
            assert f.attrs['status']=='complete' and not spec['policy_evaluation']
            d=f['data/demo_0'];cal=dict(K=f['calibration/task/intrinsics'][:],T=d['obs/task_world_T_camera'][0],table_z=1.04)
            entries.append(dict(name=spec['name'],split=spec['study_split'],image=decode(d['obs/task_image'][0],'jpeg95'),
                source_xyz=d['ground_truth/bottle_pose'][0,:3],cup_xyz=np.asarray(spec['cup_world']),cal=cal))
    train=[e for e in entries if e['split']=='train'];rows=[];candidates=[]
    with torch.inference_mode():
        for kind,height in [('source',.135)]:
            anchors=[project(e[kind+'_xyz']+[0,0,height],e['cal']) for e in train]
            for absolute in (True,False):
                for radius in (32.,48.):
                    for profile,box in [('original',None),('neck28',(10,8,38,40)),('neck24',(12,8,36,36)),('neck20',(14,8,34,34))]:
                        bank=make_bank([e['image'] for e in train],anchors,kind,device='cpu',aggregation='max',absolute_gradients=absolute,anchor_radius_px=radius)
                        bank['bounds']=[24,24,360,350]
                        if box:
                            x0,y0,x1,y1=box;keep=torch.zeros_like(bank['mask']);keep[:,:,y0:y1,x0:x1]=1
                            kernels=bank['kernels'];weighted=kernels/bank['mask'].clamp(min=1e-12)
                            norm=(weighted.square()*keep).sum((1,2,3),keepdim=True).sqrt().clamp(min=1e-6)
                            bank['kernels']=kernels*keep/norm;bank['mask']*=keep
                        rows=[]
                        for e in entries:
                            uv,score=locate(e['image'],bank,device='cpu')
                            xy=bottle_base_from_neck(uv,e['cal'])[:2]
                            negative=e['image'].copy();x,y=np.rint(project(e[kind+'_xyz']+[0,0,height],e['cal'])).astype(int)
                            negative[max(0,y-18):min(384,y+44),max(0,x-28):min(384,x+28)]=180
                            _,ns=locate(negative,bank,device='cpu')
                            rows.append(dict(name=e['name'],split=e['split'],score=score,synthetic_negative_score=ns,uv=uv.tolist(),
                                error_mm=float(np.linalg.norm(xy-e[kind+'_xyz'][:2])*1000)))
                        candidate=dict(absolute_gradients=absolute,anchor_radius_px=radius,profile=profile,mask_box=box,rows=rows,
                            max_error_mm=max(r['error_mm'] for r in rows),score_gap=min(r['score'] for r in rows)-max(r['synthetic_negative_score'] for r in rows))
                        candidates.append(candidate);print(json.dumps({k:v for k,v in candidate.items() if k!='rows'}),flush=True)
    report=dict(scope='Partial collected references, CPU diagnostic only. Training rows are resubstitution, not held-out performance. Temporary template bank only; no candidate checkpoint written/deployed and no test images.',
        template_names=[e['name'] for e in train],candidates=candidates,source_sha256=hashlib.sha256((STUDY/'staged/pour_guard_vision.py').read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
