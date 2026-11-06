"""Refit a contextual joint movement primitive using expanded-position demos."""
import json
from pathlib import Path
import numpy as np
import torch
from validate_pour_recording import h5py
from pour_workspace_study import ROOT,PREFIX


@torch.inference_mode()
def fit(parent,sequences,root=ROOT,prefix=PREFIX):
    prior=json.loads(Path('/workspace/learning/A-recovery-20261102-v1/dataset-manifest.json').read_text())
    entries=[dict(name=e['name'],path=e['path'],split=e['split']) for e in prior if e.get('accepted') and not e.get('corrective_prefix') and not e.get('expert_recovery')]
    entries.extend(dict(name=e['name'],path=e['path'],split=e['split']) for e in sequences if e['name'].startswith(prefix) and e['accepted_reference'])
    episodes={'train':[],'validation':[]}
    for e in entries:
        with h5py.File(e['path'],'r') as f:
            d=f['data/demo_0'];spec=json.loads(f.attrs['spec']);phase=np.array([x.decode() for x in d['ground_truth/phase'][:]])
            ix=np.flatnonzero((phase=='Pouring from gripper')&~d['ground_truth/stop_command'][:].astype(bool))
            assert len(ix)>5,e['name']
            lift_index=(max(0,int(ix[0])-1) if e['name'].startswith('T30_A_') else np.argmin(abs(d['time'][:]-3.25)))
            lift=d['obs/joint_pos'][lift_index,:7]
            angle=d['ground_truth/command_angle_deg'][:][ix];q=d['actions'][:][ix,:7]
            unique=np.r_[True,np.diff(angle)>1e-4];angle=angle[unique];q=q[unique]
            delta=np.stack([np.interp(np.arange(166),angle,q[:,j])-lift[j] for j in range(7)],-1)
            episodes[e['split']].append(dict(name=e['name'],context=np.r_[lift,spec['cup_world'][:2]],lift=lift,angle=angle,action=q,delta=delta,cap=angle[-1]))
    train=episodes['train'];assert len(train)>=31 and len(episodes['validation'])>=6
    raw=torch.tensor(np.stack([e['context'] for e in train]),device='cuda',dtype=torch.float64)
    mean=raw.mean(0);std=raw.std(0).clamp(min=.001);std[:7]=std[:7].clamp(min=.02)
    x=torch.cat([torch.ones((len(raw),1),device='cuda'),(raw-mean)/std],1)
    y=torch.tensor(np.stack([e['delta'] for e in train]),device='cuda',dtype=torch.float64)
    caps=torch.tensor([e['cap'] for e in train],device='cuda');candidates=[]
    for ridge in (.0001,.001,.01,.1,1.):
        coeff=[];support=[];penalty=torch.eye(10,device='cuda',dtype=torch.float64)*ridge;penalty[0,0]=1e-7
        for angle in range(166):
            mask=caps>=angle-1;support.append(int(mask.sum()))
            if int(mask.sum())<3:coeff.append(coeff[-1]);continue
            a=x[mask];b=y[mask,angle];coeff.append(torch.linalg.solve(a.T@a+penalty,a.T@b))
        weights=torch.stack(coeff).cpu().numpy();errors=[];per=[]
        for e in episodes['validation']:
            context=np.r_[1.,(e['context']-mean.cpu().numpy())/std.cpu().numpy()]
            angles=e['angle'].clip(0,parent['phase_cap_deg']);indices=np.floor(angles).astype(int).clip(0,164);u=angles-indices
            pred=e['lift']+np.einsum('i,nij->nj',context,(1-u[:,None,None])*weights[indices]+u[:,None,None]*weights[indices+1])
            valid=e['angle']<=parent['phase_cap_deg'];err=pred[valid]-e['action'][valid]
            errors.extend(err.tolist());per.append(dict(name=e['name'],mse=float(np.mean(err**2)),rmse_deg=float(np.degrees(np.sqrt(np.mean(err**2))))))
        candidates.append(dict(ridge=ridge,episode_balanced_mse=float(np.mean([e['mse'] for e in per])),
            rmse_deg=np.degrees(np.sqrt(np.mean(np.asarray(errors)**2,axis=0))).tolist(),episodes=per,weights=weights.tolist(),support=support))
    best=min(candidates,key=lambda x:x['episode_balanced_mse'])
    result=dict(parent,weights=best['weights'],mean=mean.cpu().tolist(),std=std.cpu().tolist(),regularizer=best['ridge'],support=best['support'],
        method='CUDA ridge behavioral cloning of joint targets conditioned on own lift joints and visual cup XY; expanded-position reference demonstrations plus original successful demonstrations',
        train_episodes=[e['name'] for e in train],validation_episodes=[e['name'] for e in episodes['validation']],
        validation_target_rmse_deg=best['rmse_deg'],regularizer_validation=[{k:v for k,v in c.items() if k not in ('weights','support')} for c in candidates],
        fast_rate_deg_s=30.,slow_rate_deg_s=5.,training_device=torch.cuda.get_device_name(0),
        context_dimensions=10,phase_grid_points=166,serialized_weight_entries=166*10*7)
    (root/'pour-primitive-fit.json').write_text(json.dumps(result,indent=2)+'\n')
    metadata={k:v for k,v in result.items() if k not in ('weights','mean','std')}
    (root/'motion-summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return result,metadata
