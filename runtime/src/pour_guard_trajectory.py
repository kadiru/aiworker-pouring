"""CUDA ridge imitation conditioned on own lift pose and visually located cup."""
import json
from pathlib import Path
import numpy as np
import torch
from pour_stop_data import h5py
from pour_guard_study import ROOT


def fit():
    entries=json.loads(Path('/workspace/learning/A-recovery-20261102-v1/dataset-manifest.json').read_text())
    episodes={'train':[],'validation':[]};rates=[];closure=[]
    for e in entries:
        if not e.get('accepted') or e.get('corrective_prefix') or e.get('expert_recovery'):continue
        with h5py.File(e['path'],'r') as f:
            d=f['data/demo_0'];spec=json.loads(f.attrs['spec']);phase=np.array([x.decode() for x in d['ground_truth/phase'][:]])
            ix=np.flatnonzero((phase=='Pouring from gripper')&~d['ground_truth/stop_command'][:].astype(bool))
            time=d['time'][:];lift=d['obs/joint_pos'][np.argmin(abs(time-3.25)),:7]
            angle=d['ground_truth/command_angle_deg'][:][ix];q=d['actions'][:][ix,:7]
            unique=np.r_[True,np.diff(angle)>1e-4];angle=angle[unique];q=q[unique]
            delta=np.stack([np.interp(np.arange(166),angle,q[:,j])-lift[j] for j in range(7)],-1)
            context=np.r_[lift,spec['cup_world'][:2]]
            episodes[e['split']].append(dict(name=e['name'],context=context,lift=lift,angle=angle,action=q,delta=delta,cap=angle[-1]))
            if e['split']=='train':
                jq=d['obs/joint_pos'][:];held=(time>=2.8)&(time<=3.2)&(d['ground_truth/grasp_verified'][:]>0)
                closure.extend((np.pi/4-jq[held,7:][:,[0,2]]).reshape(-1).tolist())
                raw=d['ground_truth/command_angle_deg'][:];r=np.diff(raw)/np.diff(time)
                rates.extend((float(a),float(v)) for a,v in zip(raw[1:],r) if .1<v<50)
    train=episodes['train'];assert len(train)==19
    raw=torch.tensor(np.stack([e['context'] for e in train]),device='cuda',dtype=torch.float32)
    mean=raw.mean(0);std=raw.std(0).clamp(min=.001);std[:7]=std[:7].clamp(min=.02)
    x=torch.cat([torch.ones((len(raw),1),device='cuda'),(raw-mean)/std],1)
    y=torch.tensor(np.stack([e['delta'] for e in train]),device='cuda',dtype=torch.float32)
    caps=torch.tensor([e['cap'] for e in train],device='cuda');candidates=[]
    for ridge in (.001,.01,.1,1.,10.):
        coeff=[];support=[];penalty=torch.eye(10,device='cuda')*ridge;penalty[0,0]=.0001
        for angle in range(166):
            mask=caps>=angle-1;support.append(int(mask.sum()))
            if int(mask.sum())<3:coeff.append(coeff[-1]);continue
            a=x[mask];b=y[mask,angle];coeff.append(torch.linalg.solve(a.T@a+penalty,a.T@b))
        weights=torch.stack(coeff).cpu().numpy();errors=[]
        for e in episodes['validation']:
            context=np.r_[1.,(e['context']-mean.cpu().numpy())/std.cpu().numpy()]
            pred=np.array([e['lift']+context@weights[int(np.clip(round(a),0,165))] for a in e['angle']])
            errors.extend((pred-e['action']).tolist())
        err=np.array(errors);candidates.append(dict(ridge=ridge,mse=float(np.mean(err**2)),rmse_deg=np.degrees(np.sqrt(np.mean(err**2,axis=0))).tolist(),weights=weights.tolist(),support=support))
    chosen=min(candidates,key=lambda e:e['mse']);r=np.array(rates)
    gaps=torch.tensor(closure,device='cuda');minimum=float(torch.quantile(gaps,.01))
    result=dict(method='Contextual movement primitive, own lift joints plus visual cup XY; CUDA ridge imitation',
        weights=chosen['weights'],mean=mean.cpu().tolist(),std=std.cpu().tolist(),regularizer=chosen['ridge'],support=chosen['support'],
        validation_target_rmse_deg=chosen['rmse_deg'],regularizer_validation=[{k:v for k,v in c.items() if k not in ('weights','support')} for c in candidates],
        fast_rate_deg_s=float(np.median(r[r[:,0]<82,1])),slow_rate_deg_s=float(np.median(r[r[:,0]>=82,1])),
        phase_cap_deg=float(max(e['cap'] for e in train)),train_episodes=[e['name'] for e in train],
        validation_episodes=[e['name'] for e in episodes['validation']],training_device=torch.cuda.get_device_name(0),
        grasp_closure=dict(min_deficit_rad=float(np.clip(.5*minimum,np.radians(1.),np.radians(4.))),
            positive_p01_deficit_rad=minimum,positive_finger_samples=len(closure),method='Half the first percentile of observed opposing-finger closure deficits in verified expert holds'))
    (ROOT/'pour-primitive-fit.json').write_text(json.dumps(result,indent=2)+'\n');return result


def target(profile,lift,cup_xy,phase):
    context=np.r_[1.,(np.r_[lift,cup_xy]-np.array(profile['mean']))/np.array(profile['std'])]
    weights=np.asarray(profile['weights']);i=int(np.clip(phase,0,164));u=np.clip(phase-i,0,1)
    return lift+context@((1-u)*weights[i]+u*weights[i+1])
