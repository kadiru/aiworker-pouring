"""Contextual movement primitive, fitted from successful expert joint paths.

Offline angle labels align demonstrated paths. Runtime advances its own phase;
there is no simulator-angle/pose/inventory feedback. Ridge fitting runs on4070.
"""
import json
from pathlib import Path
import numpy as np
import torch
from pour_stop_study import ROOT
from pour_stop_data import h5py


def fit():
    entries=json.loads(Path('/workspace/learning/A-recovery-20261102-v1/dataset-manifest.json').read_text())
    episodes={'train':[],'validation':[]};rates=[]
    for e in entries:
        if not e.get('accepted') or e.get('corrective_prefix') or e.get('expert_recovery'):continue
        with h5py.File(e['path'],'r') as f:
            d=f['data/demo_0'];ph=np.array([x.decode() for x in d['ground_truth/phase'][:]])
            # Keep the consistent original registered-grasp teacher family.
            take=(ph=='Pouring from gripper')&~d['ground_truth/stop_command'][:].astype(bool)
            ix=np.flatnonzero(take)
            if len(ix)<10:continue
            time=d['time'][:];qlift=d['obs/joint_pos'][np.argmin(abs(time-3.25)),:7]
            angle=d['ground_truth/command_angle_deg'][:][ix];q=d['actions'][:][ix,:7]
            unique=np.r_[True,np.diff(angle)>1e-4];angle=angle[unique];q=q[unique]
            if len(angle)<5:continue
            grid=np.arange(166,dtype=float)
            delta=np.stack([np.interp(grid,angle,q[:,j])-qlift[j] for j in range(7)],-1)
            episodes[e['split']].append(dict(name=e['name'],qlift=qlift,delta=delta,max_angle=float(angle[-1]),
                action=q,angles=angle))
            if e['split']=='train':
                raw=d['ground_truth/command_angle_deg'][:];r=np.diff(raw)/np.diff(time)
                rates.extend([(float(a),float(v)) for a,v in zip(raw[1:],r) if .1<v<50])
    train=episodes['train'];assert len(train)>=15
    lifts=torch.tensor(np.stack([e['qlift'] for e in train]),device='cuda')
    mean=lifts.mean(0);std=lifts.std(0).clamp(min=.02)
    x=torch.cat([torch.ones((len(lifts),1),device='cuda'),(lifts-mean)/std],-1)
    y=torch.tensor(np.stack([e['delta'] for e in train]),device='cuda',dtype=torch.float32)
    caps=torch.tensor([e['max_angle'] for e in train],device='cuda')
    candidates=[]
    for regularizer in (.1,1.,10.,100.):
        coeff=[];support=[]
        penalty=torch.eye(8,device='cuda')*regularizer;penalty[0,0]=.001
        for angle in range(166):
            valid=caps>=angle-1
            if int(valid.sum())<3:
                coeff.append(coeff[-1]);support.append(int(valid.sum()));continue
            a=x[valid];b=y[valid,angle]
            coeff.append(torch.linalg.solve(a.T@a+penalty,a.T@b));support.append(int(valid.sum()))
        weights=torch.stack(coeff).cpu().numpy();validation=[]
        for e in episodes['validation']:
            context=np.r_[1.,(e['qlift']-mean.cpu().numpy())/std.cpu().numpy()]
            predicted=np.array([e['qlift']+context@weights[int(np.clip(round(a),0,165))] for a in e['angles']])
            validation.extend((predicted-e['action']).tolist())
        error=np.array(validation)
        candidates.append(dict(regularizer=regularizer,weights=weights,support=support,mse=float(np.mean(error**2)),
            rmse_deg=np.degrees(np.sqrt(np.mean(error**2,axis=0))).tolist()))
    chosen=min(candidates,key=lambda c:c['mse']);weights=chosen['weights'];support=chosen['support']
    r=np.asarray(rates);fast=float(np.median(r[r[:,0]<82,1]));slow=float(np.median(r[r[:,0]>=82,1]))
    result=dict(method='Contextual joint-space movement primitive; per-phase CUDA ridge regression',regularizer=chosen['regularizer'],
        regularizer_validation=[{k:v for k,v in c.items() if k not in ('weights','support')} for c in candidates],
        weights=weights.tolist(),mean=mean.cpu().tolist(),std=std.cpu().tolist(),support=support,
        fast_rate_deg_s=fast,slow_rate_deg_s=slow,phase_cap_deg=float(max(e['max_angle'] for e in train)),
        train_episodes=[e['name'] for e in train],validation_episodes=[e['name'] for e in episodes['validation']],
        validation_target_rmse_deg=chosen['rmse_deg'],
        runtime_inputs='own measured lift joints, own scalar phase; visual flow only advances/holds phase',
        training_device=torch.cuda.get_device_name(0))
    (ROOT/'pour-primitive-fit.json').write_text(json.dumps(result,indent=2)+'\n');return result


def target(profile,qlift,phase):
    context=np.r_[1.,(qlift-np.array(profile['mean']))/np.array(profile['std'])]
    weights=np.asarray(profile['weights']);i=int(np.clip(phase,0,164));u=np.clip(phase-i,0,1)
    return qlift+context@((1-u)*weights[i]+u*weights[i+1])
