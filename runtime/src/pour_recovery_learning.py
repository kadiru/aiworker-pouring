"""Full-sequence corrective BC with an unlabeled learner-visited history prefix."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pour_learning import Policy, pixels, PHASES, CHUNK, loss
from pour_recovery_study import ROOT
from pour_study import ROOT as ORIGINAL_ROOT

SEED=20261108


def load_episodes():
    import sys
    sys.path.append('/workspace/pydeps51')
    import h5py
    from validate_pour_recording import validate
    frozen=json.loads((ROOT/'frozen-plan.json').read_text())
    allowed={s['name']:split for split in ('train','validation') for s in frozen[split]}
    latest={}
    for path in sorted(Path('/workspace/outputs/fountain').glob('*/*/episode.hdf5')):
        with h5py.File(path,'r') as f:
            spec=json.loads(f.attrs['spec'])
        if spec['name'] in allowed:latest[spec['name']]=path
    data={'train':[],'validation':[]};manifest=[]
    for name,path in sorted(latest.items()):
        report=validate(path);metrics=report['metrics']
        with h5py.File(path,'r') as f:
            spec=json.loads(f.attrs['spec']);prefix=bool(spec.get('corrective_prefix'))
            recovery=bool(spec.get('expert_recovery'))
            if recovery:accepted=report['recovery_training_eligible']
            elif prefix:accepted=report['corrective_training_eligible']
            else:accepted=report['training_eligible'] and (allowed[name]!='train' or metrics['zero_spill'])
            row=dict(name=name,path=str(path),split=allowed[name],accepted=bool(accepted),
                     corrective_prefix=prefix,expert_recovery=recovery)
            manifest.append(row)
            if not accepted:continue
            d=f['data/demo_0'];codec=f.attrs.get('image_encoding','raw');n=len(d['actions'])
            e={c:torch.from_numpy(np.stack([pixels(frame,codec) for frame in d['obs/'+c+'_image']]))
               for c in ('task','wrist')}
            for dest,key in [('q','obs/joint_pos'),('v','obs/joint_vel'),('a','teacher_actions' if 'teacher_actions' in d else 'actions')]:
                e[dest]=torch.from_numpy(d[key][:].astype(np.float32))
            reasons=[x.decode() if isinstance(x,bytes) else str(x) for x in d['ground_truth/stop_reason'][:]]
            phases=[x.decode() if isinstance(x,bytes) else str(x) for x in d['ground_truth/phase'][:]]
            e['stop']=torch.tensor([{'cup_near_full':1,'bottle_empty':2}.get(x,0) for x in reasons])
            e['phase']=torch.tensor([PHASES.index(x) if x in PHASES else 0 for x in phases])
            # Supervision corresponds to the preceding observation, not next_obs.
            cup=np.r_[spec.get('cup_initial_ml',0.)/72.8,d['ground_truth/cup_inventory_fraction'][:-1]]
            source=np.r_[spec['initial_ml'],d['ground_truth/bottle_ml'][:-1]]/100.
            e['fill']=torch.tensor(np.stack([cup,source],-1),dtype=torch.float32)
            e['done']=torch.zeros(n)
            if not prefix:e['done'][-5:]=1.
            e['mask']=torch.from_numpy(d['expert_supervision'][:].astype(bool)) if recovery else torch.ones(n,dtype=torch.bool)
            indices=torch.arange(n)[:,None]+torch.arange(CHUNK)[None,:]
            valid=indices<n;indices=indices.clamp(max=n-1)
            e['future_valid']=valid&e['mask'][indices]
            e['future_a']=e['a'][indices];e['future_stop']=e['stop'][indices]
            e.update(n=n,path=str(path),corrective=prefix,recovery=recovery)
            data[allowed[name]].append(e)
            row.update(transitions=n,supervised_transitions=int(e['mask'].sum()),
                       supervision='expert suffix only' if recovery else 'all transitions')
    (ROOT/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert sum(e['recovery'] for e in data['train'])>=2,'Need at least two reviewed zero-spill recoveries'
    assert any(e['recovery'] for e in data['validation']),'Need a reviewed independent recovery validation episode'
    return data


def full_batch(e,augment=False):
    b={k:e[k][None].cuda() for k in ('task','wrist','q','v','a','stop','phase','fill','done','mask','future_valid','future_a','future_stop')}
    if augment:
        for c in ('task','wrist'):
            x=b[c].float();gain=torch.empty((),device='cuda').uniform_(.97,1.03)
            b[c]=(x*gain+torch.randn_like(x)*1.5).clamp(0,255)
        b['q']=b['q']+torch.randn_like(b['q'])*.003
        b['v']=b['v']+torch.randn_like(b['v'])*.01
    return b


def train(state,updates=1800):
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(SEED)
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name(0)
    state.update(phase='Loading reviewed expert recoveries',learning=dict(stage='loading'))
    data=load_episodes()
    warm=ORIGINAL_ROOT/'best-bc-v5.pt'
    saved=torch.load(warm,map_location='cuda',weights_only=False)
    model=Policy().cuda();model.load_state_dict(saved['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    groups=[[e for e in data['train'] if e['recovery']],
            [e for e in data['train'] if not e['recovery'] and not e['corrective']],
            [e for e in data['train'] if e['corrective']]]
    rng=np.random.default_rng(SEED);history=[];best=float('inf');started=time.monotonic()
    for step in range(1,updates+1):
        model.train();group=groups[int(rng.choice(3,p=[.5,.3,.2]))];e=group[int(rng.integers(len(group)))]
        b=full_batch(e,True);optimizer.zero_grad(set_to_none=True);total,parts=loss(model,b)
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite recovery fit')
        total.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        del b,total
        if step==1 or step%100==0:
            model.eval()
            with torch.no_grad():
                values=[loss(model,full_batch(v))[1]['action'] for v in data['validation']]
            val=float(np.mean(values));elapsed=time.monotonic()-started
            row=dict(step=step,train=parts,validation_action=val,elapsed_s=elapsed)
            history.append(row);(ROOT/'training-history.json').write_text(json.dumps(history,indent=2)+'\n')
            state.update(phase=f'Full-sequence corrective BC {step}/{updates}',progress=step/updates,
                         learning=dict(stage='training',**row))
            payload=dict(saved,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                seed=SEED,model_version='recovery_sequence_bc_v6',validation_action=val,
                warm_start_checkpoint_sha256=hashlib.sha256(warm.read_bytes()).hexdigest(),
                training_device=dict(name=torch.cuda.get_device_name(0),visible_cuda_devices=1),
                policy_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                actor_source_sha256=hashlib.sha256(Path('/workspace/src/pour_learning.py').read_bytes()).hexdigest(),
                dataset_manifest_sha256=hashlib.sha256((ROOT/'dataset-manifest.json').read_bytes()).hexdigest(),
                scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),
                full_sequence=True,expert_prefix_masked=True,
                train_episodes=len(data['train']),validation_episodes=len(data['validation']),
                full_training_episodes=sum(not e['corrective'] and not e['recovery'] for e in data['train']),
                corrective_prefixes=sum(e['corrective'] for e in data['train']),
                recovery_episodes=sum(e['recovery'] for e in data['train']),
                selection='Lowest full-sequence validation action loss; actual native autonomy scored separately')
            torch.save(payload,ROOT/'last.partial.pt');(ROOT/'last.partial.pt').replace(ROOT/'last.pt')
            if val<best:
                best=val;torch.save(payload,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(ROOT/'best.pt')
    state.update(phase='Recovery fit complete',learning=dict(stage='fit_complete',updates=updates,
                 best_validation_action=best,elapsed_s=time.monotonic()-started))
    del optimizer,model,data
    torch.cuda.empty_cache()
    return str(ROOT/'best.pt')
