"""Targeted supervised gripper-head correction; all other v6 weights frozen."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pour_learning import Policy
from pour_recovery_learning import load_episodes,full_batch
from pour_recovery_study import ROOT


@torch.no_grad()
def bank(model,episodes):
    xs=[];ys=[];valid=[];weights=[]
    for e in episodes:
        b=full_batch(e)
        model(b['task'],b['wrist'],b['q'],b['v'])
        mask=e['mask']
        xs.append(model.features[0,b['mask'][0]].detach())
        ys.append((e['future_a'][mask,:,7:].mean(-1)/(np.pi/4)).clamp(0,1).cuda())
        valid.append(e['future_valid'][mask].cuda())
        release=(e['phase'][mask]==6)&(e['stop'][mask]>0)
        weights.append((1.+11.*release.float())[:,None].cuda())
    return tuple(torch.cat(v) for v in (xs,ys,valid,weights))


def objective(head,b):
    x,y,valid,weight=b
    error=F.binary_cross_entropy_with_logits(head(x),y,reduction='none')
    weighted=(error*valid*weight).sum()/(valid*weight).sum()
    overall=(error*valid).sum()/valid.sum()
    release=valid&(weight>1)
    release_error=(error*release).sum()/release.sum().clamp(min=1)
    return weighted,dict(grip_bce=float(overall.detach()),release_bce=float(release_error.detach()))


def train(state,updates=600):
    torch.set_num_threads(4);torch.manual_seed(20261110)
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name(0)
    state.update(phase='Loading reviewed release examples',learning=dict(stage='loading'))
    data=load_episodes();path=ROOT/'best-bc-v6.pt'
    saved=torch.load(path,map_location='cuda',weights_only=False)
    model=Policy().cuda();model.load_state_dict(saved['model']);model.eval()
    model.requires_grad_(False);model.grip.requires_grad_(True)
    state['phase']='Encoding full recorded histories for release correction'
    train_bank=bank(model,data['train']);val_bank=bank(model,data['validation'])
    optimizer=torch.optim.AdamW(model.grip.parameters(),lr=3e-3,weight_decay=1e-5)
    with torch.no_grad():initial,initial_parts=objective(model.grip,val_bank)
    history=[];best=float('inf');started=time.monotonic()
    for step in range(1,updates+1):
        indices=torch.randint(len(train_bank[0]),(512,),device='cuda')
        b=tuple(v[indices] for v in train_bank)
        optimizer.zero_grad(set_to_none=True);total,parts=objective(model.grip,b)
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite gripper-head fit')
        total.backward();torch.nn.utils.clip_grad_norm_(model.grip.parameters(),1.);optimizer.step()
        if step==1 or step%25==0:
            with torch.no_grad():value,metrics=objective(model.grip,val_bank)
            score=float(value);elapsed=time.monotonic()-started
            row=dict(step=step,validation_weighted_grip_bce=score,validation=metrics,train=parts,elapsed_s=elapsed)
            history.append(row);(ROOT/'release-training-history.json').write_text(json.dumps(history,indent=2)+'\n')
            state.update(phase=f'Gripper release-head BC {step}/{updates}',learning=dict(stage='training',step=step,**metrics))
            unchanged=all(torch.equal(v,saved['model'][k]) for k,v in model.state_dict().items() if not k.startswith('grip.'))
            assert unchanged,'A frozen arm/vision/GRU/stop parameter changed'
            payload=dict(saved,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                seed=20261110,model_version='release_head_bc_v7',validation_action=None,
                validation_weighted_grip_bce=score,validation_release_bce=metrics['release_bce'],
                initial_validation_weighted_grip_bce=float(initial),initial_validation=initial_parts,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                shared_v6_parameters_bitwise_unchanged=unchanged,release_frame_loss_weight=12.,
                warm_start_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                policy_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                data_loader_source_sha256=hashlib.sha256(Path('/workspace/src/pour_recovery_learning.py').read_bytes()).hexdigest(),
                dataset_manifest_sha256=hashlib.sha256((ROOT/'dataset-manifest.json').read_bytes()).hexdigest(),
                selection='Lowest validation release-weighted gripper BCE; all non-gripper weights frozen')
            torch.save(payload,ROOT/'last.partial.pt');(ROOT/'last.partial.pt').replace(ROOT/'last.pt')
            if score<best:
                best=score;torch.save(payload,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(ROOT/'best.pt')
    state.update(phase='Release-head fit complete',learning=dict(stage='fit_complete',updates=updates,
        best_validation_weighted_grip_bce=best,elapsed_s=time.monotonic()-started))
    del model,optimizer,data,train_bank,val_bank
    torch.cuda.empty_cache()
    return str(ROOT/'best.pt')
