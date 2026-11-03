"""Hierarchical imitation: image-only state learning + expert pour increments."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pour_learning import Policy
from pour_stop_models import LiquidDetector,PourIncrement
from pour_stop_data import perception_data,motion_data
from pour_stop_study import ROOT,plan

BASE=Path('/workspace/learning/A-20261102-v1/best-bc-v5.pt')


def batch(data,idx,augment=False):
    b={k:v[idx].cuda() for k,v in data.items()}
    if augment:
        for k in ('cup','source','wrist'):
            x=b[k].float();gain=torch.empty((len(idx),1,1,1),device='cuda').uniform_(.96,1.04)
            b[k]=(x*gain+torch.randn_like(x)*1.5).clamp(0,255)
            x=b[k].permute(0,3,1,2);theta=torch.eye(2,3,device='cuda')[None].repeat(len(idx),1,1)
            theta[:,:2,2].uniform_(-.14 if k=='cup' else -.04,.14 if k=='cup' else .04)
            grid=F.affine_grid(theta,x.shape,align_corners=False)
            b[k]=F.grid_sample(x,grid,padding_mode='border',align_corners=False).permute(0,2,3,1)
    return b


def detector_loss(model,b,pos):
    p=model(b['cup'],b['source'],b['wrist']);source_valid=b['source_valid']>.5;mask=(b['present']>.5)&source_valid;cup_valid=b['cup_valid']>.5
    fill=F.smooth_l1_loss(p['fill'][cup_valid],b['fill'][cup_valid],beta=.03)
    source=F.smooth_l1_loss(p['left'][mask],b['left'][mask],beta=.02)
    empty=F.binary_cross_entropy_with_logits(p['empty_logit'][mask],b['empty'][mask],pos_weight=pos[0])
    full=F.binary_cross_entropy_with_logits(p['full_logit'][cup_valid],(b['fill'][cup_valid]>=.9).float(),pos_weight=pos[1])
    present=F.binary_cross_entropy_with_logits(p['present_logit'][source_valid],b['present'][source_valid])
    return 20*fill+.1*source+empty+.2*full+present


@torch.no_grad()
def evaluate_detector(model,data):
    parts={k:[] for k in ('fill','left','empty_logit','full_logit','present_logit')}
    for start in range(0,len(data['fill']),128):
        b=batch(data,slice(start,start+128));p=model(b['cup'],b['source'],b['wrist'])
        for k in parts:parts[k].append(p[k].cpu())
    p={k:torch.cat(v) for k,v in parts.items()};y=data
    cup_valid=y['cup_valid']>.5;near=(y['fill']>.65)&cup_valid;bias=float(torch.median(y['fill'][near]-p['fill'][near]))
    corrected=(p['fill']+bias).clamp(0,1)
    low=(y['fill']<.75)&cup_valid;valid=y['present']>.5;positive=valid&(y['empty']>.5);negative=valid&~positive
    balanced=.5*(F.binary_cross_entropy_with_logits(p['empty_logit'][positive],y['empty'][positive])+F.binary_cross_entropy_with_logits(p['empty_logit'][negative],y['empty'][negative]))
    metrics=dict(cup_mae_fraction=float((corrected[cup_valid]-y['fill'][cup_valid]).abs().mean()),
        cup_excluded_label_frames=int((~cup_valid).sum()),cup_valid_frames=int(cup_valid.sum()),fullness_label='surface height / rim height',
        cup_bias_correction=bias,early_full_false_positives=int(((corrected>=.82)&low).sum()),low_fill_frames=int(low.sum()),
        source_mae_nominal_ml=float((p['left'][valid]-y['left'][valid]).abs().mean()*100),
        empty_bce=float(F.binary_cross_entropy_with_logits(p['empty_logit'][valid],y['empty'][valid])),
        empty_balanced_bce=float(balanced),presence_bce=float(F.binary_cross_entropy_with_logits(p['present_logit'],y['present'])))
    score=metrics['cup_mae_fraction']+.04*metrics['empty_bce']+.01*metrics['presence_bce']+2*metrics['early_full_false_positives']/max(1,metrics['low_fill_frames'])
    return score,metrics,p


@torch.no_grad()
def motion_bank(base,detector,episodes):
    xs=[];ys=[]
    for e in episodes:
        for start in range(0,len(e['q']),128):
            sl=slice(start,start+128);task=torch.from_numpy(e['task'][sl]).cuda();wrist=torch.from_numpy(e['wrist'][sl]).cuda()
            n=len(task);q=torch.from_numpy(e['q'][sl]).cuda();v=torch.from_numpy(e['v'][sl]).cuda()
            a=base.task(task.permute(0,3,1,2).float()/255.);b=base.wrist(wrist.permute(0,3,1,2).float()/255.)
            p=detector(*[torch.from_numpy(e[k][sl]).cuda() for k in ('cup','source','source_wrist')])
            lift=(torch.from_numpy(e['qlift']).float().cuda()-base.q_mean[:7])/base.q_std[:7]
            xs.append(torch.cat([a,b,(q-base.q_mean)/base.q_std,v.clamp(-5,5)/2.,lift.expand(n,-1),p['fill'][:,None],p['left'][:,None]],-1))
            ys.append(torch.from_numpy(e['delta'][sl]).cuda()/.02)
    return torch.cat(xs),torch.cat(ys)


def train(state,detector_updates=1800,motion_updates=1600):
    plan();torch.set_num_threads(4);torch.manual_seed(20261114)
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name(0)
    state.update(phase='Loading independent visual liquid labels',learning=dict(stage='loading'))
    data=perception_data();model=LiquidDetector().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    train=data['train'];valid=train['present']>.5
    ep=float(train['empty'][valid].sum());fp=float((train['fill']>=.9).sum())
    pos=[torch.tensor(min(30.,(int(valid.sum())-ep)/max(1,ep)),device='cuda'),
         torch.tensor(min(30.,(len(train['fill'])-fp)/max(1,fp)),device='cuda')]
    history=[];best_cup=float('inf');best_source=float('inf');started=time.monotonic()
    empty_indices=torch.nonzero((train['empty']>.5)&(train['present']>.5)).flatten()
    full_indices=torch.nonzero((train['fill']>.85)&(train['cup_valid']>.5)).flatten()
    low_indices=torch.nonzero((train['fill']<.65)&(train['cup_valid']>.5)).flatten()
    pos=[torch.tensor(2.,device='cuda'),torch.tensor(1.,device='cuda')]
    for step in range(1,detector_updates+1):
        model.train();idx=torch.cat([torch.randint(len(train['fill']),(16,))]+[pool[torch.randint(len(pool),(16,))] for pool in (empty_indices,full_indices,low_indices)]);b=batch(train,idx,True)
        optimizer.zero_grad(set_to_none=True);loss=detector_loss(model,b,pos)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite visual training')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==1 or step%100==0:
            model.eval();score,metrics,_=evaluate_detector(model,data['validation']);elapsed=time.monotonic()-started
            history.append(dict(step=step,loss=float(loss.detach()),score=score,elapsed_s=elapsed,**metrics))
            state.update(phase=f'Image-only liquid estimator {step}/{detector_updates}',learning=dict(stage='training',component='liquid_detector',step=step,**metrics))
            (ROOT/'detector-history.json').write_text(json.dumps(history,indent=2)+'\n')
            cup_score=metrics['cup_mae_fraction']+2*metrics['early_full_false_positives']/max(1,metrics['low_fill_frames'])
            source_score=metrics['empty_balanced_bce']+.05*metrics['presence_bce']
            for key,value in (('cup',cup_score),('source',source_score)):
                improved=value<(best_cup if key=='cup' else best_source)
                if improved:
                    if key=='cup':best_cup=value
                    else:best_source=value
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,metrics=metrics),ROOT/f'detector-{key}-best.pt')
    selected=torch.load(ROOT/'detector-cup-best.pt',weights_only=False);source_selected=torch.load(ROOT/'detector-source-best.pt',weights_only=False)
    combined=dict(selected['model']);combined.update({k:v for k,v in source_selected['model'].items() if not k.startswith(('cup.','cup_head.'))})
    selected['model']=combined;model.load_state_dict(combined);model.eval()
    _,metrics,p=evaluate_detector(model,data['validation']);v=data['validation'];present=v['present']>.5
    probs=p['empty_logit'].sigmoid();positive=present&(v['empty']>.5);negative=present&(v['empty']<.5)
    candidates=[];gate=present&(v['empty_gate']>.5)
    for threshold in (.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.97,.98,.99,.995,.999):
        called=(probs>=threshold)&(p['present_logit'].sigmoid()>.7)
        candidates.append(dict(threshold=threshold,true_positive=int((called&positive).sum()),false_positive=int((called&negative).sum()),positives=int(positive.sum()),
            gated_true_positive=int((called&positive&gate).sum()),gated_false_positive=int((called&negative&gate).sum()),gated_positives=int((positive&gate).sum())))
    eligible=[r for r in candidates if r['gated_false_positive']==0 and r['gated_true_positive']>0]
    operating=eligible[0] if eligible else candidates[-1]
    metrics.update(empty_operating_point=operating,empty_threshold_sweep=candidates,
        empty_calibration_note='Tilt/low-flow active-pour proxy gate; very few gated positive validation frames. Native rollouts remain required.',
        train_frames=len(train['fill']),validation_frames=len(v['fill']),selected_detector_step=selected['step'],selected_source_step=source_selected['step'])
    (ROOT/'detector-validation.json').write_text(json.dumps(metrics,indent=2)+'\n')
    thresholds=dict(cup_bias=metrics['cup_bias_correction'],empty_probability=operating['threshold'],
        full_floor=.82,full_future_target=.91,source_present=.7)
    del data,optimizer,train,b,loss
    state.update(phase='Learning expert pour increments and return primitive',learning=dict(stage='loading',component='motor_policy'))
    motion,primitive=motion_data();saved=torch.load(BASE,map_location='cuda',weights_only=False)
    base=Policy().cuda().eval();base.load_state_dict(saved['model']);base.requires_grad_(False)
    banks={s:motion_bank(base,model,es) for s,es in motion.items()}
    head=PourIncrement().cuda();opt=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-5)
    hist=[];best=float('inf');clock=time.monotonic()
    for step in range(1,motion_updates+1):
        x,y=banks['train'];idx=torch.randint(len(x),(256,),device='cuda');pred=head(x[idx]);loss=F.mse_loss(pred,y[idx])
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        if step==1 or step%100==0:
            with torch.no_grad():error=head(banks['validation'][0])-banks['validation'][1];score=float(error.square().mean());rmse=np.degrees(np.sqrt(error.square().mean(0).cpu().numpy())*.02).tolist()
            hist.append(dict(step=step,validation_delta_mse=score,validation_delta_rmse_deg=rmse,elapsed_s=time.monotonic()-clock))
            state.update(phase=f'Pour increment BC {step}/{motion_updates}',learning=dict(stage='training',component='pour_policy',step=step,validation_action=score))
            if score<best:
                best=score;torch.save(dict(model=head.state_dict(),optimizer=opt.state_dict(),step=step,rmse_deg=rmse),ROOT/'pour-best.pt')
    (ROOT/'motion-history.json').write_text(json.dumps(hist,indent=2)+'\n')
    chosen=torch.load(ROOT/'pour-best.pt',weights_only=False)
    from pour_stop_trajectory import fit
    pour_primitive=fit()
    payload=dict(model_version='visual_stop_hierarchy_v9',step=chosen['step'],seed=20261114,
        base_model=saved['model'],detector=selected['model'],pour=chosen['model'],primitive=primitive,thresholds=thresholds,pour_primitive=pour_primitive,
        detector_step=selected['step'],detector_validation=metrics,pour_validation_rmse_deg=chosen['rmse_deg'],
        inputs=['task RGB','wrist RGB','measured joint positions/velocities','own motion history and latched stop'],
        method='Hierarchical supervised imitation: image-only liquid estimation, contextual ridge-regression pouring primitive, demonstration-derived terminal primitive',
        training_device=dict(name=torch.cuda.get_device_name(0),visible_cuda_devices=1),
        baseline_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        sources={p:hashlib.sha256(Path('/workspace/src',p).read_bytes()).hexdigest() for p in ('pour_stop_train.py','pour_stop_data.py','pour_stop_models.py','pour_stop_sequence.py','pour_stop_actor.py','pour_stop_trajectory.py')},
        scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),
        perception_manifest_sha256=hashlib.sha256((ROOT/'perception-manifest.json').read_bytes()).hexdigest(),
        perception_correction_sha256=hashlib.sha256((ROOT/'corrective-perception.json').read_bytes()).hexdigest(),
        perception_label_quality_sha256=hashlib.sha256((ROOT/'perception-label-quality.json').read_bytes()).hexdigest(),
        motion_manifest_sha256=hashlib.sha256((ROOT/'motion-manifest.json').read_bytes()).hexdigest(),
        no_privileged_runtime_inputs=True)
    torch.save(payload,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(ROOT/'best.pt')
    state.update(phase='Visual-stop hierarchy fit complete',learning=dict(stage='fit_complete',detector_updates=detector_updates,motion_updates=motion_updates,elapsed_s=time.monotonic()-started))
    del model,base,head,opt,banks,motion
    torch.cuda.empty_cache();return str(ROOT/'best.pt')
