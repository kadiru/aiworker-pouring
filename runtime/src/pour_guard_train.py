"""Bounded 4070 training for visual localization and aligned cup-height fitting."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pour_guard_study import ROOT,plan
from pour_guard_data import build_data
from pour_guard_vision import make_bank,locate,plane_point,surface_features
from pour_guard_models import CupEstimator

BASE=Path('/workspace/learning/A-stop-20261103-v1/best.pt')


@torch.inference_mode()
def evaluate(model,data):
    output=[]
    for start in range(0,len(data['fill']),128):output.append(model(data['image'][start:start+128].cuda()).cpu())
    p=torch.cat(output);v=data['fill'];mask=(data['calm']>.5)&(data['good']>.5)&(v>.65)
    bias=float(torch.median(v[mask]-p[mask]));p=(p+bias).clamp(0,1)
    return dict(mae=float((p-v).abs().mean()),calm_near_full_mae=float((p[mask]-v[mask]).abs().mean()),bias=bias,
        calm_near_full_frames=int(mask.sum()),calm_near_full_p90_error=float(torch.quantile((p[mask]-v[mask]).abs(),.9))),p


def fit_geometry(data):
    train=data['train'];val=data['validation'];features={}
    for split,d in data.items():
        fs=[];mass=[]
        for start in range(0,len(d['fill']),256):
            x,_,n=surface_features(d['image'][start:start+256].cuda());fs.append(x);mass.append(n)
        features[split]=(torch.cat(fs),torch.cat(mass))
    result=[]
    for dimensions in (2,3,4):
        x,m=features['train'];y=train['fill'].cuda();mask=(train['calm'].cuda()>.5)&(train['good'].cuda()>.5)&(y>.12)&(m>1.)
        a=x[mask,:dimensions];b=y[mask];weight=torch.where(b>.7,4.,1.)
        penalty=torch.eye(dimensions,device='cuda')*.0001;penalty[0,0]=.000001
        for _ in range(8):
            coef=torch.linalg.solve(a.T@(a*weight[:,None])+penalty,a.T@(b*weight))
            residual=(a@coef-b).abs();weight=torch.where(b>.7,4.,1.)*(.025/residual.clamp(min=.025))
        vx,vm=features['validation'];vy=val['fill'].cuda();near=(val['calm'].cuda()>.5)&(val['good'].cuda()>.5)&(vy>.65)&(vm>1.)
        pred=vx[:,:dimensions]@coef;bias=float(torch.median(vy[near]-pred[near]));pred+=bias
        error=(pred[near]-vy[near]).abs()
        result.append(dict(dimensions=dimensions,coef=coef.cpu().tolist(),bias=bias,near_full_mae=float(error.mean()),
            near_full_p90_error=float(torch.quantile(error,.9)),near_full_max_error=float(error.max()),validation_frames=int(near.sum())))
    selected=min(result,key=lambda r:r['near_full_mae']);return dict(selected=selected,candidates=result)


def fit_localization(initial,cal):
    train=initial['train'];indices=np.unique(np.round(np.linspace(0,len(train)-1,16)).astype(int))
    selected=[train[i] for i in indices];source=make_bank([e['image'] for e in selected],[e['source_uv'] for e in selected],'source')
    cups=selected[::2];cup=make_bank([e['image'] for e in cups],[e['cup_uv'] for e in cups],'cup')
    errors=[]
    for e in initial['validation']:
        row=dict(name=e['name'])
        for kind,bank,z in (('source',source,cal['table_z']+.135),('cup',cup,cal['table_z']+.038)):
            uv,score=locate(e['image'],bank);xy=plane_point(uv,z,cal)[:2]
            row[kind+'_error_mm']=float(np.linalg.norm(xy-e[kind+'_xy'])*1000);row[kind+'_score']=score
        errors.append(row)
    report=dict(method='Supervised gradient-template localization, CUDA normalized correlation and subpixel refinement',
        source_templates=len(selected),cup_templates=len(cups),source_training_names=[e['name'] for e in selected],
        cup_training_names=[e['name'] for e in cups],validation=errors,
        max_source_error_mm=max(e['source_error_mm'] for e in errors),max_cup_error_mm=max(e['cup_error_mm'] for e in errors))
    (ROOT/'localization-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return source,cup,report


def train(state,updates=1800):
    plan();torch.set_num_threads(4);torch.manual_seed(20261123)
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name(0)
    started=time.monotonic();state.update(phase='Loading rim-normalized training labels',learning=dict(stage='loading'))
    bundle=build_data();data=bundle['data'];cal=bundle['calibration']
    source_bank,cup_bank,localization=fit_localization(bundle['initial'],cal)
    geometry=fit_geometry(data);(ROOT/'surface-geometry-validation.json').write_text(json.dumps(geometry,indent=2)+'\n')
    model=CupEstimator().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    tr=data['train'];near=torch.nonzero(tr['fill']>.65).flatten();history=[];best=float('inf')
    for step in range(1,updates+1):
        idx=torch.cat([torch.randint(len(tr['fill']),(32,)),near[torch.randint(len(near),(32,))]])
        image=tr['image'][idx].cuda().float();gain=torch.empty((len(idx),1,1,1),device='cuda').uniform_(.97,1.03)
        image=(image*gain+torch.randn_like(image)*.5).clamp(0,255)
        y=tr['fill'][idx].cuda();pred=model(image);weights=(1+3*(y>.65).float())*(1+2*tr['calm'][idx].cuda())
        loss=(F.smooth_l1_loss(pred,y,beta=.02,reduction='none')*weights).mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step==1 or step%100==0:
            model.eval();metrics,_=evaluate(model,data['validation']);score=metrics['calm_near_full_mae']+.2*metrics['mae']
            history.append(dict(step=step,loss=float(loss.detach()),elapsed_s=time.monotonic()-started,**metrics))
            state.update(phase=f'Rim-normalized cup model {step}/{updates}',learning=dict(stage='training',step=step,**metrics))
            (ROOT/'training-history.json').write_text(json.dumps(history,indent=2)+'\n')
            if score<best:
                best=score;torch.save(dict(model=model.state_dict(),step=step,metrics=metrics),ROOT/'cup-best.pt')
            model.train()
    chosen=torch.load(ROOT/'cup-best.pt',map_location='cpu',weights_only=False)
    old=torch.load(BASE,map_location='cpu',weights_only=False)
    from pour_guard_trajectory import fit as fit_trajectory
    movement=fit_trajectory()
    payload=dict(model_version='visual_guard_hierarchy_v10',step=chosen['step'],cup_model=chosen['model'],
        cup_bias=chosen['metrics']['bias'],cup_validation=chosen['metrics'],surface_geometry=geometry['selected'],
        source_detector=old['detector'],source_bank=source_bank,cup_bank=cup_bank,calibration=cal,
        primitive=old['primitive'],pour_primitive=movement,thresholds=dict(full_settled_target=.96,
            coarse_brake_target=.89,runoff_seconds=.25,empty_probability=old['thresholds']['empty_probability'],
            source_localization_score=.8*min(e['source_score'] for e in localization['validation']),
            cup_localization_score=.8*min(e['cup_score'] for e in localization['validation']),
            source_present=.7,empty_votes=12,settle_seconds=1.2,trim_increment=.015,max_attempts=2),
        method='Visual servo grasp with lift confirmation/retry; rim-normalized supervised cup model and color-geometry calibration; contextual movement primitive and settled-level top-up control',
        inputs=['task RGB','wrist RGB','measured joint positions/velocities','own history/phase','fixed camera/robot/bench calibration'],
        training_device=dict(name=torch.cuda.get_device_name(0),visible_cuda_devices=1),
        training_frames={s:len(d['fill']) for s,d in data.items()},localization_validation=localization,
        no_privileged_runtime_inputs=True,parent_checkpoint_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        scene_plan_sha256=hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest(),
        data_manifest_sha256=hashlib.sha256((ROOT/'data-manifest.json').read_bytes()).hexdigest(),
        sources={n:hashlib.sha256(Path('/workspace/src',n).read_bytes()).hexdigest() for n in
            ('pour_guard_train.py','pour_guard_data.py','pour_guard_models.py','pour_guard_vision.py','pour_guard_actor.py','pour_guard_trajectory.py')})
    torch.save(payload,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(ROOT/'best.pt')
    state.update(phase='Visual-guard training complete',learning=dict(stage='fit_complete',updates=updates,elapsed_s=time.monotonic()-started,
        cup_validation=chosen['metrics'],geometry_validation=geometry['selected']))
    del data,bundle,model,opt;torch.cuda.empty_cache();return str(ROOT/'best.pt')
