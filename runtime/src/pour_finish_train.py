"""4070 supervised refit using explicitly consumed historic pause images.

The former wide test cohort is now train/validation, never new-test evidence.
Localizer, empty CNN and cloned motor weights are retained from the parent.
"""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from validate_pour_recording import h5py,decode
from pour_guard_models import CupEstimator
from pour_guard_vision import cup_crop,locate,plane_point
from pour_settled_model import features,predict
from pour_finish_study import ROOT,PARENT,PARENT_SHA,PREFIX,OUTPUTS,sha,plan
from pour_wide_train import RUNTIME_SOURCES as PREVIOUS_SOURCES
RUNTIME_SOURCES=tuple(dict.fromkeys(PREVIOUS_SOURCES+('pour_finish_study.py','pour_finish_train.py','pour_finish_calibration.py',
    'pour_camera_clock.py','validate_pour_recording.py','audit_finish_study.py')))
def write(n,d):(ROOT/n).write_text(json.dumps(d,indent=2)+'\n')

def reference_crop_anchors(saved,calibrations,device='cuda'):
    """Reuse initial RGB anchors from matching historical low-fill fixtures.

    The autonomous actor fixes its cup anchor before pouring. Re-localizing a
    prefilled static cup is a different observation distribution and can match
    a background edge. Training fixture poses select the corresponding record;
    its RGB estimate, rather than the simulator pose, defines the crop.
    """
    def key(s):return tuple(s['cup_world'])+tuple(s['source_xy'])
    needed={key(s):s['study_split'] for s in calibrations}
    accepted=set(json.loads((PARENT/'reference-acceptance.json').read_text())['accepted']);anchors={}
    for path in sorted(OUTPUTS.glob('*/*T30_A_demo_*/episode.hdf5')):
        spec=json.loads((path.parent/'parameters.json').read_text());k=key(spec)
        if spec['name'] not in accepted or k not in needed:continue
        assert k not in anchors and spec['study_split']==needed[k]
        with h5py.File(path) as f:
            assert f.attrs['status']=='complete';d=f['data/demo_0']
            uv=np.median([locate(decode(d['obs/task_image'][i],'jpeg95').copy(),saved['cup_bank'],device=device)[0] for i in range(8)],axis=0)
        xy=plane_point(uv,1.078,saved['calibration'])[:2]
        error=float(np.linalg.norm(xy-np.asarray(spec['cup_world'][:2]))*1000)
        assert error<3.,(spec['name'],'Reference crop anchor error',error)
        anchors[k]=dict(name=spec['name'],path=str(path),sha256=sha(path),initial_image_indices=list(range(8)),cup_xy=xy.tolist(),localization_error_mm=error,split=spec['study_split'])
    assert set(anchors)==set(needed),'Every calibration layout needs its recorded RGB reference anchor'
    return anchors

def build(saved):
    cached=torch.load(PARENT/'wide-cache.pt',map_location='cpu',weights_only=False)
    arrays={s:{k:[] for k in ('image','target','cup_xy','episode')} for s in ('train','validation')}
    records=[]
    # Whole swap pairs 3, 8 and 13 remain validation; both goals/orientations.
    # All other former tests and all historical development become training.
    for path in sorted(OUTPUTS.glob('*/*XW30_A_*/episode.hdf5')):
        spec=json.loads((path.parent/'parameters.json').read_text());name=spec['name']
        if not ('_test_' in name or name.endswith('_r3')):continue
        split='validation' if '_test_' in name and spec['pair_index'] in (3,8,13) else 'train'
        rows=json.loads((path.parent/'samples.json').read_text());dec=json.loads((path.parent/'policy-decisions.json').read_text())
        xy=next((r['cup_xy'] for r in dec if r.get('cup_xy') is not None),None)
        assert xy is not None and len(rows)==len(dec)
        selected={};i=0;quantum=.0018**3*1e6
        while i<len(dec):
            if dec[i]['mode']!='Pour control: settle':i+=1;continue
            start=i
            while i<len(dec) and dec[i]['mode']=='Pour control: settle':i+=1
            end=i-1
            if rows[end]['t']-rows[start]['t']<.65:continue
            last=rows[-1] if abs(rows[end]['cup_ml']-rows[-1]['cup_ml'])<=8*quantum else rows[end]
            # Fixed labels from the end of each stable inventory plateau.
            for j in range(start,end+1,2):
                if rows[j]['t']-rows[start]['t']>=.5 and abs(rows[j]['cup_ml']-last['cup_ml'])<=8*quantum:
                    selected[j]=(last['cup_fill_fraction'],'settle')
        for j in range(0,len(rows),3):
            if dec[j]['mode'] in ('Setting bottle down','Releasing bottle','Holding completed pose') and abs(rows[j]['cup_ml']-rows[-1]['cup_ml'])<=8*quantum:
                selected[j]=(rows[-1]['cup_fill_fraction'],'final')
        with h5py.File(path) as f:
            assert f.attrs['status']=='complete';d=f['data/demo_0']
            for j,(target,phase) in selected.items():
                image=cup_crop(decode(d['next_obs/task_image'][j],'jpeg95'),xy,saved['calibration'])
                for k,v in dict(image=image,target=target,cup_xy=xy,episode=name).items():arrays[split][k].append(v)
        assert selected,name
        records.append(dict(name=name,path=str(path),sha256=sha(path),split=split,former_split=spec['study_split'],
                            frames=[dict(index=j,target=v[0],view=v[1]) for j,v in selected.items()]))
    assert len(records)==38
    calibration=json.loads((ROOT/'calibration-plan.json').read_text())['cases']
    anchors=reference_crop_anchors(saved,calibration)
    write('calibration-anchor-manifest.json',dict(records=list(anchors.values()),scope='RGB crop anchors from matching accepted historical low-fill references. Fixed calibration layouts select records only during supervision; autonomous inference still estimates its own initial cup location.'))
    expected={s['name']:s for s in calibration};found=set()
    for path in sorted(OUTPUTS.glob('*/*FCAL_A_*/episode.hdf5')):
        spec=json.loads((path.parent/'parameters.json').read_text());name=spec['name']
        if name not in expected:continue
        assert spec==expected[name] and name not in found;found.add(name)
        rows=json.loads((path.parent/'samples.json').read_text());split=spec['study_split'];selected=[]
        with h5py.File(path) as f:
            assert f.attrs['status']=='complete';d=f['data/demo_0']
            anchor=anchors[tuple(spec['cup_world'])+tuple(spec['source_xy'])];xy=anchor['cup_xy']
            for j in range(0,len(rows),2):
                if rows[j]['t']<1.5:continue
                image=cup_crop(decode(d['next_obs/task_image'][j],'jpeg95'),xy,saved['calibration'])
                for k,v in dict(image=image,target=rows[-1]['cup_fill_fraction'],cup_xy=xy,episode=name).items():arrays[split][k].append(v)
                selected.append(dict(index=j,target=rows[-1]['cup_fill_fraction'],view='static_calibration'))
        assert selected
        records.append(dict(name=name,path=str(path),sha256=sha(path),split=split,former_split='new_supervised_calibration',crop_anchor_reference=anchor,frames=selected))
    assert found==set(expected),'All planned static calibration recordings are required before fitting'
    data={}
    for split,a in arrays.items():
        old=cached['data'][split]
        data[split]={k:old[k]+v if k=='episode' else torch.cat((old[k],torch.from_numpy(np.asarray(v,dtype=np.uint8 if k=='image' else np.float32)))) for k,v in a.items()}
    assert not set(data['train']['episode'])&set(data['validation']['episode'])
    assert all(not n.startswith((PREFIX+'dev_',PREFIX+'test_')) for d in data.values() for n in d['episode'])
    manifest=dict(parent_manifest_sha256=sha(PARENT/'data-manifest.json'),consumed_historical_records=records,
        validation_pair_indices=[3,8,13],scope='Prior frozen benchmark is complete and remains unchanged; these historical test images are explicitly consumed for this new model. FN30 development/test images excluded.')
    write('data-manifest.json',manifest)
    return data

@torch.inference_mode()
def train(state):
    assert not (ROOT/'final-selection.json').exists() and not (ROOT/'best.pt').exists()
    assert not list(OUTPUTS.glob(f'*/*{PREFIX}test*/parameters.json'))
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name(0)
    assert sha(PARENT/'best.pt')==PARENT_SHA
    torch.set_num_threads(4);torch.manual_seed(20261106);start=time.monotonic();plan()
    saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
    state.update(phase='Fitting pause-view liquid level',learning=dict(stage='finish_level_fit'))
    data=build(saved);model=CupEstimator().cuda().eval();model.load_state_dict(saved['cup_model']);matrices={}
    for split,d in data.items():
        xs=[]
        for i in range(0,len(d['target']),128):
            a=d['image'][i:i+128].cuda();cnn=(model(a)+saved['cup_bias']).clamp(0,1)
            xs.append(features(a,cnn,d['cup_xy'][i:i+128].cuda()).double())
        matrices[split]=torch.cat(xs)
    x,vx=matrices['train'],matrices['validation'];y=data['train']['target'].cuda().double();vy=data['validation']['target'].cuda().double()
    count=Counter(data['train']['episode']);w=y.new_tensor([1/count[n] for n in data['train']['episode']])*torch.where(y>.8,3.,1.)
    recent=torch.tensor([n.startswith(('XW30_A_','FCAL_A_')) for n in data['train']['episode']],device='cuda')
    wideval=torch.tensor([n.startswith(('XW30_A_','FCAL_A_')) for n in data['validation']['episode']],device='cuda');near=vy>.8
    candidates=[]
    feature_sets=[list(range(21)),list(range(60)),list(range(340)),list(range(21))+list(range(340,604)),list(range(604)),
                  list(range(21))+list(range(604,x.shape[1])),list(range(x.shape[1]))]
    for weight in (2.,8.,32.):
        weights=w*torch.where(recent,weight,1.);weights/=weights.mean()
        for columns in feature_sets:
            a=x[:,columns];mean=a.mean(0);scale=a.std(0).clamp(min=.001);mean[0]=0;scale[0]=1;a=(a-mean)/scale
            for ridge in (.1,1.,10.,100.):
                reg=torch.eye(len(columns),device='cuda',dtype=torch.float64)*ridge;reg[0,0]=1e-9
                coef=torch.linalg.solve(a.T@(a*weights[:,None])+reg,a.T@(y*weights))
                fit=dict(columns=columns,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),coef=coef.cpu().tolist(),ridge=ridge)
                for bias in (0.,-.005,-.01,-.015,-.02):
                    fit=dict(fit,output_bias=bias)
                    error=predict(vx,fit)-vy;mask=wideval&near
                    metrics=dict(mae=float(error.abs().mean()),near_mae=float(error[near].abs().mean()),wide_near_mae=float(error[mask].abs().mean()),
                        wide_near_max=float(error[mask].abs().max()),wide_near_overestimate=float(error[mask].max()),near_p95=float(torch.quantile(error[near].abs(),.95)))
                    candidates.append(dict(fit=fit,weight=weight,metrics=metrics,score=metrics['wide_near_mae']+metrics['near_mae']+.3*metrics['wide_near_max']))
    # Ridge leaves opposite perspective-dependent errors. Fit a bounded
    # nonlinear residual on training episodes; validation selects the epoch.
    base=min(candidates,key=lambda c:c['score'])['fit']
    for columns in (list(range(7))+list(range(340,367)),list(range(7))+list(range(340,367))+list(range(604,x.shape[1]))):
        with torch.inference_mode(False),torch.enable_grad():
            xx=x[:,columns].float().clone();vv=vx[:,columns].float().clone()
            mean=xx.mean(0);scale=xx.std(0).clamp(min=.001);xx=(xx-mean)/scale;vv=(vv-mean)/scale
            baseline=predict(x,base).float().clone();vbase=predict(vx,base).float().clone()
            yy=y.float().clone();vyy=vy.float().clone();weights=(w*torch.where(recent,16.,1.)).float().clone();weights/=weights.mean()
            for width in (32,64):
                torch.manual_seed(20261106+width)
                net=torch.nn.Sequential(torch.nn.Linear(len(columns),width),torch.nn.Tanh(),torch.nn.Linear(width,16),torch.nn.Tanh(),torch.nn.Linear(16,1)).cuda()
                torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
                opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
                for step in range(1,1201):
                    opt.zero_grad();delta=net(xx)[:,0];loss=((baseline+delta-yy).square()*weights).mean()+.02*delta.square().mean();loss.backward();opt.step()
                    if step%20:continue
                    with torch.no_grad():
                        error=(vbase+net(vv)[:,0]).clamp(0,1)-vyy;mask=wideval&near
                        metrics=dict(mae=float(error.abs().mean()),near_mae=float(error[near].abs().mean()),wide_near_mae=float(error[mask].abs().mean()),
                            wide_near_max=float(error[mask].abs().max()),wide_near_overestimate=float(error[mask].max()),near_p95=float(torch.quantile(error[near].abs(),.95)))
                        layers=[dict(weight=l.weight.cpu().tolist(),bias=l.bias.cpu().tolist()) for l in net if isinstance(l,torch.nn.Linear)]
                        fitted=dict(base,residual_mlp=dict(columns=columns,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),layers=layers))
                        candidates.append(dict(fit=fitted,weight=16.,residual_inputs=len(columns),residual_width=width,optimizer_steps=step,metrics=metrics,
                            score=metrics['wide_near_mae']+metrics['near_mae']+.3*metrics['wide_near_max']))
    write('level-candidates.json',candidates)
    eligible=[c for c in candidates if c['metrics']['wide_near_max']<.04 and c['metrics']['wide_near_overestimate']<=.04]
    chosen=min(eligible or candidates,key=lambda c:c['score']);m=chosen['metrics']
    target=.90+max(.01,m['wide_near_overestimate'])+.005
    report=dict(status='component_failed',chosen={k:v for k,v in chosen.items() if k!='fit'},target=target,
                training={s:dict(episodes=len(set(d['episode'])),frames=len(d['target'])) for s,d in data.items()},elapsed_s=time.monotonic()-start,
                device=torch.cuda.get_device_name(0),parameters=len(chosen['fit']['coef'])+sum(len(l['bias'])+sum(len(row) for row in l['weight']) for l in chosen['fit'].get('residual_mlp',{}).get('layers',[])))
    write('fit-diagnostic.json',report)
    assert m['wide_near_max']<.04 and target<=.945,'Pause-level component needs inspection before native tests'
    saved['settled_regressor']=chosen['fit'];saved['settled_validation']=m
    saved['terminal_drain_hold_s']=0.
    saved['pour_primitive']['slow_rate_deg_s']=2.5
    saved['thresholds'].update(full_settled_target=target,brake_rate_deg_s=45.,trim_fast_rate_deg_s=18.,trim_total_seconds=2.5,
                               coarse_flow_limit_ml_s=10.,coarse_brake_target=.87)
    saved['model_version']='visual_finish_hierarchy_v23'
    saved['parent_checkpoint_sha256']=PARENT_SHA
    saved['method']='Supervised ridge'+(' with a validation-selected nonlinear residual' if 'residual_mlp' in chosen['fit'] else '')+' on cup-normalized RGB surface profiles and spatial liquid-surface descriptors; historical settling pauses and stationary calibration; retained RGB localizer, wrist empty CNN and contextual behavioral cloning; slower pour/brake and continuous upright-to-table return without drain hold.'
    # A staged component fit records the exact files that must be deployed.
    saved['sources']={n:sha(next(p for p in (ROOT/'staged'/n,Path('/workspace/src')/n) if p.exists())) for n in RUNTIME_SOURCES}
    torch.save(saved,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(ROOT/'best.pt')
    report.update(status='fit_complete',checkpoint_sha256=sha(ROOT/'best.pt'),parent_checkpoint_sha256=PARENT_SHA,sources=saved['sources'],
                  scene_plan_sha256=sha(ROOT/'frozen-plan.json'),calibration_plan_sha256=sha(ROOT/'calibration-plan.json'),data_manifest_sha256=sha(ROOT/'data-manifest.json'),thresholds=saved['thresholds'],
                  terminal_drain_hold_s=0.,method=saved['method'],no_privileged_runtime_inputs=True,elapsed_s=time.monotonic()-start)
    write('model-summary.json',report)
    state.update(phase='Finish-study fit complete',learning=dict(stage='fit_complete',elapsed_s=report['elapsed_s']))
    del saved,data,matrices;torch.cuda.empty_cache()
    return str(ROOT/'best.pt')
