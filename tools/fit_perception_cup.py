"""4070 supervised cup localization and level fit from disjoint native scenes.

Writes component artifacts only. No active checkpoint replacement, test-data
access, or changes to physics/task scores occur in this helper.
"""
import argparse,copy,hashlib,json,sys,time,urllib.request
from itertools import product
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path[:0]=['/workspace/learning/A-perception-20261108-v1/staged','/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_vision import make_bank,locate,project,plane_point,cup_crop,surface_features,color_geometry_features
from pour_guard_models import CupEstimator
from pour_perception_geometry import settled_label
from pour_temporal_localization import consensus
from perception_fit_data import admitted_scenes,admitted_tasks
from perception_task_labels import task_level_frames
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')

def collect():
    plan=read(ROOT/'plan.json');admitted={s['name']:(s,split,reason) for s,split,reason in admitted_scenes(plan)}
    expected={n:a[0] for n,a in admitted.items()}
    seq=[];manifest=[];cal=None
    for path in sorted(Path('/workspace/outputs/fountain').glob('*/*PV_A_*/parameters.json')):
        spec=read(path)
        if spec['name'] not in expected:continue
        assert all(spec[k]==v for k,v in expected[spec['name']].items())
        _,split,admission=admitted[spec['name']]
        h5=path.parent/'episode.hdf5';assert h5.exists();rows=read(path.parent/'samples.json');label=settled_label(rows)
        entry=dict(name=spec['name'],split=split,recorded_split=spec['study_split'],admission=admission,path=str(h5),sha256=sha(h5),label=label)
        manifest.append(entry)
        with h5py.File(h5) as f:
            assert f.attrs['status']=='complete';d=f['data/demo_0'];current=dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),
                T=d['obs/cup_detail_world_T_camera'][0].tolist(),table_z=1.04,resolution_wh=[1024,1024])
            if cal is None:cal=current
            else:assert np.allclose(cal['K'],current['K']) and np.allclose(cal['T'],current['T'])
            images=[np.asarray(Image.fromarray(decode(d['obs/cup_detail_image'][i],'jpeg95')).resize((384,384),Image.Resampling.BILINEAR)).copy() for i in range(12)]
        seq.append(dict(spec=spec,fit_split=split,entry=entry,images=images))
    assert len(seq)==len(expected) and len({e['spec']['name'] for e in seq})==len(expected)
    indexed={r['name']:r for r in read(ROOT/'recording-audit.json')['records']}
    for original,split,admission in admitted_tasks(plan):
        spec=copy.deepcopy(original)
        from pour_vessel_geometry import cup_geometry
        spec['declared_cup_geometry']=cup_geometry(spec)
        r=indexed[spec['name']];path=Path(r['path']);assert sha(path)==r['sha256']
        entry=dict(name=spec['name'],split=split,recorded_split=spec['study_split'],admission=admission,
            path=str(path),sha256=r['sha256'],task_frames=task_level_frames(read(path.parent/'samples.json'),read(path.parent/'policy-decisions.json')))
        with h5py.File(path) as f:
            assert f.attrs['status']=='complete';d=f['data/demo_0']
            assert np.allclose(f['calibration/cup_detail/intrinsics'][:],cal['K'])
            assert np.allclose(d['obs/cup_detail_world_T_camera'][:],cal['T'])
            images=[np.asarray(Image.fromarray(decode(d['obs/cup_detail_image'][i],'jpeg95')).resize((384,384),Image.Resampling.BILINEAR)).copy() for i in range(12)]
        manifest.append(entry);seq.append(dict(spec=spec,fit_split=split,entry=entry,images=images))
    write('cup-data-manifest.json',manifest)
    return plan,seq,cal

@torch.inference_mode()
def template_localize_rejected(seq,detail,gate):
    cal=copy.deepcopy(detail);K=np.array(cal['K']);K[:2]*=384/1024;cal['K']=K.tolist();cal['resolution_wh']=[384,384]
    train=[e for e in seq if e['spec']['study_split']=='train'];anchors=[]
    for e in train:anchors.append(project(np.array(e['spec']['cup_world'])+[0,0,e['spec']['declared_cup_geometry']['rim_z_m']],cal))
    xy=np.array([e['spec']['cup_world'][:2] for e in train]);lo=xy.min(0)-.025;hi=xy.max(0)+.025
    region=[project([x,y,1.078],cal).tolist() for x,y in ((lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1]))]
    candidates=[]
    for absolute,clip,mask in product((False,True),(.05,.1,None),((10,0,70,16),(10,2,70,24),(16,4,64,22),None)):
        bank=make_bank([e['images'][0] for e in train],anchors,'cup',aggregation='max',absolute_gradients=absolute,
            gradient_clip=clip,cup_mask_box=mask,anchor_radius_px=48.)
        bank['bounds']=[24,24,360,356];bank['initial_region_uv']=region;rows=[]
        for e in seq:
            image=e['images'][0];s=e['spec'];z=1.04+s['declared_cup_geometry']['rim_z_m'];uv,score=locate(image,bank)
            point=plane_point(uv,z,cal)[:2];error=float(np.linalg.norm(point-np.array(s['cup_world'][:2]))*1000)
            true=project(np.array(s['cup_world'])+[0,0,z-1.04],cal);x,y=np.rint(true).astype(int)
            negative=image.copy();negative[max(0,y-20):min(384,y+35),max(0,x-45):min(384,x+45)]=180
            _,negative_score=locate(negative,bank)
            rows.append(dict(name=s['name'],split=s['study_split'],error_mm=error,score=score,synthetic_absent_score=negative_score))
        minimum=min(r['score'] for r in rows);maximum=max(r['synthetic_absent_score'] for r in rows)
        row=dict(absolute_gradients=absolute,gradient_clip=clip,cup_mask_box=mask,max_error_mm=max(r['error_mm'] for r in rows),
            margin=minimum-maximum,threshold=(minimum+maximum)/2,rows=rows)
        candidates.append((row,bank));print('CUP_LOCALIZATION',json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)
    write('cup-localization-candidates.json',[r for r,b in candidates])
    eligible=[(r,b) for r,b in candidates if r['max_error_mm']<gate['localization_max_error_mm'] and r['margin']>gate['localization_negative_margin']]
    if not eligible:return None,dict(passed=False,reason='No precise, occlusion-rejecting bank',best=min((r for r,b in candidates),key=lambda r:r['max_error_mm']))
    chosen,bank=min(eligible,key=lambda rb:(rb[0]['max_error_mm'],-rb[0]['margin']))
    config=dict(window=20,votes=8,radius_px=1.5,fraction=.6,score_floor=max(.25,chosen['threshold']-.2));windows=[]
    for e in seq:
        history=[];accepted=None;s=e['spec'];z=1.04+s['declared_cup_geometry']['rim_z_m']
        for image in e['images']:
            uv,score=locate(image,bank);history.append((uv.tolist(),score));accepted=consensus(history,chosen['threshold'],config)
            if accepted:break
        if accepted is None:return None,dict(passed=False,reason='Temporal localization unconfirmed',name=s['name'],chosen=chosen)
        xy=plane_point(accepted['uv'],z,cal)[:2];error=float(np.linalg.norm(xy-s['cup_world'][:2])*1000)
        if error>=gate['localization_max_error_mm']:return None,dict(passed=False,reason='Temporal localization inaccurate',name=s['name'],error_mm=error)
        e['localized_xy']=xy;windows.append(dict(name=s['name'],error_mm=error,frames=len(history)))
    return dict(bank=bank,calibration=cal,threshold=chosen['threshold'],consensus=config),dict(passed=True,chosen=chosen,windows=windows,
        method='Supervised gradient-template banks and confidence/temporal-consensus calibration; fixture dimensions are declared, not visually inferred. Synthetic absent masks are a diagnostic, not an independent real missing-cup test.')

@torch.inference_mode()
def localize(seq,detail,gate):
    selection=read(ROOT/'scope-selection.json');assert selection['passed'] and selection['scope']=='baseline_cup_only'
    assert sha(ROOT/'cup-neural-localization.pt')==selection['checkpoint_sha256']
    selected=torch.load(ROOT/'cup-neural-localization.pt',map_location='cpu',weights_only=False)
    cal=selected['calibration'];windows=[]
    for e in seq:
        history=[];accepted=None;s=e['spec'];z=1.04+s['declared_cup_geometry']['rim_z_m']
        for image in e['images']:
            uv,score=locate(image,selected['bank']);history.append((uv.tolist(),score))
            accepted=consensus(history,selected['threshold'],selected['consensus'])
            if accepted is not None:break
        baseline=s.get('cup_height_scale',1.)==s.get('cup_diameter_scale',1.)==1.
        e['localized_xy']=None if accepted is None else plane_point(accepted['uv'],z,cal)[:2]
        error=None if accepted is None else float(np.linalg.norm(e['localized_xy']-s['cup_world'][:2])*1000)
        e['level_fit_usable']=error is not None and error<gate['localization_max_error_mm']
        windows.append(dict(name=s['name'],baseline_cup=baseline,confirmed=accepted is not None,error_mm=error,frames=len(history),level_fit_usable=e['level_fit_usable']))
        if baseline and (accepted is None or error>=gate['localization_max_error_mm']):
            return None,dict(passed=False,scope='baseline_cup_only',reason='Baseline temporal localization failed',windows=windows)
    return selected,dict(passed=True,scope='baseline_cup_only',windows=windows,
        max_baseline_error_mm=max(r['error_mm'] for r in windows if r['baseline_cup']),
        all_size_temporal_checks_passed=all(r['confirmed'] and r['error_mm']<gate['localization_max_error_mm'] for r in windows),
        resized_autonomy_authorized=False,selection_sha256=sha(ROOT/'scope-selection.json'),
        method='Supervised RGB center heatmap plus subpixel offsets and unchanged eight-vote temporal confirmation. Original vessel dimensions only; resized autonomy remains gated.')

def fit_level(seq,cal,gate,parent,fixed_cnn=None):
    data={s:dict(image=[],target=[],name=[],domain=[],xy=[],prototype=[]) for s in ('train','validation')}
    for e in seq:
        label=e['entry'].get('label');s=e['spec']
        if not e['level_fit_usable'] or (label is not None and not label['accepted']):continue
        actual=copy.deepcopy(cal);actual['cup_geometry']=s['declared_cup_geometry'];a=data[e['fit_split']]
        with h5py.File(e['entry']['path']) as f:
            d=f['data/demo_0'];times=d['next_time'][:];ids=np.flatnonzero(times>=times[-1]-1.-1e-7)
            frames=e['entry'].get('task_frames')
            if frames is None:frames=[dict(group='next_obs',index=int(i),target=label['target'],prototype='static') for i in ids[::2]]
            for frame in frames:
                image=cup_crop(decode(d[frame['group']+'/cup_detail_image'][frame['index']],'jpeg95'),e['localized_xy'],actual)
                a['image'].append(image);a['target'].append(frame['target']);a['name'].append(s['name'])
                a['domain'].append('native_detail' if label is not None else 'consumed_task_detail')
                a['prototype'].append(s['name']+'/'+frame['prototype'])
                a['xy'].append(e['localized_xy'])
    # Rehearse earlier real pouring appearances to avoid replacing dynamic
    # perception with a model trained only on stationary high-resolution cups.
    # These remain original lower-resolution task images, never synthetic
    # detail-camera recordings. Only established train/validation references
    # are read; no historical or new reserved test images enter this fit.
    replay=[]
    for entry in read(Path('/workspace/learning/A-wide-20261105-v1/empty-readout-candidates.json'))['data_manifest']:
        path=Path(entry['path']);assert sha(path)==entry['sha256'];rows=read(path.parent/'samples.json');used=[]
        with h5py.File(path) as f:
            d=f['data/demo_0'];uv,score=locate(decode(d['obs/task_image'][0],'jpeg95'),parent['cup_bank'])
            assert score>=parent['thresholds']['cup_localization_score']
            xy=plane_point(uv,1.078,parent['calibration'])[:2];a=data[entry['split']]
            for i in np.unique(np.linspace(0,len(rows)-1,24,dtype=int)):
                r=rows[i]
                if r['ever_table_ml'] or r['spilled_ml']:continue
                target=r['cup_fill_fraction'];label_source='instantaneous_dynamic_surface'
                if i>=20:
                    window=rows[:i+1];stable=settled_label(window)
                    if stable['accepted']:target=stable['target'];label_source='measured_trailing_stable_median'
                image=cup_crop(decode(d['next_obs/task_image'][int(i)],'jpeg95'),xy,parent['calibration'])
                a['image'].append(image);a['target'].append(target);a['name'].append(entry['name']);a['domain'].append('historical_task_replay')
                a['xy'].append(xy)
                a['prototype'].append(entry['name'])
                used.append(dict(index=int(i),target=target,label_source=label_source))
        replay.append(dict(name=entry['name'],split=entry['split'],path=str(path),sha256=entry['sha256'],frames=used))
    write('cup-replay-manifest.json',replay)
    assert set(data['train']['name']).isdisjoint(data['validation']['name'])
    for s,a in data.items():
        assert len({n for n,d in zip(a['name'],a['domain']) if d=='native_detail'})>=(20 if s=='train' else 14),f'Insufficient stable episodes in {s}'
        a['x']=torch.from_numpy(np.stack(a.pop('image'))).cuda();a['y']=torch.tensor(a['target'],device='cuda')
        a['xy']=torch.tensor(np.asarray(a['xy']),device='cuda',dtype=torch.float32)
    tr,va=data['train'],data['validation'];all_native=torch.tensor([d!='historical_task_replay' for d in va['domain']],device='cuda')
    baseline_names={e['spec']['name'] for e in seq if e['spec'].get('cup_height_scale',1.)==e['spec'].get('cup_diameter_scale',1.)==1.}
    native=all_native & torch.tensor([n in baseline_names for n in va['name']],device='cuda');replay_mask=~all_native
    near=(va['y']>.85)&native
    task_mask=torch.tensor([d=='consumed_task_detail' for d in va['domain']],device='cuda')
    task_near=(va['y']>.85)&task_mask
    if admitted_tasks(read(ROOT/'plan.json')):
        assert task_near.sum()>=15,'No actual-pour near-full validation coverage'
        assert fixed_cnn is None,'Task-domain repair requires fitting and checking the CNN'
    assert int(near.sum())>=20 and float(va['y'][native].max())>=.94,'No near-full validation coverage'
    model=CupEstimator().cuda();model.load_state_dict(parent['cup_model']);model.eval()
    with torch.no_grad():
        baseline=torch.cat([model(x) for x in va['x'].split(128)])+parent['cup_bias']
        replay_baseline_mae=float((baseline.clamp(0,1)-va['y'])[replay_mask].abs().mean())
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    domain_weight=torch.tensor([6. if d=='consumed_task_detail' else 4. if d=='native_detail' else 1. for d in tr['domain']],device='cuda')
    train_near=torch.tensor([d!='historical_task_replay' and n in baseline_names for d,n in zip(tr['domain'],tr['name'])],device='cuda') & (tr['y']>.85)
    history=[];best=None;best_attempt=None;steps=0
    if fixed_cnn is not None:
        model.load_state_dict(fixed_cnn['cup_model']);model.eval()
        with torch.no_grad():
            pred=torch.cat([model(x) for x in va['x'].split(128)]);error=pred-va['y']
            row=dict(epoch=fixed_cnn['selected']['epoch'],mae=float(error[native].abs().mean()),near_mae=float(error[near].abs().mean()),
                near_max=float(error[near].abs().max()),near_overestimate=float(error[near].max()),historical_replay_mae=float(error[replay_mask].abs().mean()),
                unvalidated_size_mae=float(error[all_native & ~native].abs().mean()))
            assert row['mae']<gate['level_validation_mae'] and row['near_max']<gate['level_validation_near_max_error'] and row['near_overestimate']<gate['level_validation_near_overestimate']
            assert row['historical_replay_mae']<=max(.05,replay_baseline_mae+.015)
            best=best_attempt=dict(row=row,state=fixed_cnn['cup_model'],score=row['mae']+row['near_mae']+.3*row['near_max'])
    epochs=0 if fixed_cnn is not None else 240
    for epoch in range(1,epochs+1):
        model.train()
        if epoch in (121,181):
            for g in opt.param_groups:g['lr']/=3
        for ids in torch.randperm(len(tr['y']),device='cuda').split(64):
            x=tr['x'][ids].float();x=(x*(.98+.04*torch.rand(len(x),1,1,1,device='cuda'))).clamp(0,255)
            opt.zero_grad();error=model(x)-tr['y'][ids];loss=(error.square()*torch.where(tr['y'][ids]>.8,3.,1.)*domain_weight[ids]).mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();steps+=1
        model.eval()
        with torch.no_grad():
            pred=torch.cat([model(x) for x in va['x'].split(128)]);error=pred-va['y'];row=dict(epoch=epoch,
                mae=float(error[native].abs().mean()),near_mae=float(error[near].abs().mean()),near_max=float(error[near].abs().max()),near_overestimate=float(error[near].max()),
                historical_replay_mae=float(error[replay_mask].abs().mean()),unvalidated_size_mae=float(error[all_native & ~native].abs().mean()))
            history.append(row);score=row['mae']+row['near_mae']+.3*row['near_max']
            eligible=row['mae']<gate['level_validation_mae'] and row['near_max']<gate['level_validation_near_max_error'] and row['near_overestimate']<gate['level_validation_near_overestimate'] and row['historical_replay_mae']<=max(.05,replay_baseline_mae+.015)
            if task_mask.any():
                row.update(actual_pour_mae=float(error[task_mask].abs().mean()),actual_pour_near_max=float(error[task_near].abs().max()),actual_pour_near_overestimate=float(error[task_near].max()))
                eligible=eligible and row['actual_pour_mae']<gate['level_validation_mae'] and row['actual_pour_near_max']<gate['level_validation_near_max_error'] and row['actual_pour_near_overestimate']<gate['level_validation_near_overestimate']
            if eligible and task_mask.any():
                train_error=torch.cat([model(x) for x in tr['x'][train_near].split(128)])-tr['y'][train_near]
                row.update(training_near_max=float(train_error.abs().max()),training_near_overestimate=float(train_error.max()))
                eligible=row['training_near_max']<gate['level_validation_near_max_error'] and row['training_near_overestimate']<gate['level_validation_near_overestimate']
            if best_attempt is None or score<best_attempt['score']:best_attempt=dict(row=row,score=score,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            if eligible and (best is None or score<best['score']):best=dict(row=row,score=score,state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        if epoch%20==0:print('CUP_LEVEL',json.dumps(row),flush=True)
    with torch.no_grad():
        train_native=torch.tensor([d!='historical_task_replay' for d in tr['domain']],device='cuda')
        train_baseline=train_native & torch.tensor([n in baseline_names for n in tr['name']],device='cuda')
        all_features=color_geometry_features(tr['x'],tr['xy']).double();valid_features=color_geometry_features(va['x'][native],va['xy'][native]).double();geo=[]
        _,_,valid_mass=surface_features(va['x'][native])
        for domain,mask in (('all_native',train_native),('baseline_native',train_baseline),('native_and_replay',torch.ones_like(train_native))):
            for dimensions in (4,13,19,27):
                raw=all_features[mask,:dimensions];mean=raw.mean(0);scale=raw.std(0).clamp(min=.01);mean[0]=0;scale[0]=1
                x=(raw-mean)/scale;vx=(valid_features[:,:dimensions]-mean)/scale;y=tr['y'][mask].double()
                for ridge in (1e-4,.01,.1,1.,10.):
                    reg=torch.eye(dimensions,device='cuda',dtype=torch.float64)*ridge;reg[0,0]=1e-12
                    coef=torch.linalg.solve(x.T@x+reg,x.T@y);pred=torch.where(valid_mass>1.,(vx@coef).clamp(0,1),0.);err=pred-va['y'][native]
                    geo.append(dict(feature_kind='color_envelope_pose_v2' if dimensions==27 else 'color_envelope_v1',coef=coef.cpu().tolist(),mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),
                        dimensions=dimensions,bias=0.,ridge=ridge,training_domain=domain,mae=float(err.abs().mean()),max_error=float(err.abs().max())))
        # Four predefined nonlinear candidates: two feature sets x two native
        # domains, fixed bandwidth and ridge. One prototype per training
        # stationary scene or task pause/level bin avoids weighting a long hold
        # as hundreds of independent scenes and never averages distinct levels.
        from pour_guard_vision import color_geometry_readout
        retained=copy.deepcopy(parent['surface_geometry'])
        pred=color_geometry_readout(va['x'][native],retained,va['xy'][native]);pred=torch.where(valid_mass>1.,pred,0.)
        err=pred-va['y'][native]
        retained.update(retained_from_parent=True,mae=float(err.abs().mean()),max_error=float(err.abs().max()))
        geo.append(retained)
        for domain,mask in (('all_native',train_native),('baseline_native',train_baseline)):
            names=sorted({n for n,keep in zip(tr['prototype'],mask.tolist()) if keep});prototypes=[];targets=[]
            for name in names:
                ids=torch.tensor([n==name for n in tr['prototype']],device='cuda')&mask
                prototypes.append(all_features[ids].mean(0));targets.append(tr['y'][ids].double().mean())
            targets=torch.stack(targets)
            for dimensions in (13,27):
                raw=torch.stack(prototypes)[:,:dimensions];mean=raw.mean(0);scale=raw.std(0).clamp(min=.01);mean[0]=0;scale[0]=1
                centers=(raw-mean)/scale;distance=(centers[:,None]-centers[None]).square().mean(-1)
                kernel=torch.exp(-distance/2);bias=targets.mean();alpha=torch.linalg.solve(kernel+.01*torch.eye(len(names),device='cuda',dtype=torch.float64),targets-bias)
                fitted=dict(feature_kind='color_kernel_v3',dimensions=dimensions,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),
                    centers=centers.cpu().tolist(),alpha=alpha.cpu().tolist(),bias=float(bias),length_scale=1.,ridge=.01,
                    training_domain=domain,training_episodes=names,parameter_count=len(names)+1)
                pred=color_geometry_readout(va['x'][native],fitted,va['xy'][native]);pred=torch.where(valid_mass>1.,pred,0.)
                err=pred-va['y'][native];fitted.update(mae=float(err.abs().mean()),max_error=float(err.abs().max()));geo.append(fitted)
        geometry=min(geo,key=lambda g:g['max_error'])
    passed=best is not None and geometry['max_error']<gate['geometry_validation_max_error']
    public_geo=lambda g:{k:v for k,v in g.items() if k not in ('coef','mean','scale','centers','alpha','training_episodes')}
    color_parameters=geometry.get('parameter_count',geometry['dimensions'])
    report=dict(passed=passed,scope='baseline_cup_only',selected=None if best is None else best['row'],geometry=public_geo(geometry),geometry_candidates=[public_geo(g) for g in geo],
        primary_native_validation=dict(episodes=len({n for n,m in zip(va['name'],native.tolist()) if m}),images=int(native.sum()),near_full_images=int(near.sum())),
        actual_pour_validation=dict(episodes=len({n for n,m in zip(va['name'],task_mask.tolist()) if m}),images=int(task_mask.sum()),near_full_images=int(task_near.sum()),scope='Explicitly consumed development validation; not an independent test result.'),
        counts={s:dict(episodes=len(set(a['name'])),images=len(a['y']),min_target=float(a['y'].min()),max_target=float(a['y'].max()),
            domains={d:dict(images=sum(v==d for v in a['domain']),episodes=len({n for n,v in zip(a['name'],a['domain']) if v==d})) for d in set(a['domain'])}) for s,a in data.items()},
        historical_replay_baseline_mae=replay_baseline_mae,replay_scope='Original lower-resolution task views from 24/8 train/validation references. New-record weight 4, replay weight 1. Dynamic appearances retained; no reserved tests or generated detail pixels.',
        candidates=history,learned_parameters=sum(p.numel() for p in model.parameters())+color_parameters,epochs=epochs,optimizer_steps=steps,cnn_weights_retained=fixed_cnn is not None,
        newly_fitted_parameters=(0 if geometry.get('retained_from_parent') else color_parameters)+(0 if fixed_cnn is not None else sum(p.numel() for p in model.parameters())),
        method='Supervised RGB cup-level regression, with optional reuse of an already accepted CNN. Independent explicit-color ridge or RBF kernel regression; optional image-localized XY handles viewing angle. Kernel candidates use one prototype per training episode, fixed unit bandwidth and .01 ridge. No level-CNN features or runtime ground truth. Episode-disjoint validation selects candidates; no reserved test images.',
        target_source='Measured final-second surface median for stationary native recordings; consumed actual task observations use a causal stable half-second window with constant inventory and <=1 point instantaneous deviation. No future-final labels on earlier frames. Image-localized crops, declared cup dimensions; >=3 mm baseline localization error rejects the component. No truth-centered fitting crops or inventory-derived target lookup.',
        task_label_helper_sha256=sha(ROOT/'perception_task_labels.py'))
    attempt=best or best_attempt
    torch.save(dict(cup_model=attempt['state'],geometry=geometry,selected=attempt['row'],accepted=passed),ROOT/'cup-level-attempt.pt')
    report['attempt_checkpoint_sha256']=sha(ROOT/'cup-level-attempt.pt')
    model.load_state_dict(attempt['state']);model.eval();predictions=[]
    with torch.no_grad():
        from pour_guard_vision import color_geometry_readout
        for split,a in data.items():
            cnn=torch.cat([model(x) for x in a['x'].split(128)]).clamp(0,1)
            color=color_geometry_readout(a['x'],geometry,a['xy']);_,_,mass=surface_features(a['x']);color=torch.where(mass>1.,color,0.)
            predictions.extend(dict(name=n,split=split,domain=d,target=y,cnn=c,geometry=g,baseline_cup=n in baseline_names)
                for n,d,y,c,g in zip(a['name'],a['domain'],a['target'],cnn.cpu().tolist(),color.cpu().tolist()))
    write('cup-level-predictions.json',predictions)
    artifact=None if not passed else dict(cup_model=best['state'],cup_bias=0.,surface_geometry=geometry,
        settled_regressor=dict(columns=[4],mean=[0.],scale=[1.],coef=[1.],output_bias=0.,method='Identity of fitted settled-level CNN output; no additional learned parameters.'))
    return artifact,report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cnn-from',type=Path);args=parser.parse_args()
    fixed_path=None;fixed=None
    if args.cnn_from is not None:
        fixed_path=(ROOT/args.cnn_from).resolve();assert fixed_path.is_relative_to(ROOT) and fixed_path.suffix=='.pt' and fixed_path.is_file()
        fixed=torch.load(fixed_path,map_location='cpu',weights_only=False)
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error')
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    assert not (ROOT/'cup-component.json').exists();torch.set_num_threads(4);torch.manual_seed(261108)
    start=time.perf_counter();plan,seq,cal=collect();parent=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
    loc,loc_report=localize(seq,cal,plan['component_gates']);write('cup-localization-validation.json',loc_report)
    report=dict(passed=False,scope='baseline_cup_only',device=torch.cuda.get_device_name(),localization_passed=loc is not None,source_sha256=sha(Path(__file__)),
        admission_helper_sha256=sha(ROOT/'perception_fit_data.py'),vision_module_sha256=sha(ROOT/'staged/pour_guard_vision.py'))
    if loc is not None:
        level,level_report=fit_level(seq,cal,plan['component_gates'],parent,fixed_cnn=fixed)
        if fixed_path is not None:level_report['cnn_initialization']=dict(path=str(fixed_path),sha256=sha(fixed_path))
        write('cup-level-validation.json',level_report)
        report.update(level_passed=level is not None)
        if level is not None:
            torch.save(dict(localization=loc,level=level,detail_calibration=cal),ROOT/'cup-component.pt')
            report.update(passed=True,checkpoint_sha256=sha(ROOT/'cup-component.pt'))
    report['total_seconds']=time.perf_counter()-start;write('cup-component.json',report);print(json.dumps(report))
if __name__=='__main__':main()
