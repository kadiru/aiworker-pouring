"""Supervised wide-view localization, level fitting and constrained imitation."""
import copy
import hashlib
from itertools import product
import json
from pathlib import Path
import time
from collections import Counter

import numpy as np
import torch
from validate_pour_recording import h5py, decode
from pour_guard_vision import make_bank, locate, project, plane_point, bottle_base_from_neck, cup_crop
from pour_temporal_localization import consensus
from pour_guard_models import CupEstimator
from pour_settled_model import features, predict
from pour_workspace_motion import fit as fit_motion
from pour_workspace_study import RUNTIME_SOURCES as PARENT_SOURCES
from pour_wide_study import ROOT, PARENT, PARENT_SHA, PREFIX, REFERENCE_PREFIX, plan

RUNTIME_SOURCES=tuple(dict.fromkeys(PARENT_SOURCES+(
    'pour_wide_study.py','pour_wide_train.py','pour_wide_motion.py','pour_wide_geometry.py','pour_recording.py',
    'run_fountain.py','robot_pour.py','pour_guard_actor.py','pour_guard_vision.py','pour_source_alignment.py','pour_wide_source.py','pour_wide_empty.py')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')


def sequences(saved):
    expected={s['name']:s for s in plan()['demonstrations']}
    result=[]; rejected=[]; seen=set()
    for path in sorted(Path('/workspace/outputs/fountain').glob(f'*/*{REFERENCE_PREFIX}demo_*/episode.hdf5')):
        with h5py.File(path,'r') as f:
            s=json.loads(f.attrs['spec']);m=json.loads(f.attrs['metrics']);d=f['data/demo_0']
            reference=s.get('reference_case',s['name'])
            if s.get('protocol')!=17:continue
            assert reference in expected and not s['policy_evaluation'] and not s['expert_recovery']
            for key in ('cup_world','source_xy','initial_ml','cup_initial_ml','expected_stop','study_split','mouth_z','vessel_appearance'):
                assert s[key]==expected[reference][key],(s['name'],key,'Reference fixture changed')
            assert f.attrs['status']=='complete' and s['task_camera_focal_m']==.030
            assert s['spacing']==.0018 and s['dt']==.0025 and s['study_split'] in ('train','validation')
            assert s['name'] not in seen;seen.add(s['name'])
            accepted=bool(m['grasp_verified'] and m['zero_spill'] and not m['gates'])
            if not accepted:
                rejected.append(dict(name=s['name'],gates=m['gates'],grasp=m['grasp_verified'],zero_spill=m['zero_spill']))
                continue
            indices=np.flatnonzero(d['time'][:]-d['time'][0]<=1.+1e-7)
            cal=dict(K=f['calibration/task/intrinsics'][:].tolist(),T=d['obs/task_world_T_camera'][0].tolist(),table_z=1.04)
            if not result:
                cal['source_reference_calibration']=copy.deepcopy(saved['calibration'])
                saved['calibration']=cal
            else:
                assert np.allclose(cal['K'],saved['calibration']['K']) and np.allclose(cal['T'],saved['calibration']['T'])
            phases=np.array([v.decode() for v in d['ground_truth/phase'][:]])
            held=np.flatnonzero((phases=='Lifting and checking grip')&d['ground_truth/grasp_verified'][:])[-6:]
            assert len(held)==6,(s['name'],'Need six held-neck calibration frames')
            neck_uv=[]
            for i in held:
                pose=d['ground_truth/bottle_pose'][i];x,y,z,w=pose[3:]/np.linalg.norm(pose[3:])
                up=np.array([2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)])
                neck_uv.append(project(pose[:3]+.135*up,saved['calibration']))
            result.append(dict(name=s['name'],reference_case=reference,split=s['study_split'],path=str(path),sha256=sha(path),
                images=np.stack([decode(d['obs/task_image'][i],'jpeg95') for i in indices]),
                times=d['time'][:][indices]-float(d['time'][0]),
                source_xyz=d['ground_truth/bottle_pose'][0,:3],cup_xyz=np.asarray(s['cup_world']),accepted_reference=True,
                held_indices=held.tolist(),held_images=np.stack([decode(d['next_obs/task_image'][i],'jpeg95') for i in held]),
                held_neck_uv=np.asarray(neck_uv)))
    write('reference-acceptance.json',dict(accepted=[e['name'] for e in result],rejected=rejected))
    assert {e['reference_case'] for e in result}==set(expected), 'Complete a successful retained reference for every planned layout before fitting'
    assert len({e['reference_case'] for e in result})==len(result), 'Duplicate successful reference targets'
    return result


def development_calibration(saved):
    """Consume audited development r0/r1 for localization and stable levels."""
    expected={s['name']+f'_r{r}':s for r in (0,1) for s in plan()['development']};result=[];manifest=[]
    integrity=ROOT/'development-recording-integrity.json'
    audit=json.loads(integrity.read_text());assert audit['native_checks_passed'] and not audit['pending_attempts']
    indexed={r['name']:r for r in audit['files']}
    for path in sorted(Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}dev_*/episode.hdf5')):
        with h5py.File(path) as f:
            spec=json.loads(f.attrs['spec']);name=spec['name']
            if not name.endswith(('_r0','_r1')):continue
            assert name in expected and spec['protocol']==17 and spec['study_split']=='development'
            assert spec['development_round'] in (0,1) and spec['policy_evaluation'] and f.attrs['status']=='complete'
            assert all(spec[k]==v for k,v in expected[name].items() if k!='name')
            digest=sha(path);assert indexed[name]['valid'] and indexed[name]['sha256']==digest
            d=f['data/demo_0'];indices=np.flatnonzero(d['time'][:]-d['time'][0]<=.35+1e-7)
            assert len(indices)==8
            assert np.allclose(f['calibration/task/intrinsics'][:],saved['calibration']['K'])
            assert np.allclose(d['obs/task_world_T_camera'][0],saved['calibration']['T'])
            result.append(dict(name=name,split='development',path=str(path),sha256=digest,
                images=np.stack([decode(d['obs/task_image'][i],'jpeg95') for i in indices]),
                times=d['time'][:][indices]-float(d['time'][0]),
                source_xyz=d['ground_truth/bottle_pose'][0,:3],cup_xyz=np.asarray(spec['cup_world'])))
            manifest.append(dict(name=name,path=str(path),sha256=digest,indices=indices.tolist(),
                use='Localization candidate/confidence calibration and stable-level regression supervision; no localization templates, empty-CNN or motion targets.'))
    assert len(result)==16 and {e['name'] for e in result}==set(expected),'All sixteen closed r0/r1 development recordings are required'
    write('localization-development-manifest.json',dict(records=manifest,integrity_sha256=sha(integrity),
        scope='Explicitly consumed development rounds zero and one; not held-out performance. Eight layouts repeated, sixteen recordings. Reserved tests remain untouched.'))
    return result


@torch.inference_mode()
def fit_localization(saved,seq):
    train=[e for e in seq if e['split']=='train'];reports={}
    for kind,height in [('source',.135),('cup',.038)]:
        anchors=[project(e[kind+'_xyz']+[0,0,height],saved['calibration']) for e in train]
        # Learn the supported tabletop rectangle from training labels only.
        # Twenty-five millimetres of padding permits unseen placements while
        # excluding robot/background edges that cannot be tabletop anchors.
        xy=np.stack([e[kind+'_xyz'][:2] for e in train]);lo=xy.min(0)-.025;hi=xy.max(0)+.025
        corners=[[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]]
        region=[project([*p,1.04+height],saved['calibration']).tolist() for p in corners]
        candidates=[]
        # Restrict source templates to the neck when nearby robot/background
        # edges dominate the wider image. Validation chooses the mask; runtime
        # matching still uses RGB only and the stored template support.
        masks=(None,(10,8,38,40),(12,8,36,36),(14,8,34,34)) if kind=='source' else (None,(10,2,70,24),(16,4,64,22),(10,0,70,16))
        clips=(None,.05,.1)
        aggregations=('max','top3') if kind=='source' else ('max',)
        for absolute,radius,mask_box,clip,aggregation in product((True,False),(32.,48.),masks,clips,aggregations):
            bank=make_bank([e['images'][0] for e in train],anchors,kind,aggregation=aggregation,
                           absolute_gradients=absolute,anchor_radius_px=radius,source_mask_box=mask_box if kind=='source' else None,
                           cup_mask_box=mask_box if kind=='cup' else None,gradient_clip=clip)
            bank['bounds']=[24,24,360,350]
            bank['initial_region_uv']=region
            bank['initial_workspace_xy_m']=[lo.tolist(),hi.tolist()]
            rows=[]
            for e in seq:
                image=e['images'][0];uv,score=locate(image,bank)
                xy=(bottle_base_from_neck(uv,saved['calibration']) if kind=='source' else
                    plane_point(uv,1.04+height,saved['calibration']))[:2]
                error=float(np.linalg.norm(xy-e[kind+'_xyz'][:2])*1000)
                negative=image.copy();x,y=np.rint(project(e[kind+'_xyz']+[0,0,height],saved['calibration'])).astype(int)
                dx,dy=(28,44) if kind=='source' else (44,22)
                negative[max(0,y-18):min(384,y+dy),max(0,x-dx):min(384,x+dx)]=180
                _,ns=locate(negative,bank)
                rows.append(dict(name=e['name'],split=e['split'],uv=uv.tolist(),score=score,error_mm=error,synthetic_negative_score=ns))
            minimum=min(r['score'] for r in rows);maximum=max(r['synthetic_negative_score'] for r in rows)
            threshold=(minimum+maximum)/2
            candidates.append(dict(absolute_gradients=absolute,anchor_radius_px=radius,source_mask_box=mask_box if kind=='source' else None,
                cup_mask_box=mask_box if kind=='cup' else None,
                gradient_clip=clip,aggregation=aggregation,
                initial_workspace_xy_m=bank['initial_workspace_xy_m'],
                min_positive=minimum,max_negative=maximum,threshold=threshold,margin=minimum-maximum,
                max_error_mm=max(r['error_mm'] for r in rows),rows=rows,bank=bank))
            print('LOCALIZATION',kind,absolute,radius,mask_box,clip,aggregation,'error',candidates[-1]['max_error_mm'],'margin',minimum-maximum,flush=True)
        eligible=[c for c in candidates if c['max_error_mm']<2. and c['margin']>.015]
        write(kind+'-localization-candidates.json',[{k:v for k,v in c.items() if k!='bank'} for c in candidates])
        assert eligible, 'No precise, occlusion-rejecting '+kind+' localization candidate'
        chosen=min(eligible,key=lambda c:(c['max_error_mm'],-c['margin']))
        saved[kind+'_bank']=chosen['bank'];saved['thresholds'][kind+'_localization_score']=chosen['threshold']
        reports[kind]={k:v for k,v in chosen.items() if k!='bank'}
    configs={};windows=[]
    for kind,height in [('source',.135),('cup',.038)]:
        threshold=saved['thresholds'][kind+'_localization_score']
        config=dict(window=20,votes=8,radius_px=1.5,fraction=.6,score_floor=max(.25,threshold-.2))
        for e in seq:
            history=[];accepted=None
            for i,im in enumerate(e['images']):
                uv,score=locate(im,saved[kind+'_bank']);history.append((uv.tolist(),score))
                accepted=consensus(history,threshold,config)
                if accepted:
                    xy=(bottle_base_from_neck(accepted['uv'],saved['calibration']) if kind=='source' else
                        plane_point(accepted['uv'],1.04+height,saved['calibration']))[:2]
                    error=float(np.linalg.norm(xy-e[kind+'_xyz'][:2])*1000)
                    assert error<2.,(kind,e['name'],error)
                    windows.append(dict(kind=kind,name=e['name'],split=e['split'],latency_s=float(e['times'][i]),error_mm=error))
                    break
            assert accepted,(kind,e['name'],'No consistent localization in calibration window')
        configs[kind]=config
    saved['localization_consensus']=dict(**configs,timeout_s=2.)
    report=dict(template_training_names=[e['name'] for e in train],families=reports,windows=windows,
        consumed_development_names=[e['name'] for e in seq if e['split']=='development'],
        method='Gradient template matching with a training-derived tabletop region, view-local support, learned reference banks, calibrated confidence and temporal consensus.',
        scope='Reference validation and explicitly consumed development calibration; no reserved test images. Synthetic occlusions are a check, not independent missing-object performance.')
    write('localization-validation.json',report)
    return report


@torch.inference_mode()
def fit_tracking(saved,seq):
    """Separate lifted-neck templates avoid table-view background contamination."""
    train=[e for e in seq if e['split']=='train'];candidates=[]
    projections={};training_drift=[]
    for e in seq:
        initial,_=locate(e['images'][0],saved['source_bank'])
        expected=project(bottle_base_from_neck(initial,saved['calibration'])+[0,0,.135+.044],saved['calibration'])
        projections[e['name']]=(initial,expected)
        if e['split']=='train':training_drift.extend(np.linalg.norm(e['held_neck_uv']-expected,axis=1).tolist())
    # Closing fingers can center the free bottle laterally. Calibrate that
    # observed grasp displacement from training labels, without widening the
    # image-search window or weakening the measured neck-error requirement.
    radius=max(12.,float(np.quantile(training_drift,.99))+2.)
    assert radius<=18.,'Observed grasp drift exceeds the bounded tracking calibration'
    calibration=dict(radius_px=radius,training_p99_drift_px=float(np.quantile(training_drift,.99)),
        training_max_drift_px=max(training_drift),margin_px=2.,maximum_allowed_px=18.,
        source='Training-reference neck labels versus RGB-localized expected lift; no validation/test radius fitting.')
    for absolute,box in product((True,False),(None,(10,8,38,40),(12,8,36,36),(14,8,34,34))):
        bank=make_bank([e['held_images'][-1] for e in train],[e['held_neck_uv'][-1] for e in train],
            'source',aggregation='max',absolute_gradients=absolute,source_mask_box=box)
        bank['bounds']=[24,24,360,350];rows=[];episodes=[]
        for e in seq:
            initial,expected=projections[e['name']]
            delta=expected-initial;votes=0
            for i,image,truth in zip(e['held_indices'],e['held_images'],e['held_neck_uv']):
                uv,score=locate(image,bank,tracking=True,expected_uv=expected)
                distance=float(np.linalg.norm(uv-expected));follows=distance<radius
                moved=np.linalg.norm(delta)>=8. and np.dot(uv-initial,delta)>.65*np.dot(delta,delta)
                passed=bool(score>.23 and follows and moved);votes+=passed
                rows.append(dict(name=e['name'],split=e['split'],index=i,uv=uv.tolist(),score=score,
                    follows_distance_px=distance,visual_predicate=passed,neck_error_px=float(np.linalg.norm(uv-truth))))
            episodes.append(dict(name=e['name'],split=e['split'],votes=votes,frames=len(e['held_indices'])))
        valid=[r['neck_error_px'] for r in rows if r['split']=='validation']
        candidates.append(dict(absolute_gradients=absolute,source_mask_box=box,bank=bank,rows=rows,episodes=episodes,
            min_votes=min(e['votes'] for e in episodes),max_error_px=max(r['neck_error_px'] for r in rows),
            validation_mean_error_px=float(np.mean(valid)),validation_max_error_px=max(valid)))
    write('source-tracking-candidates.json',[{k:v for k,v in c.items() if k!='bank'} for c in candidates])
    eligible=[c for c in candidates if c['min_votes']>=5 and c['max_error_px']<=6.]
    assert eligible,'Held-neck tracking needs five of six positive views per reference and <=6px error before native deployment'
    chosen=min(eligible,key=lambda c:c['validation_mean_error_px']+.25*c['validation_max_error_px'])
    saved['source_tracking_bank']=chosen['bank']
    saved['thresholds']['grasp_follow_radius_px']=radius
    result=dict(**{k:v for k,v in chosen.items() if k!='bank'},template_training_names=[e['name'] for e in train],
        follow_calibration=calibration,
        template_indices={e['name']:e['held_indices'][-1] for e in train},
        method='Distinct RGB gradient template bank learned from lifted bottles; disjoint validation chooses the mask.',
        scope='Simulator neck positions label training/error only. Runtime uses RGB and its expected lift projection; finger/own-motion gates remain. Positive tracking coverage is not a missing-bottle rejection test.')
    write('source-tracking-validation.json',result)
    return result


def level_data(saved,seq,development=()):
    old=torch.load(PARENT/'workspace-cache.pt',map_location='cpu',weights_only=False)
    arrays={s:{k:[] for k in ('image','target','cup_xy','episode')} for s in ('train','validation')}
    manifest=copy.deepcopy(old['manifest'])
    for e in list(seq)+list(development):
        p=Path(e['path']);rows=json.loads((p.parent/'samples.json').read_text());last=rows[-1]
        consumed=e['split']=='development';split='train' if consumed else e['split']
        decisions=json.loads((p.parent/'policy-decisions.json').read_text()) if consumed else None
        uv,_=locate(e['images'][0],saved['cup_bank'])
        xy=plane_point(uv,1.078,saved['calibration'])[:2].tolist()
        # A few returning droplets must not discard an otherwise unchanged
        # settled pause. Only consumed development permits this bounded label
        # tolerance; reference/validation sampling keeps its original rule.
        inventory_tolerance=(8. if consumed else .5)*.0018**3*1e6
        changed=[r['t'] for r in rows if abs(r['cup_ml']-last['cup_ml'])>inventory_tolerance]
        stable=max(changed,default=0.)+.35;used=[]
        with h5py.File(p,'r') as f:
            d=f['data/demo_0']
            for i in range(0,len(rows),2):
                # Upright draining above the cup can hide its surface. Use
                # the settled view after transfer back to the source position.
                if rows[i]['t']<stable:continue
                if consumed:
                    if decisions[i]['mode'] not in ('Pour control: settle','Setting bottle down','Releasing bottle','Holding completed pose','Aborting unconfirmed grasp'):continue
                elif rows[i]['phase'] not in ('Setting bottle down','Releasing bottle'):continue
                image=cup_crop(decode(d['next_obs/task_image'][i],'jpeg95'),xy,saved['calibration'])
                for k,v in dict(image=image,target=last['cup_fill_fraction'],cup_xy=xy,episode=e['name']).items():
                    arrays[split][k].append(v)
                used.append(i)
        assert used,e['name']
        manifest.append(dict(name=e['name'],split=split,recording_split=e['split'],consumed_development=consumed,path=e['path'],sha256=e['sha256'],frames=len(used),indices=used,
            target_final_height=last['cup_fill_fraction'],cup_crop_xy=xy,stable_inventory_after_s=stable,
            inventory_tolerance_nominal_ml=inventory_tolerance,
            visibility_window='Stable-inventory pause, placement and final views from consumed development' if consumed else 'Setting down/releasing after the bottle has transferred away from the cup',
            label='Simulator-measured final settled rim fraction within the recorded inventory tolerance; RGB-localized crop',scripted_reference=not consumed))
    data={}
    for split,a in arrays.items():
        data[split]={k:old['data'][split][k]+v if k=='episode' else torch.cat((old['data'][split][k],
            torch.from_numpy(np.asarray(v,dtype=np.uint8 if k=='image' else np.float32)))) for k,v in a.items()}
    allowed={e['name'] for e in development}
    assert all(not e['name'].startswith(PREFIX+'test') and (not e['name'].startswith(PREFIX+'dev') or e['name'] in allowed) for e in manifest)
    assert not set(data['train']['episode'])&set(data['validation']['episode'])
    write('data-manifest.json',manifest)
    result=dict(data=data,manifest=manifest);torch.save(result,ROOT/'wide-cache.pt')
    return result


@torch.inference_mode()
def fit_level(saved,bundle):
    model=CupEstimator().cuda().eval();model.load_state_dict(saved['cup_model']);matrices={}
    for split,d in bundle['data'].items():
        xs=[]
        for start in range(0,len(d['target']),128):
            image=d['image'][start:start+128].cuda();cnn=(model(image)+saved['cup_bias']).clamp(0,1)
            xs.append(features(image,cnn,d['cup_xy'][start:start+128].cuda()))
        matrices[split]=torch.cat(xs).double()
    tr,va=bundle['data']['train'],bundle['data']['validation']
    x,vx=matrices['train'],matrices['validation'];y,vy=tr['target'].cuda().double(),va['target'].cuda().double()
    counts=Counter(tr['episode']);weights=y.new_tensor([1./counts[n] for n in tr['episode']])*torch.where(y>.8,3.,1.)
    # The inherited narrow-workspace set otherwise dominates the newly
    # collected camera perspectives. Ratios were selected in an explicit
    # reference/development-only diagnostic; reserved scenes are never read.
    domain_weights={'parent':1.,'wide_reference':4.,'consumed_development':16.}
    weights*=y.new_tensor([domain_weights['consumed_development'] if n.startswith(PREFIX+'dev')
        else domain_weights['wide_reference'] if n.startswith(REFERENCE_PREFIX+'demo_')
        else domain_weights['parent'] for n in tr['episode']])
    weights/=weights.mean();candidates=[]
    feature_sets=([0,1,2,3],[0,1,2,3,4],list(range(7)),[0,4]+list(range(7,21)),list(range(21)),
        list(range(21))+list(range(21,25))+list(range(39,43))+list(range(57,60)),list(range(60)))
    for columns in feature_sets:
        a=x[:,columns];mean=a.mean(0);scale=a.std(0).clamp(min=.001);mean[0]=0.;scale[0]=1.;a=(a-mean)/scale
        for ridge in (.001,.01,.1,1.):
            penalty=torch.eye(len(columns),device='cuda',dtype=torch.float64)*ridge;penalty[0,0]=1e-9
            coef=torch.linalg.solve(a.T@(a*weights[:,None])+penalty,a.T@(y*weights))
            fitted=dict(columns=columns,mean=mean.cpu().tolist(),scale=scale.cpu().tolist(),coef=coef.cpu().tolist(),ridge=ridge,
                training_domain_weights=domain_weights)
            error=predict(vx,fitted)-vy;near=vy>.8
            development_mask=torch.tensor([n.startswith(PREFIX+'dev') for n in tr['episode']],device='cuda')
            development_error=predict(x,fitted)-y
            development_near=development_mask&(y>.8)
            wide=torch.tensor([n.startswith(REFERENCE_PREFIX+'demo_') for n in va['episode']],device='cuda')
            stats=dict(mae=float(error.abs().mean()),near_full_mae=float(error[near].abs().mean()),
                near_full_p95_abs=float(torch.quantile(error[near].abs(),.95)),
                near_full_p95_overestimate=float(torch.quantile(error[near],.95)),
                wide_mae=float(error[wide].abs().mean()),wide_near_full_max_abs=float(error[wide&near].abs().max()),
                wide_near_full_p95_overestimate=float(torch.quantile(error[wide&near],.95)),
                consumed_development_mae=float(development_error[development_mask].abs().mean()) if development_mask.any() else None,
                consumed_development_max_abs=float(development_error[development_mask].abs().max()) if development_mask.any() else None,
                consumed_development_near_full_max_abs=float(development_error[development_near].abs().max()) if development_near.any() else None,
                consumed_development_near_full_max_overestimate=float(development_error[development_near].max()) if development_near.any() else None)
            candidates.append(dict(fit=fitted,metrics=stats,score=stats['near_full_mae']+.5*stats['near_full_p95_abs']+stats['wide_mae']))
    eligible=[c for c in candidates if (c['metrics']['consumed_development_max_abs'] is None or c['metrics']['consumed_development_max_abs']<.04)
        and (c['metrics']['consumed_development_near_full_max_abs'] is None or c['metrics']['consumed_development_near_full_max_abs']<.02)]
    write('level-candidates.json',candidates)
    assert eligible,'Improve consumed stable-level calibration: <4 points overall and <2 points near full are required'
    chosen=min(eligible,key=lambda c:c['score'])
    full_target=.90+max(.01,chosen['metrics']['near_full_p95_overestimate'],
                       chosen['metrics']['wide_near_full_p95_overestimate'],
                       chosen['metrics']['consumed_development_near_full_max_overestimate'] or 0.)+.005
    assert full_target<=.94 and chosen['metrics']['wide_near_full_max_abs']<.04,'Improve wider settled-level uncertainty before native use'
    saved['settled_regressor']=chosen['fit'];saved['settled_validation']=chosen['metrics']
    saved['thresholds']['full_settled_target']=full_target
    return chosen,dict(train_episodes=len(set(tr['episode'])),validation_episodes=len(set(va['episode'])),
        train_frames=len(y),validation_frames=len(vy),learned_level_parameters=len(chosen['fit']['coef']),
        consumed_development_level_episodes=len({n for n in tr['episode'] if n.startswith(PREFIX+'dev')}),
        training_domain_weights=domain_weights,
        objective='Episode-balanced weighted squared settled-height error plus ridge penalty')


@torch.inference_mode()
def train(state):
    assert not (ROOT/'final-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}test*/parameters.json'))
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name(0)
    assert sha(PARENT/'best.pt')==PARENT_SHA
    torch.set_num_threads(4);torch.manual_seed(20261105);started=time.monotonic()
    saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
    state.update(phase='Fitting wider-table RGB localization',learning=dict(stage='localization'))
    seq=sequences(saved);development=development_calibration(saved);localization=fit_localization(saved,seq+development)
    tracking=fit_tracking(saved,seq)
    state.update(phase='Fitting wider-table settled level and motion imitation',learning=dict(stage='supervised_fit'))
    bundle=level_data(saved,seq,development);chosen,training=fit_level(saved,bundle)
    from pour_wide_source import probe,presence_coverage
    from pour_wide_empty import fit_readout
    localized={e['name']:e['cup_crop_xy'] for e in bundle['manifest'] if e['name'].startswith(REFERENCE_PREFIX+'demo_')}
    source_fit=fit_readout(saved,seq,localized)
    saved['wide_empty_readout']=source_fit
    saved['source_presence_fusion']='held_raw_or_aligned_v1'
    write('empty-readout-summary.json',source_fit)
    empty_coverage=probe(saved,seq,localized)
    write('empty-detector-coverage.json',empty_coverage)
    assert empty_coverage['validation_negative_frames']>0,'Empty detection needs observed eligible nonempty negatives'
    assert not empty_coverage['validation_empty_episodes_without_detection'], 'Validate empty recognition on every wider validation pour before native use'
    assert not empty_coverage['false_positive_frames'], 'Wide validation frames falsely identify remaining liquid as empty'
    presence=presence_coverage(saved,seq+development)
    write('source-presence-coverage.json',presence)
    assert not presence['false_lost_source_episodes'],'Fused source presence loses a verified held bottle before stopping'
    # Expand the old angle cap only where at least three successful training
    # demonstrations reached that phase. Validation/test caps do not set it.
    training_caps=[]
    for e in seq:
        if e['split']!='train':continue
        with h5py.File(e['path']) as f:
            d=f['data/demo_0'];mask=~d['ground_truth/stop_command'][:].astype(bool)
            training_caps.append(float(d['ground_truth/command_angle_deg'][:][mask].max()))
    parent_motion=dict(saved['pour_primitive']);old_cap=parent_motion['phase_cap_deg']
    parent_motion['phase_cap_deg']=min(150.,max(old_cap,sorted(training_caps)[-3]))
    saved['pour_primitive'],motion=fit_motion(parent_motion,seq,root=ROOT,prefix=REFERENCE_PREFIX)
    motion['cap_selection']=dict(parent_deg=old_cap,selected_deg=parent_motion['phase_cap_deg'],training_caps_deg=training_caps,
        method='Preserve parent cap or expand to third-largest successful new training-demonstration phase, limited to150 degrees; no validation/test targets.')
    write('motion-summary.json',motion)
    saved['pour_primitive']['slow_rate_deg_s']=4.
    saved['source_frame_normalization']='cup_aligned_v2'
    saved['approach_clearance_m']=.07;saved['grasp_projection_consistency']=True
    saved['level_transit_approach']=True
    saved['project_pour_kinematics']=True;saved['pour_mouth_z']=.052
    saved['centered_return']=True
    saved['early_return_seconds']=4.3
    saved['gentle_release_retreat']=dict(place_s=1.5,release_s=.6,retreat_s=2.)
    saved['thresholds'].update(approach_seconds=2.,approach_tracking_wait_s=1.,settled_geometry_tolerance=.10)
    saved['parent_training_step']=saved.get('step');saved['step']=0
    saved['parent_checkpoint_sha256']=PARENT_SHA
    saved['model_version']='visual_wide_hierarchy_v21'
    saved['method']='Supervised RGB template localization calibrated with consumed development scenes, settled-height ridge regression and a fine-tuned wrist-crop empty CNN; contextual motion behavioral cloning with IK constraints, dual-view presence plus encoder hold, feedback stopping and programmed upright return.'
    saved['sources']={n:sha(Path('/workspace/src')/n) for n in RUNTIME_SOURCES}
    training.update(motion_training_episodes=len(motion['train_episodes']),motion_validation_episodes=len(motion['validation_episodes']),
        localization_template_episodes=sum(e['split']=='train' for e in seq),localization_validation_episodes=sum(e['split']=='validation' for e in seq),
        localization_development_calibration_episodes=len(development),localization_development_names=[e['name'] for e in development],
        lifted_tracking_template_episodes=len(tracking['template_training_names']),
        lifted_tracking_validation_max_error_px=tracking['validation_max_error_px'],
        frozen_existing_visual_encoders=True,empty_encoder_finetuned=True,learned_empty_parameters=source_fit['learned_parameters'],empty_readout_fit=source_fit,
        kinematic_projection='Programmed geometry constraint; no live object or particle state')
    saved['training_device']=dict(name=torch.cuda.get_device_name(0),visible_cuda_devices=1)
    saved['wide_training']=training
    torch.save(saved,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(ROOT/'best.pt')
    summary=dict(checkpoint_sha256=sha(ROOT/'best.pt'),parent_checkpoint_sha256=PARENT_SHA,method=saved['method'],training=training,
        device=saved['training_device'],chosen=chosen,thresholds=saved['thresholds'],
        localization={k:v for k,v in localization.items() if k!='families'},motion={k:v for k,v in motion.items() if k!='regularizer_validation'},
        elapsed_s=time.monotonic()-started,data_manifest_sha256=sha(ROOT/'data-manifest.json'),sources=saved['sources'],
        scene_plan_sha256=sha(ROOT/'frozen-plan.json'),reference_repair_plan_sha256=sha(ROOT/'reference-repair-plan.json'),
        empty_detector_coverage={k:v for k,v in empty_coverage.items() if k!='episodes'},
        source_presence_coverage={k:v for k,v in presence.items() if k!='episodes'},
        source_presence_fusion=saved['source_presence_fusion'],early_return_seconds=saved['early_return_seconds'],
        gentle_release_retreat=saved['gentle_release_retreat'],consumed_development_rounds=[0,1],
        localization_development_manifest_sha256=sha(ROOT/'localization-development-manifest.json'),
        camera_focal_m=.030,source_frame_normalization=saved['source_frame_normalization'],no_privileged_runtime_inputs=True,
        task_goal='Sustained verified grasp, actual pour, correct full/empty stop, upright return/release and zero measured loss; original scoring unchanged.')
    write('model-summary.json',summary)
    state.update(phase='Wider-table supervised fitting complete',learning=dict(stage='fit_complete',elapsed_s=summary['elapsed_s']))
    del saved,bundle;torch.cuda.empty_cache()
    return str(ROOT/'best.pt')
