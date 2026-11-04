"""Expanded-position demonstrations and consumed parent data; no W30 tests."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from validate_pour_recording import decode,h5py
from pour_guard_vision import make_bank,project,locate,plane_point,cup_crop
from pour_temporal_localization import consensus
from pour_workspace_study import ROOT,PARENT,PARENT_SHA,PREFIX,RANGES,plan,reference_specs

PARENT_VALIDATION={f'C30_A_test_{i:02}' for i in (2,7,9,17,21,27)}


def sequences():
    expected={s['name']:s for s in reference_specs()}; result=[]
    files=sorted(Path('/workspace/outputs/fountain').glob('*/*C30_A_test_*/episode.hdf5'))
    files+=sorted(Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}demo_*/episode.hdf5'))
    for p in files:
        with h5py.File(p,'r') as f:
            s=json.loads(f.attrs['spec']);name=s['name'];m=json.loads(f.attrs['metrics']);d=f['data/demo_0']
            assert f.attrs['status']=='complete' and f.attrs['camera_rig']=='A_task_wrist_v4'
            assert s['spacing']==.0018 and s['dt']==.0025
            if name.startswith('C30_'):
                assert m['policy']['checkpoint_sha256']==PARENT_SHA
                split='validation' if name in PARENT_VALIDATION else 'train'
            else:
                assert name in expected and not s['policy_evaluation'] and not s['expert_recovery']
                for k,v in expected[name].items():assert s[k]==v,(name,k)
                split=s['study_split']
            ix=np.flatnonzero(d['time'][:]-d['time'][0]<=1.+1e-7)
            result.append(dict(name=name,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                reference_case=s.get('reference_case',name),
                split=split,images=np.stack([decode(d['obs/task_image'][i],'jpeg95') for i in ix]),
                times=d['time'][:][ix],source_xyz=d['ground_truth/bottle_pose'][0,:3],cup_xyz=np.asarray(s['cup_world']),
                accepted_reference=bool(m['grasp_verified'] and m['zero_spill'] and not m['gates']),
                rejection_gates=m['gates'],zero_spill=m['zero_spill']))
    assert len(result)==30+len(expected) and {e['name'] for e in result if e['name'].startswith(PREFIX)}==set(expected)
    accepted={e['reference_case'] for e in result if e['name'].startswith(PREFIX) and e['accepted_reference']}
    assert accepted=={s['name'] for s in plan()['demonstrations']},'Every planned reference case needs a successful retained original or explicitly named repair before fitting'
    return result


def cover_workspace(saved):
    points={'source':[],'cup':[]};cal=saved['calibration']
    for dx,dy,sx,sy in itertools.product(*[(-r,r) for r in RANGES]):
        cx,cy=.49+dx,-.15+dy
        points['source'].append(project([cx-sy,cy-.12+sx,1.1752],cal))
        points['cup'].append(project([cx,cy,1.078],cal))
    bounds={}
    for kind,uv in points.items():
        a=np.asarray(uv);bounds[kind]=[*(np.floor(a.min(0)-6).astype(int).tolist()),*(np.ceil(a.max(0)+6).astype(int).tolist())]
        def extend(bank):
            old=bank['bounds'];b=bounds[kind]
            bank['bounds']=[min(old[0],b[0]),min(old[1],b[1]),max(old[2],b[2]),max(old[3],b[3])]
            for child in bank.get('alternatives',[]):extend(child)
        extend(saved[kind+'_bank'])
    return dict(projected_corner_bounds=bounds,rule='All16 domain corners plus6pixel margin; recursively update every template family, not only the wrapper.')


@torch.inference_mode()
def fit_banks(saved,seq):
    coverage=cover_workspace(saved);train=[e for e in seq if e['name'].startswith(PREFIX) and e['split']=='train' and e['name']==e['reference_case']]
    assert len(train)==16;report={}
    for kind,height in (('source',.135),('cup',.038)):
        views=([e for e in seq if e['split']=='train' and e['name']==e['reference_case']] if kind=='source' else train)
        anchors=[project(e[kind+'_xyz']+[0,0,height],saved['calibration']) for e in views]
        bank=make_bank([e['images'][0] for e in views],anchors,kind,aggregation='max',
            absolute_gradients=kind=='source',anchor_radius_px=28. if kind=='source' else None)
        prior=saved[kind+'_bank'];bank['bounds']=prior['bounds']
        original_threshold=saved['thresholds'][kind+'_localization_score']
        if kind=='source':
            # Glass edges reverse contrast against the robot/background across
            # the wider domain. Use unsigned gradients and a view-local maximum;
            # a higher confidence anchor rejects the measured synthetic negatives.
            assert len(views)==40
            saved[kind+'_bank']=bank;families=[]
            saved['thresholds']['source_localization_score']=.70
        else:
            families=copy.deepcopy(prior.get('alternatives',[prior]))
            saved[kind+'_bank']=dict(prior,alternatives=families+[bank])
        report[kind]=dict(new_templates=len(views),retained_families=len(families),training_names=[e['name'] for e in views],
            bounds=bank['bounds'],original_anchor_threshold=original_threshold,
            anchor_threshold=saved['thresholds'][kind+'_localization_score'],aggregation=bank['aggregation'],
            absolute_gradients=bank['absolute_gradients'],anchor_radius_px=bank['anchor_radius_px'])
    return dict(coverage=coverage,families=report,validation_excluded_from_templates=True)


def build_data(saved,seq):
    # Reuse only the parent's named training/validation tensors, never a glob
    # admitting this study's development/evaluation outcomes by accident.
    old=torch.load(PARENT/'reliable-cache-v1.pt',map_location='cpu',weights_only=False)
    arrays={s:{k:[] for k in ('image','target','cup_xy','episode')} for s in ('train','validation')}
    manifest=copy.deepcopy(old['manifest']);decisions=[]
    for e in seq:
        if not e['accepted_reference']:
            decisions.append(dict(name=e['name'],accepted=False,gates=e['rejection_gates'],zero_spill=e['zero_spill']));continue
        p=Path(e['path']);rows=json.loads((p.parent/'samples.json').read_text())
        detections=[locate(image,saved['cup_bank']) for image in e['images']]
        accepted=consensus([(uv.tolist(),score) for uv,score in detections],saved['thresholds']['cup_localization_score'],saved['localization_consensus']['cup'])
        assert accepted,('cup_crop_localization',e['name'])
        xy=plane_point(accepted['uv'],1.078,saved['calibration'])[:2].tolist()
        error=float(np.linalg.norm(np.asarray(xy)-e['cup_xyz'][:2])*1000)
        assert error<2.,(e['name'],error)
        final=rows[-1];changed=[x['t'] for x in rows if abs(x['cup_ml']-final['cup_ml'])>.5*.0018**3*1e6]
        stable_after=max(changed,default=0.)+.35;used=[];split=e['split']
        with h5py.File(p,'r') as f:
            d=f['data/demo_0'];assert len(rows)==len(d['actions'])
            for i in range(0,len(rows),2):
                if rows[i]['t']<stable_after:continue
                assert abs(rows[i]['t']-float(d['next_time'][i]))<1e-5
                image=cup_crop(decode(d['next_obs/task_image'][i],'jpeg95'),xy,saved['calibration'])
                for k,v in dict(image=image,target=final['cup_fill_fraction'],cup_xy=xy,episode=e['name']).items():arrays[split][k].append(v)
                used.append(i)
        assert used,e['name']
        manifest.append(dict(name=e['name'],path=e['path'],sha256=e['sha256'],split=split,frames=len(used),indices=used,
            stable_inventory_after_s=stable_after,target_final_height=final['cup_fill_fraction'],cup_crop_xy=xy,
            label='final settled particle surface; stable-inventory frames, RGB-localized cup crop',
            consumed_previous_episode=e['name'].startswith('C30_'),scripted_reference=e['name'].startswith(PREFIX)))
        decisions.append(dict(name=e['name'],accepted=True,cup_error_mm=error,frames=len(used),split=split))
    data={}
    for split,a in arrays.items():
        data[split]={k:old['data'][split][k]+v if k=='episode' else torch.cat((old['data'][split][k],torch.from_numpy(np.asarray(v,dtype=np.uint8 if k=='image' else np.float32)))) for k,v in a.items()}
    assert all(not e['name'].startswith(PREFIX+'test') and not e['name'].startswith(PREFIX+'dev') for e in manifest)
    (ROOT/'data-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'data-acceptance.json').write_text(json.dumps(decisions,indent=2)+'\n')
    result=dict(data=data,manifest=manifest)
    torch.save(result,ROOT/'workspace-cache.pt')
    return result


@torch.inference_mode()
def probe_empty_detector(saved,seq):
    """Measure the retained empty detector on new views; this performs no fit."""
    from pour_stop_models import LiquidDetector
    from pour_stop_data import crops
    model=LiquidDetector().cuda().eval();model.load_state_dict(saved['source_detector'])
    episodes=[]
    for e in seq:
        p=Path(e['path']);rows=json.loads((p.parent/'samples.json').read_text());samples=[]
        with h5py.File(p,'r') as f:
            d=f['data/demo_0']
            for i in range(0,len(rows),3):
                r=rows[i]
                if r['bottle_tilt_deg']<95 or r['source_outflow_ml_s']>=3:continue
                c,s,w=crops(decode(d['next_obs/task_image'][i],'jpeg95'),decode(d['next_obs/wrist_image'][i],'jpeg95'))
                pred=model(*[torch.as_tensor(x,device='cuda')[None] for x in (c,s,w)])
                samples.append(dict(t=r['t'],source_ml=r['bottle_ml'],empty_probability=float(pred['empty_logit'][0].sigmoid()),present_probability=float(pred['present_logit'][0].sigmoid())))
        episodes.append(dict(name=e['name'],split=e['split'],samples=samples))
    valid=[s for e in episodes if e['split']=='validation' for s in e['samples']]
    positive=[s for s in valid if s['source_ml']<=.12];negative=[s for s in valid if s['source_ml']>=.5]
    report=dict(method='Frozen parent empty/presence detector replay on recorded views; no fitting or threshold selection',episodes=episodes,
        validation_positive_frames=len(positive),validation_negative_frames=len(negative),
        validation_positive_above_090=sum(s['empty_probability']>=.90 and s['present_probability']>=.7 for s in positive),
        validation_negative_above_090=sum(s['empty_probability']>=.90 and s['present_probability']>=.7 for s in negative),
        caveat='Offline tilt/outflow filtering uses labels. Sparse frame counts and reference motions do not replace closed-loop development or the reserved task evaluation.')
    (ROOT/'empty-detector-coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    del model;torch.cuda.empty_cache()
    return {k:v for k,v in report.items() if k!='episodes'}
