"""Consume the completed parent study; never admit reserved C30 evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from validate_pour_recording import decode, h5py
from pour_guard_vision import cup_crop, locate, plane_point
from pour_reliable_study import ROOT, PARENT, PARENT_SHA

VALIDATION = {'L30_A_test_01','L30_A_test_06','L30_A_test_11','L30_A_test_15','L30_A_test_21','L30_A_test_27'}


def build_data(saved):
    cache=ROOT/'reliable-cache-v1.pt'
    if cache.exists():
        return torch.load(cache,map_location='cpu',weights_only=False)
    assert hashlib.sha256((PARENT/'best.pt').read_bytes()).hexdigest()==PARENT_SHA
    old=torch.load(PARENT/'settled-cache-v2-high-fill.pt',map_location='cpu',weights_only=False)
    arrays={s:{k:[] for k in ('image','target','cup_xy','episode')} for s in ('train','validation')}
    manifest=list(old['manifest']);sequences=[]
    for p in sorted(Path('/workspace/outputs/fountain').glob('*/*L30_A_test_*/episode.hdf5')):
        with h5py.File(p,'r') as f:
            spec=json.loads(f.attrs['spec']);name=spec['name'];m=json.loads(f.attrs['metrics'])
            assert f.attrs['status']=='complete' and f.attrs['camera_rig']=='A_task_wrist_v4'
            assert m['policy']['checkpoint_sha256']==PARENT_SHA
            assert spec['spacing']==.0018 and spec['dt']==.0025
            d=f['data/demo_0'];rows=json.loads((p.parent/'samples.json').read_text())
            decisions=json.loads((p.parent/'policy-decisions.json').read_text())
            assert len(rows)==len(decisions)==len(d['actions'])
            # First second precedes bottle contact in parent runs; includes the aborted cases.
            ix=[i for i,t in enumerate(d['time'][:]) if t-d['time'][0]<=1.+1e-7]
            sequences.append(dict(name=name,images=np.stack([decode(d['obs/task_image'][i],'jpeg95') for i in ix]),
                times=d['time'][:][ix],source_xyz=d['ground_truth/bottle_pose'][0,:3],cup_xyz=np.array(spec['cup_world']),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),path=str(p)))
            if not m['grasp_verified'] or not m['zero_spill']:continue
            xy=next(x['cup_xy'] for x in decisions if x.get('cup_xy') is not None)
            final=rows[-1]
            changed=[x['t'] for x in rows if abs(x['cup_ml']-final['cup_ml'])>.5*.0018**3*1e6]
            stable_after=max(changed,default=0.)+.35
            split='validation' if name in VALIDATION else 'train';used=[]
            for i in range(0,len(rows),2):
                if rows[i]['t']<stable_after:continue
                assert abs(rows[i]['t']-float(d['next_time'][i]))<1e-5
                image=cup_crop(decode(d['next_obs/task_image'][i],'jpeg95'),xy,saved['calibration'])
                for k,v in dict(image=image,target=final['cup_fill_fraction'],cup_xy=xy,episode=name).items():arrays[split][k].append(v)
                used.append(i)
            manifest.append(dict(name=name,path=str(p),sha256=sequences[-1]['sha256'],split=split,frames=len(used),indices=used,
                stable_inventory_after_s=stable_after,target_final_height=final['cup_fill_fraction'],cup_crop_xy=xy,
                label='final settled particle surface; frames after final inventory stabilizes',consumed_previous_episode=True))
    assert len(sequences)==30 and len(manifest)==76
    data={}
    for split,a in arrays.items():
        data[split]={k:old['data'][split][k]+v if k=='episode' else torch.cat((old['data'][split][k],torch.from_numpy(np.asarray(v,dtype=np.uint8 if k=='image' else np.float32)))) for k,v in a.items()}
    result=dict(data=data,initial=old['initial'],manifest=manifest,localization_sequences=sequences)
    torch.save(result,cache)
    (ROOT/'data-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return result


@torch.inference_mode()
def calibrate_localization(saved,sequences):
    """Choose consensus support on consumed episodes, with synthetic absence checks."""
    from pour_temporal_localization import consensus
    from pour_guard_vision import bottle_base_from_neck, project
    observations=[];negatives=[]
    for e in sequences:
        row=dict(name=e['name'],sha256=e['sha256'],split='validation' if e['name'] in VALIDATION else 'train',detections=[])
        for i,image in enumerate(e['images']):
            item=dict(time=float(e['times'][i]-e['times'][0]))
            for kind in ('source','cup'):
                uv,score=locate(image,saved[kind+'_bank']);item[kind]=dict(uv=uv.tolist(),score=score)
            row['detections'].append(item)
        row['source_xyz']=e['source_xyz'].tolist();row['cup_xyz']=e['cup_xyz'].tolist();observations.append(row)
        # Explicitly synthetic: remove neck edges, retaining body/background distractions.
        image=e['images'][0].copy();uv=project(e['source_xyz']+[0,0,.135],saved['calibration']);x,y=np.rint(uv).astype(int)
        image[max(0,y-14):min(384,y+38),max(0,x-24):min(384,x+24)]=np.median(image[10:40,10:40],axis=(0,1)).astype(np.uint8)
        n,s=locate(image,saved['source_bank'])
        negatives.append(dict(name=e['name'],type='synthetic neck occlusion repeated across window',uv=n.tolist(),score=s))
    candidates=[]
    for radius in (1.,1.5,2.):
      for votes in (6,8):
       for floor in (.24,.28,.30):
        config=dict(window=20,votes=votes,radius_px=radius,fraction=.60,score_floor=floor)
        errors=[];latencies=[];misses=[];false_accepts=0
        for e in observations:
            accepted=None
            for i in range(len(e['detections'])):
                history=[(x['source']['uv'],x['source']['score']) for x in e['detections'][:i+1]]
                accepted=consensus(history,saved['thresholds']['source_localization_score'],config)
                if accepted:
                    error=float(np.linalg.norm(bottle_base_from_neck(accepted['uv'],saved['calibration'])[:2]-np.array(e['source_xyz'][:2]))*1000)
                    errors.append(error);latencies.append(e['detections'][i]['time']);false_accepts+=error>2.;break
            if not accepted:misses.append(e['name'])
        synthetic_accepts=sum(consensus([(e['uv'],e['score'])]*20,saved['thresholds']['source_localization_score'],config) is not None for e in negatives)
        candidates.append(dict(settings=config,misses=misses,false_accepts=false_accepts,synthetic_accepts=synthetic_accepts,
            max_error_mm=max(errors,default=999.),mean_latency_s=float(np.mean(latencies)) if latencies else 999.))
    # These are calibration episodes, not the new reserved task evaluation.
    eligible=[x for x in candidates if not x['misses'] and not x['false_accepts'] and not x['synthetic_accepts']]
    assert eligible,'No consensus calibration meets accuracy and synthetic rejection gates'
    best=min(eligible,key=lambda x:(x['max_error_mm'],x['mean_latency_s']))
    source=best['settings'];cup=dict(source,score_floor=saved['thresholds']['cup_localization_score'])
    for e in observations:
        accepted=None
        for i in range(len(e['detections'])):
            history=[(x['cup']['uv'],x['cup']['score']) for x in e['detections'][:i+1]]
            accepted=consensus(history,saved['thresholds']['cup_localization_score'],cup)
            if accepted:break
        assert accepted,e['name']
        error=np.linalg.norm(plane_point(accepted['uv'],1.078,saved['calibration'])[:2]-np.array(e['cup_xyz'][:2]))*1000
        assert error<2.,(e['name'],error)
    settings=dict(source=source,cup=cup,timeout_s=2.)
    report=dict(settings=settings,candidates=candidates,chosen=best,observations=observations,synthetic_negatives=negatives,
        note='All episodes are consumed parent evidence. Candidate selection uses these calibration windows; no independent localization-generalization claim. Synthetic occlusions are not real absent-object tests. Original per-frame anchor thresholds are unchanged; consensus rejects isolated spatial outliers.')
    (ROOT/'localization-calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    return settings,dict(chosen=best,positive_windows=len(observations),synthetic_negative_windows=len(negatives))
