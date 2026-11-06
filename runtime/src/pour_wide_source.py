"""Offline empty/presence coverage under wider optics, with explicit labels."""
import json
from pathlib import Path
import torch
from pour_stop_models import LiquidDetector
from pour_source_alignment import aligned_crops
from validate_pour_recording import h5py,decode
import numpy as np
from collections import Counter


def fit_readout(saved,seq,localized):
    """Fit only the 65-parameter empty logit on frozen visual embeddings.

    Cup, source and wrist encoders, hidden features, presence and inventory
    outputs remain byte-identical. Disjoint validation chooses regularization.
    """
    model=LiquidDetector().cuda().eval();model.load_state_dict(saved['source_detector'])
    arrays={s:dict(x=[],y=[],name=[],index=[],eligible=[]) for s in ('train','validation')}
    manifest=[]
    with torch.inference_mode():
        for e in seq:
            path=Path(e['path']);rows=json.loads((path.parent/'samples.json').read_text())
            used=[]
            with h5py.File(path,'r') as f:
                spec=json.loads(f.attrs['spec']);d=f['data/demo_0'];allowance=max(.25,.01*spec['initial_ml'])
                ids=[i for i,r in enumerate(rows) if i%2==0 and r['bottle_tilt_deg']>=75 and r['grasp_verified']
                     and (r['bottle_ml']<=allowance or r['bottle_ml']>=.5)]
                for start in range(0,len(ids),64):
                    indices=ids[start:start+64];crops=[]
                    for i in indices:
                        crops.append(aligned_crops(decode(d['next_obs/task_image'][i],'jpeg95'),
                            decode(d['next_obs/wrist_image'][i],'jpeg95'),localized[e['name']],saved['calibration']))
                    if not crops:continue
                    _,source,wrist=[torch.as_tensor(np.stack(a),device='cuda') for a in zip(*crops)]
                    embedding=model.source_head[:2](torch.cat([model.source(source),model.wrist(wrist)],1))
                    a=arrays[e['split']];a['x'].append(torch.cat([embedding,torch.ones((len(indices),1),device='cuda')],1).cpu())
                    a['y'].extend(float(rows[i]['bottle_ml']<=allowance) for i in indices)
                    a['eligible'].extend(rows[i]['bottle_tilt_deg']>=95 and rows[i]['source_outflow_ml_s']<3. for i in indices)
                    a['name'].extend([e['name']]*len(indices));a['index'].extend(indices);used.extend(indices)
            manifest.append(dict(name=e['name'],split=e['split'],path=e['path'],sha256=e['sha256'],indices=used,
                                 empty_allowance_ml=allowance,labels='Offline particle inventory; encoder inputs are RGB only'))
    with torch.inference_mode(False),torch.enable_grad():
        data={s:dict(x=torch.cat(a['x']).to(device='cuda',dtype=torch.float64).clone(),
                    y=torch.tensor(a['y'],device='cuda',dtype=torch.float64),eligible=torch.tensor(a['eligible'],device='cuda'),
                    name=a['name'],index=a['index']) for s,a in arrays.items()}
        tr,va=data['train'],data['validation'];assert set(tr['name']).isdisjoint(va['name'])
        assert all(len(set(a['y']))==2 for a in arrays.values())
        counts=Counter(zip(tr['name'],arrays['train']['y']))
        weight=tr['y'].new_tensor([1./counts[n,y] for n,y in zip(tr['name'],arrays['train']['y'])]);weight/=weight.mean()
        prior=torch.cat([model.source_head[2].weight[1],model.source_head[2].bias[1,None]]).detach().to(dtype=torch.float64).clone()
        normal=saved['thresholds']['empty_probability'];slow=saved['thresholds']['slow_empty_probability']
        candidates=[]
        for ridge in (1e-5,1e-4,1e-3,1e-2):
            coef=torch.nn.Parameter(prior.clone());optimizer=torch.optim.LBFGS([coef],max_iter=200,line_search_fn='strong_wolfe',tolerance_grad=1e-9)
            def closure():
                optimizer.zero_grad()
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(tr['x']@coef,tr['y'],reduction='none')*weight).mean()
                loss=loss+ridge*(coef[:-1]-prior[:-1]).square().sum();loss.backward();return loss
            optimizer.step(closure)
            with torch.no_grad():
                probability=(va['x']@coef).sigmoid();positive=va['y']>.5;negative=~positive
                stats=dict(validation_bce=float(torch.nn.functional.binary_cross_entropy(probability,va['y'])),
                    positive_frames=int(positive.sum()),negative_frames=int(negative.sum()),
                    false_positive_normal=int(((probability>=normal)&negative&va['eligible']).sum()),
                    false_positive_slow=int(((probability>=slow)&negative&va['eligible']).sum()),
                    false_positive_slow_all_tilts=int(((probability>=slow)&negative).sum()),
                    positive_detected=int(((probability>=normal)&positive&va['eligible']).sum()))
                episodes=[]
                for name in sorted(set(va['name'])):
                    ix=[i for i,n in enumerate(va['name']) if n==name and arrays['validation']['y'][i]>.5 and arrays['validation']['eligible'][i]]
                    if ix:episodes.append(dict(name=name,positive_frames=len(ix),
                        detected=int((probability[ix]>=normal).sum()),min_probability=float(probability[ix].min()),max_probability=float(probability[ix].max())))
                candidates.append(dict(ridge=ridge,metrics=stats,positive_episodes=episodes,coef=coef.detach().cpu().tolist()))
        valid=[c for c in candidates if c['metrics']['false_positive_slow']==0 and c['positive_episodes']
               and all(e['detected']>=6 for e in c['positive_episodes'])]
        from pour_wide_study import ROOT
        report=dict(method='Episode/class-balanced binary cross entropy with L2 penalty around the parent empty readout; 64 frozen hidden features plus bias.',
            validation_operating_gate='Labeled tilt>=95 degrees and source outflow<3 nominal mL/s, an offline proxy for the existing runtime proprioception/visual-flow gate. All-angle errors also reported. No test data.',
            learned_parameters=65,train_episodes=len(set(tr['name'])),validation_episodes=len(set(va['name'])),
            train_frames=len(tr['y']),validation_frames=len(va['y']),
            candidates=[{k:v for k,v in c.items() if k!='coef'} for c in candidates],data_manifest=manifest)
        (ROOT/'empty-readout-candidates.json').write_text(json.dumps(report,indent=2)+'\n')
        assert valid,'Empty readout has false-positive or missing-positive validation episodes; improve training before native use'
        chosen=min(valid,key=lambda c:c['metrics']['validation_bce'])
        state={k:v.detach().cpu().clone() for k,v in saved['source_detector'].items()}
        state['source_head.2.weight'][1]=torch.tensor(chosen['coef'][:-1],dtype=state['source_head.2.weight'].dtype)
        state['source_head.2.bias'][1]=chosen['coef'][-1]
        for k,v in state.items():
            old=saved['source_detector'][k].detach().cpu()
            assert torch.equal(v[[0,2]],old[[0,2]]) if k.startswith('source_head.2.') else torch.equal(v,old)
        saved['source_detector']=state
        return dict(**{k:v for k,v in report.items() if k not in ('candidates','data_manifest')},
                    selected={k:v for k,v in chosen.items() if k!='coef'})


@torch.inference_mode()
def probe(saved,seq,localized,device='cuda'):
    model=LiquidDetector().to(device).eval();model.load_state_dict(saved['source_detector'])
    episodes=[]
    for e in seq:
        path=Path(e['path']);rows=json.loads((path.parent/'samples.json').read_text())
        samples=[]
        with h5py.File(path,'r') as f:
            spec=json.loads(f.attrs['spec']);d=f['data/demo_0']
            indices=[i for i,r in enumerate(rows) if i%2==0 and r['bottle_tilt_deg']>=95 and r['source_outflow_ml_s']<3.]
            for start in range(0,len(indices),32):
                ids=indices[start:start+32];crops=[]
                for i in ids:
                    crops.append(aligned_crops(decode(d['next_obs/task_image'][i],'jpeg95'),
                        decode(d['next_obs/wrist_image'][i],'jpeg95'),localized[e['name']],saved['calibration']))
                if not crops:continue
                p=model(*[torch.as_tensor(np.stack(a),device=device) for a in zip(*crops)])
                for j,i in enumerate(ids):
                    r=rows[i]
                    samples.append(dict(index=i,t=r['t'],source_ml=r['bottle_ml'],phase=r['phase'],
                        label_empty=r['bottle_ml']<=max(.25,.01*spec['initial_ml']),
                        empty_probability=float(p['empty_logit'][j].sigmoid()),
                        present_probability=float(p['present_logit'][j].sigmoid())))
        episodes.append(dict(name=e['name'],split=e['split'],expected_stop=spec['expected_stop'],samples=samples))
    valid=[s for e in episodes if e['split']=='validation' for s in e['samples']]
    threshold=saved['thresholds']['empty_probability']
    def detected(s):return s['empty_probability']>=threshold and s['present_probability']>=.7
    positive=[s for s in valid if s['label_empty']];negative=[s for s in valid if s['source_ml']>=.5]
    missed=[e['name'] for e in episodes if e['split']=='validation' and e['expected_stop']=='bottle_empty'
            and not any(s['label_empty'] and detected(s) for s in e['samples'])]
    return dict(method=('Frozen visual encoders with refitted empty readout' if saved.get('wide_empty_readout') else 'Frozen parent visual networks')+
        ' on affine-normalized wide-camera RGB and unchanged wrist RGB; this probe itself performs no fit.',
        source_frame_normalization='cup_aligned_v2',episodes=episodes,empty_probability_threshold=threshold,
        validation_positive_frames=len(positive),validation_negative_frames=len(negative),
        detected_positive_frames=sum(map(detected,positive)),false_positive_frames=sum(map(detected,negative)),
        validation_empty_episodes_without_detection=missed,
        caveat='Offline filtering uses labeled tilt/flow. Closed-loop dwell, visibility and task tests remain necessary; no test images consumed.')
