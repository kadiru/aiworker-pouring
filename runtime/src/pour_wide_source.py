"""Offline empty/presence coverage under wider optics, with explicit labels."""
import json
from pathlib import Path
import torch
from pour_stop_models import LiquidDetector
from pour_source_alignment import aligned_crops
from validate_pour_recording import h5py,decode
import numpy as np


@torch.inference_mode()
def probe(saved,seq,localized,device='cuda'):
    model=LiquidDetector().to(device).eval();model.load_state_dict(saved['source_detector'])
    empty_model=None
    if 'wide_empty_model' in saved:
        from pour_wide_empty import load_empty,wrist_crop
        empty_model=load_empty(saved,device)
    episodes=[]
    for e in seq:
        path=Path(e['path']);rows=json.loads((path.parent/'samples.json').read_text())
        samples=[]
        with h5py.File(path,'r') as f:
            spec=json.loads(f.attrs['spec']);d=f['data/demo_0']
            indices=[i for i,r in enumerate(rows) if i%2==0 and r['bottle_tilt_deg']>=95 and r['source_outflow_ml_s']<3.]
            for start in range(0,len(indices),32):
                ids=indices[start:start+32];crops=[];empty_crops=[]
                for i in ids:
                    wrist=decode(d['next_obs/wrist_image'][i],'jpeg95')
                    crops.append(aligned_crops(decode(d['next_obs/task_image'][i],'jpeg95'),wrist,localized[e['name']],saved['calibration']))
                    if empty_model is not None:empty_crops.append(wrist_crop(wrist,saved['wide_empty_roi']))
                if not crops:continue
                p=model(*[torch.as_tensor(np.stack(a),device=device) for a in zip(*crops)])
                if empty_model is not None:p['empty_logit']=empty_model(torch.as_tensor(np.stack(empty_crops),device=device))
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
    return dict(method=('Fine-tuned wrist-crop empty CNN and unchanged presence detector' if empty_model is not None else 'Frozen visual encoders with refitted empty readout' if saved.get('wide_empty_readout') else 'Frozen parent visual networks')+
        ' using normalized task views and the saved wrist crop; this probe itself performs no fit.',
        source_frame_normalization='cup_aligned_v2',episodes=episodes,empty_probability_threshold=threshold,
        validation_positive_frames=len(positive),validation_negative_frames=len(negative),
        detected_positive_frames=sum(map(detected,positive)),false_positive_frames=sum(map(detected,negative)),
        validation_empty_episodes_without_detection=missed,
        caveat='Offline filtering uses labeled tilt/flow. Closed-loop dwell, visibility and task tests remain necessary; no test images consumed.')
