"""Offline empty/presence coverage under wider optics, with explicit labels."""
import json
from pathlib import Path
import torch
from pour_stop_models import LiquidDetector
from pour_source_alignment import aligned_crops,held_source_presence
from pour_stop_data import crops as raw_crops
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
                ids=indices[start:start+32];crops=[];empty_crops=[];raw=[]
                for i in ids:
                    wrist=decode(d['next_obs/wrist_image'][i],'jpeg95')
                    task=decode(d['next_obs/task_image'][i],'jpeg95')
                    crops.append(aligned_crops(task,wrist,localized[e['name']],saved['calibration']))
                    if saved.get('source_presence_fusion'):raw.append(raw_crops(task,wrist))
                    if empty_model is not None:empty_crops.append(wrist_crop(wrist,saved['wide_empty_roi']))
                if not crops:continue
                p=model(*[torch.as_tensor(np.stack(a),device=device) for a in zip(*crops)])
                rp=model(*[torch.as_tensor(np.stack(a),device=device) for a in zip(*raw)])['present_logit'].sigmoid() if raw else None
                if empty_model is not None:p['empty_logit']=empty_model(torch.as_tensor(np.stack(empty_crops),device=device))
                for j,i in enumerate(ids):
                    r=rows[i]
                    present=float(p['present_logit'][j].sigmoid())
                    if rp is not None:
                        from pour_guard_actor import held_gripper
                        held=held_gripper(d['next_obs/joint_pos'][i],saved['pour_primitive']['grasp_closure']['min_deficit_rad'])
                        present=held_source_presence(float(rp[j]),present,held)
                    samples.append(dict(index=i,t=r['t'],source_ml=r['bottle_ml'],phase=r['phase'],
                        label_empty=r['bottle_ml']<=max(.25,.01*spec['initial_ml']),
                        empty_probability=float(p['empty_logit'][j].sigmoid()),
                        present_probability=present))
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
        source_presence_fusion=saved.get('source_presence_fusion'),
        validation_positive_frames=len(positive),validation_negative_frames=len(negative),
        detected_positive_frames=sum(map(detected,positive)),false_positive_frames=sum(map(detected,negative)),
        validation_empty_episodes_without_detection=missed,
        caveat='Offline filtering uses labeled tilt/flow. Closed-loop dwell, visibility and task tests remain necessary; no test images consumed.')


@torch.inference_mode()
def presence_coverage(saved,seq,device='cuda'):
    """Check the loss-of-source gate on every consumed pre-stop held frame."""
    from pour_guard_actor import held_gripper
    from pour_guard_vision import locate,plane_point
    model=LiquidDetector().to(device).eval();model.load_state_dict(saved['source_detector']);episodes=[]
    assert saved['source_presence_fusion']=='held_raw_or_aligned_v1'
    for e in seq:
        rows=json.loads((Path(e['path']).parent/'samples.json').read_text());samples=[]
        uv,_=locate(e['images'][0],saved['cup_bank'],device=device);xy=plane_point(uv,1.078,saved['calibration'])[:2]
        with h5py.File(e['path']) as f:
            d=f['data/demo_0'];stopped=d['ground_truth/stop_command'][:].astype(bool)
            ids=[i for i,r in enumerate(rows) if r['grasp_verified'] and r['both_fingers'] and not stopped[i]]
            for start in range(0,len(ids),64):
                indices=ids[start:start+64];raw=[];aligned=[]
                for i in indices:
                    task=decode(d['obs/task_image'][i],'jpeg95');wrist=decode(d['obs/wrist_image'][i],'jpeg95')
                    raw.append(raw_crops(task,wrist));aligned.append(aligned_crops(task,wrist,xy,saved['calibration']))
                if not indices:continue
                rp=model(*[torch.as_tensor(np.stack(a),device=device) for a in zip(*raw)])['present_logit'].sigmoid()
                ap=model(*[torch.as_tensor(np.stack(a),device=device) for a in zip(*aligned)])['present_logit'].sigmoid()
                for j,i in enumerate(indices):
                    held=held_gripper(d['obs/joint_pos'][i],saved['pour_primitive']['grasp_closure']['min_deficit_rad'])
                    samples.append(dict(index=i,raw=float(rp[j]),aligned=float(ap[j]),encoder_hold=held,
                        fused=held_source_presence(float(rp[j]),float(ap[j]),held)))
        longest=run=0;previous=None
        for s in samples:
            run=run+1 if s['fused']<.05 and previous==s['index']-1 else int(s['fused']<.05)
            previous=s['index'];longest=max(longest,run)
        episodes.append(dict(name=e['name'],split=e['split'],samples=samples,max_consecutive_lost_frames=longest))
    return dict(method='Unchanged detector on raw and cup-aligned views, maximum finite presence, gated by measured gripper obstruction; eight consecutive low-presence frames latch source loss.',
        episodes=episodes,frames=sum(len(e['samples']) for e in episodes),
        false_lost_source_episodes=[e['name'] for e in episodes if e['max_consecutive_lost_frames']>=8],
        scope='Offline replay on consumed positive held-bottle frames before the original stop; no counterfactual future trajectory or real missing-bottle validation. No test data.')
