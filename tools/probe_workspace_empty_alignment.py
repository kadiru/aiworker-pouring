"""CPU counterfactual replay of source-image alignment on consumed recordings.

Use only closed C30 and W30 development records, never the reserved W30 tests.
No native trial is relabeled. Replay ends at the original stop, where changed
actions would diverge; missing later evidence is explicitly right-censored.
"""
import json
from pathlib import Path
import numpy as np
import torch
from pour_workspace_study import ROOT
from pour_stop_models import LiquidDetector
from pour_source_alignment import aligned_crops,combine_evidence
from pour_guard_actor import held_gripper,slow_empty_evidence
from validate_pour_recording import decode,h5py

torch.set_num_threads(2)
saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
model=LiquidDetector().eval();model.load_state_dict(saved['source_detector'])
threshold=saved['thresholds'];cap=saved['pour_primitive']['phase_cap_deg']
gap=saved['pour_primitive']['grasp_closure']['min_deficit_rad']
base=Path('/workspace/outputs/fountain');results=[]
files=sorted(base.glob('*/*C30_A_test_*/episode.hdf5'))+sorted(base.glob('*/*W30_A_dev_*/episode.hdf5'))
for path in files:
    spec=json.loads((path.parent/'parameters.json').read_text());name=spec['name']
    assert name.startswith(('C30_A_test_','W30_A_dev_')) and not name.startswith('W30_A_test_')
    decisions=json.loads((path.parent/'policy-decisions.json').read_text())
    metrics=json.loads((path.parent/'metrics.json').read_text());samples=json.loads((path.parent/'samples.json').read_text())
    end=next((i for i,r in enumerate(decisions) if r['stop_latched'] is not None),len(decisions)-1)
    take=[i for i,r in enumerate(decisions[:end+1]) if r['cup_xy'] is not None and
        r['estimated_tilt_from_joints']>=95 and r['visual_flow_fraction_s']*72.8<3.]
    aligned={}
    with h5py.File(path,'r') as f:
        assert f.attrs['status']=='complete';d=f['data/demo_0'];q=d['obs/joint_pos'][:]
        for start in range(0,len(take),32):
            indices=take[start:start+32];images=[aligned_crops(decode(d['obs/task_image'][i],'jpeg95'),
                decode(d['obs/wrist_image'][i],'jpeg95'),decisions[i]['cup_xy'],saved['calibration']) for i in indices]
            with torch.inference_mode():p=model(*[torch.as_tensor(np.stack([x[k] for x in images])) for k in range(3)])
            for j,i in enumerate(indices):aligned[i]=(float(p['empty_logit'][j].sigmoid()),float(p['present_logit'][j].sigmoid()))
    variants={}
    for variant in ('recorded_raw','aligned','either_view'):
        votes=slow_votes=0;trigger=None;frames=[]
        for i,r in enumerate(decisions[:end+1]):
            empty,present=r['empty_probability'],r['source_present_probability']
            if variant=='aligned':empty,present=aligned.get(i,(0.,0.))
            elif variant=='either_view':
                ae,ap=aligned.get(i,(0.,0.));empty,present=combine_evidence(empty,present,ae,ap)
            flow=r['visual_flow_fraction_s']*72.8;tilt=r['estimated_tilt_from_joints']
            normal=empty>=threshold['empty_probability'] and present>=.7 and tilt>=95 and flow<3.
            slow=slow_empty_evidence(threshold,r['own_pour_phase_deg'],cap,tilt,r['cap_dwell_seconds'],held_gripper(q[i],gap),present,empty,flow)
            votes=votes+1 if normal else 0;slow_votes=slow_votes+1 if slow else 0
            source=spec['initial_ml'] if i==0 else samples[i-1]['bottle_ml']
            if i in aligned:frames.append(dict(index=i,t=r['t'],source_ml=source,empty=empty,present=present,normal_votes=votes,slow_votes=slow_votes))
            if votes>=12 or (slow and slow_votes>=threshold['slow_empty_votes']):
                trigger=dict(index=i,t=r['t'],source_ml=source,correct_empty=source<=max(.25,.01*spec['initial_ml']),empty=empty,present=present);break
        variants[variant]=dict(first_empty_stop=trigger,frames=frames,
            right_censored=trigger is None and decisions[end]['stop_latched'] is not None)
    row=dict(name=name,expected_stop=spec['expected_stop'],native_reason=metrics['bottle_result']['stop_reason'],native_success=metrics['surface_task_success_zero_spill'],
        observed_until_t=decisions[end]['t'],native_checkpoint_sha256=metrics['policy']['checkpoint_sha256'],variants=variants)
    results.append(row)
    print(name,{k:v['first_empty_stop'] for k,v in variants.items()},flush=True)
    (ROOT/'empty-alignment-replay.json').write_text(json.dumps(dict(method='CPU counterfactual classifier/stop-gate replay; actions after any hypothetical stop are unobserved, so this is not a native success',
        checkpoint_sha256=__import__('hashlib').sha256((ROOT/'best.pt').read_bytes()).hexdigest(),reserved_tests_excluded=True,results=results),indent=2)+'\n')
