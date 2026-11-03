"""CPU-only validation selection of independently trained detector components.

Both candidates were fitted on the guarded 4070. No fresh test frames are used.
The motor readout is retained; its native validation checks the late fusion.
"""
import hashlib,json,shutil
from pathlib import Path
import torch
from pour_stop_study import ROOT
from pour_stop_data import perception_data
from pour_stop_models import LiquidDetector


def main():
    torch.set_num_threads(2)
    checkpoint=ROOT/'best.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    shutil.copy2(checkpoint,ROOT/'before-component-selection.pt')
    data=perception_data()['validation'];model=LiquidDetector().eval();candidates=[]
    for path in (ROOT/'detector-source-best.pt',ROOT/'surface-fit-before-label-mask/detector-source-best.pt'):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        weights=dict(payload['detector']);weights.update({k:v for k,v in saved['model'].items() if not k.startswith(('cup.','cup_head.'))})
        model.load_state_dict(weights);parts=[]
        with torch.no_grad():
            for start in range(0,len(data['fill']),128):
                parts.append(model(*[data[k][start:start+128] for k in ('cup','source','wrist')]))
        p={k:torch.cat([part[k] for part in parts]) for k in parts[0]}
        positive=(data['empty']>.5)&(data['present']>.5);negative=(data['empty']<.5)&(data['present']>.5)
        gate=data['empty_gate']>.5;probs=p['empty_logit'].sigmoid();present=p['present_logit'].sigmoid()>.7
        sweep=[]
        for threshold in (.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.97,.98,.99,.995,.999):
            called=(probs>=threshold)&present
            sweep.append(dict(threshold=threshold,true_positive=int((called&positive).sum()),false_positive=int((called&negative).sum()),
                positives=int(positive.sum()),gated_true_positive=int((called&positive&gate).sum()),
                gated_false_positive=int((called&negative&gate).sum()),gated_positives=int((positive&gate).sum())))
        eligible=[r for r in sweep if r['gated_false_positive']==0 and r['gated_true_positive']>0]
        candidates.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),step=saved['step'],
            sweep=sweep,operating_point=eligible[0] if eligible else None))
    eligible=[r for r in candidates if r['operating_point']]
    assert eligible,'No gated empty operating point; cannot declare a calibrated empty detector'
    chosen=max(eligible,key=lambda c:(c['operating_point']['gated_true_positive'],-c['operating_point']['false_positive']))
    selected=torch.load(chosen['path'],map_location='cpu',weights_only=False)
    payload['detector'].update({k:v for k,v in selected['model'].items() if not k.startswith(('cup.','cup_head.'))})
    payload['thresholds']['empty_probability']=chosen['operating_point']['threshold']
    payload['detector_validation']['empty_operating_point']=chosen['operating_point']
    payload['detector_validation']['empty_threshold_sweep']=chosen['sweep']
    payload['detector_validation']['selected_source_step']=chosen['step']
    payload['component_selection']=dict(candidates=candidates,selected_source=chosen['path'],
        method='Independent visual-head validation selection before any fresh native cases',
        caveat='Only one gated positive frame. Motor readout retained after source-head late fusion; native validation required.',
        selection_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    torch.save(payload,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(checkpoint)
    (ROOT/'component-selection.json').write_text(json.dumps(dict(**payload['component_selection'],
        selected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),cuda_initialized=torch.cuda.is_initialized()),indent=2)+'\n')
    (ROOT/'detector-validation.json').write_text(json.dumps(payload['detector_validation'],indent=2)+'\n')
    print(json.dumps(dict(selected_source=chosen['path'],operating_point=chosen['operating_point'],cuda_initialized=torch.cuda.is_initialized()),indent=2))


if __name__=='__main__':main()
