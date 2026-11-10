"""No-fit development probe: repaired coarse crops with the retained fine model."""
import json
import numpy as np
import torch
from fit_dynamic_rim import ROOT,read,sha,collect,protected_unchanged,CupLocator
from fit_scale_rim_refiner import evaluate,windows
from scale_rim_refiner import RimRefiner,correct_rim,locate_static
from pour_guard_vision import plane_point

def main():
    assert not (ROOT/'rim-reuse-probe.json').exists()
    protected_unchanged();torch.set_num_threads(4)
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name()
    parent=torch.load(ROOT/'cup-neural-localization.pt',map_location='cpu',weights_only=False)
    state=torch.load(ROOT/'coarse-d2.pt',map_location='cpu',weights_only=False)['state']
    bank=dict(parent['bank'],coarse_state=state,identity='reuse-probe-'+sha(ROOT/'coarse-d2.pt'))
    coarse=CupLocator().cuda().eval();coarse.load_state_dict(state)
    data,cal=collect(coarse,manifest_name='rim-reuse-data.json')
    fine=RimRefiner().cuda().eval();fine.load_state_dict(bank['state'])
    results={};summaries={};gate=read(ROOT/'protocol.json')['gates']
    for split,a in data.items():
        raw=evaluate(fine,a,cal,bank['max_correction_pixels'])
        for r,m in zip(raw['rows'],a['meta']):
            uv,score=correct_rim(r['uv'],r['score'],bank)
            r.update(uv=uv.tolist(),score=score,error_mm=float(np.linalg.norm(plane_point(uv,m['z'],cal)[:2]-m['xy'])*1000))
        raw.update(max_error_mm=max(r['error_mm'] for r in raw['rows']),mean_error_mm=float(np.mean([r['error_mm'] for r in raw['rows']])),minimum_positive_score=min(r['score'] for r in raw['rows']))
        results[split]=raw
        summaries[split]={k:v for k,v in raw.items() if k!='rows'}
    minimum=min(r['minimum_positive_score'] for r in results.values());maximum=max(r['maximum_synthetic_absent_score'] for r in results.values());threshold=(minimum+maximum)/2
    config=dict(parent['consensus'],score_floor=max(.01,threshold*.6))
    win=sum([windows(results[s],data[s]['meta'],cal,threshold,config) for s in data],[])
    parity=[]
    for i,image in data['validation']['parity']:
        cpu,_=locate_static(image,bank,'cpu');gpu,_=locate_static(image,bank,'cuda');parity.append(float(np.linalg.norm(cpu-gpu)))
    passed=max(r['max_error_mm'] for r in results.values())<gate['localization_max_error_mm'] and minimum-maximum>gate['localization_negative_margin'] and all(w['confirmed'] and w['error_mm']<gate['localization_max_error_mm'] for w in win) and max(parity)<.1
    report=dict(passed=passed,results=results,windows=win,absence_margin=minimum-maximum,cpu_gpu_max_pixels=max(parity),frozen_files={n:sha(ROOT/n) for n in ('probe_dynamic_rim_reuse.py','fit_dynamic_rim.py','scale_rim_refiner.py','cup-neural-localization.pt','coarse-d2.pt')},optimizer_steps=0,autonomy_accepted=False,scope='Read-only development ablation; retained fine model and its original affine correction, repaired coarse detector. No excluded frames, reserved data, optimizer updates or active controller changes.')
    protected_unchanged();(ROOT/'rim-reuse-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=passed,summaries=summaries,absence_margin=minimum-maximum,confirmed=sum(w['confirmed'] for w in win),cpu_gpu_max_pixels=max(parity))),flush=True)
if __name__=='__main__':main()
