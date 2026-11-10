"""Read-only diagnosis of coarse crop coverage on reviewed development data."""
import json
import numpy as np
import torch
from fit_dynamic_rim import ROOT,read,sha,collect,protected_unchanged,CupLocator
def main():
    assert not (ROOT/'rim-d0-crop-diagnostic.json').exists();protected_unchanged()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4)
    artifact=torch.load(ROOT/'cup-neural-localization.pt',map_location='cpu',weights_only=False)
    model=CupLocator().cuda().eval();model.load_state_dict(artifact['bank']['coarse_state']);data,cal=collect(model)
    report={}
    for split,a in data.items():
        rows=[]
        for m,target,origin,center,score,absent in zip(a['meta'],a['target'],a['origin'],a['coarse'],a['score'],a['absent']):
            distance=float(np.linalg.norm(np.asarray(target)+origin-center))
            rows.append(dict(m,target_in_crop=np.asarray(target).tolist(),coarse_error_native_pixels=distance,score=score,absent_score=absent,outside_crop=bool((np.asarray(target)<0).any() or (np.asarray(target)>255).any()),outside_refinement_bound=distance>artifact['bank']['max_correction_pixels']))
        report[split]=dict(frames=len(rows),outside_crop=sum(r['outside_crop'] for r in rows),outside_refinement_bound=sum(r['outside_refinement_bound'] for r in rows),max_coarse_error_native_pixels=max(r['coarse_error_native_pixels'] for r in rows),minimum_positive=min(a['score']),maximum_synthetic_absent=max(a['absent']),rows=rows)
    report.update(locator_sha256=sha(ROOT/'cup-neural-localization.pt'),helper_sha256=sha(ROOT/'diagnose_dynamic_rim.py'),scope='Read-only supervised development diagnosis. No frame exclusion, fit, reserved data or model activation.')
    (ROOT/'rim-d0-crop-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');protected_unchanged()
    print(json.dumps({s:{k:v for k,v in report[s].items() if k!='rows'} for s in ('train','validation')}),flush=True)
if __name__=='__main__':main()
