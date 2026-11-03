"""CPU-only provenance summary of the selected visual hierarchy."""
import hashlib,json
import torch
from pour_stop_study import ROOT
from pour_stop_models import LiquidDetector,PourIncrement
from pour_learning import Policy


def main():
    p=ROOT/'best.pt';b=torch.load(p,map_location='cpu',weights_only=False)
    keys=('model_version','method','training_device','inputs','step','detector_step','detector_validation','thresholds',
        'baseline_sha256','scene_plan_sha256','sources','perception_manifest_sha256','motion_manifest_sha256',
        'perception_correction_sha256','perception_label_quality_sha256','component_selection','full_lead_calibration','no_privileged_runtime_inputs')
    r={k:b[k] for k in keys if k in b}
    r['checkpoint_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    r['parameter_counts']={n:sum(p.numel() for p in c().parameters()) for n,c in [('baseline',Policy),('liquid_detector',LiquidDetector),('archived_unused_pour_increment',PourIncrement)]}
    r['pour_primitive']={k:v for k,v in b.get('pour_primitive',{}).items() if k!='weights'}
    r['supervised_pour_steps']={s:sum(e['frames'] for e in es) for s,es in json.loads((ROOT/'motion-manifest.json').read_text()).items()}
    for k,n in [('label_quality','perception-label-quality.json'),('corrective_perception','corrective-perception.json')]:r[k]=json.loads((ROOT/n).read_text())
    r['method_note']='Supervised visual estimation plus imitation through contextual movement primitives. A programmed irreversible state machine joins the stages. Not end-to-end RL. The prior increment head is retained but inactive when pour_primitive is present.'
    r['cuda_initialized_by_audit']=torch.cuda.is_initialized()
    (ROOT/'model-summary.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(checkpoint_sha256=r['checkpoint_sha256'],supervised_pour_steps=r['supervised_pour_steps'],parameters=r['parameter_counts'],cuda_initialized=torch.cuda.is_initialized()),indent=2))


if __name__=='__main__':main()
