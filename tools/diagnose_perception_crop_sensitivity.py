"""CPU-only diagnosis of a failed, frozen perception evaluation; no fitting."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/src','/workspace']
from validate_pour_recording import h5py,decode
from pour_guard_models import CupEstimator
from pour_guard_vision import cup_crop
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.inference_mode()
def main():
    report=read(ROOT/'perception-holdout.json');plan=read(ROOT/'plan.json')
    assert not report['passed'] and sha(ROOT/'best.pt')==report['checkpoint_sha256']
    out=ROOT/f'crop-sensitivity-r{plan.get("perception_round",0)}.json';assert not out.exists()
    records={r['name']:r for r in read(ROOT/'recording-audit.json')['records']}
    specs={s['name']:s for s in plan['perception_holdout']}
    saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False);torch.set_num_threads(4)
    model=CupEstimator().eval();model.load_state_dict(saved['cup_model']);rows=[]
    for case in report['cases']:
        spec=specs[case['name']];record=records[case['name']];path=Path(record['path'])
        assert sha(path)==record['sha256'] and case['localized_xy'] is not None
        cal=copy.deepcopy(saved['cup_detail_calibration']);cal['cup_geometry']=spec['declared_cup_geometry']
        with h5py.File(path) as f:rgb=decode(f['data/demo_0/next_obs/cup_detail_image'][-1],'jpeg95')
        xy=np.array(case['localized_xy']);variants=[]
        for dx,dy in ((0.,0.),(-.001,0.),(.001,0.),(0.,-.001),(0.,.001)):
            crop=torch.from_numpy(cup_crop(rgb,xy+[dx,dy],cal))[None]
            pred=float((model(crop)[0]+saved['cup_bias']).clamp(0,1))
            variants.append(dict(dx_m=dx,dy_m=dy,prediction=pred,error=pred-case['measured_label']['target']))
        truth_crop=torch.from_numpy(cup_crop(rgb,spec['cup_world'][:2],cal))[None]
        truth_pred=float((model(truth_crop)[0]+saved['cup_bias']).clamp(0,1))
        rows.append(dict(name=case['name'],target=case['measured_label']['target'],variants=variants,
            truth_centered_diagnostic_prediction=truth_pred,truth_centered_error=truth_pred-case['measured_label']['target'],
            one_mm_center_perturbation_range=max(v['prediction'] for v in variants)-min(v['prediction'] for v in variants)))
    result=dict(checkpoint_sha256=report['checkpoint_sha256'],failed_report_sha256=sha(ROOT/'perception-holdout.json'),
        helper_sha256=sha(Path(__file__)),cases=rows,
        scope='Final native frame only; +/-1 mm axis-aligned crop-center perturbations. True fixture center is a diagnostic only, never used by live control or as a training crop. No fitting or acceptance-score replacement.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(name=r['name'],base_error=r['variants'][0]['error'],truth_centered_error=r['truth_centered_error'],perturbation_range=r['one_mm_center_perturbation_range']) for r in rows]))
if __name__=='__main__':main()
