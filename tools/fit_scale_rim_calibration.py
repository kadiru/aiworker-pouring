"""Six bounded RGB-coordinate correction coefficients, fitted only on train."""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/src','/workspace']
from pour_cup_locator import CupLocator
from pour_guard_vision import project,plane_point
from scale_rim_refiner import RimRefiner,locate_static,correct_rim
from fit_scale_rim_refiner import collect,evaluate,windows
from scale_data_admission import collection_paths
ROOT=Path('/workspace/learning/A-scale-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,indent=2)+'\n')

def corrected(raw,a,cal,bank):
    rows=[]
    for r,m in zip(raw['rows'],a['meta']):
        uv,score=correct_rim(r['uv'],r['score'],bank)
        error=float(np.linalg.norm(plane_point(uv,m['z'],cal)[:2]-m['xy'])*1000)
        rows.append(dict(r,uv=uv.tolist(),score=score,error_mm=error,raw_error_mm=r['error_mm'],
                         affine_correction_px=float(np.linalg.norm(uv-r['uv']))))
    return dict(max_error_mm=max(r['error_mm'] for r in rows),mean_error_mm=float(np.mean([r['error_mm'] for r in rows])),
        minimum_positive_score=min(r['score'] for r in rows),maximum_synthetic_absent_score=raw['maximum_synthetic_absent_score'],rows=rows)

@torch.inference_mode()
def main():
    assert not (ROOT/'cup-neural-localization.json').exists() and not (ROOT/'component-selection.json').exists()
    protocol=read(ROOT/'locator-refit-r6-plan.json');m=read(ROOT/'candidate.json');ap,rp=collection_paths(ROOT,m)
    assert protocol['plan_sha256']==sha(ROOT/'plan.json') and protocol['audit_sha256']==sha(ap) and protocol['review_sha256']==sha(rp)
    source=ROOT/protocol['parent'];assert sha(source)==protocol['parent_sha256']
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4);start=time.perf_counter()
    parent=torch.load(source,map_location='cpu',weights_only=False);bank=copy.deepcopy(parent['bank'])
    assert bank['kind']=='native_rim_refiner' and 'affine_uv_correction' not in bank
    coarse=CupLocator().cuda().eval();coarse.load_state_dict(bank['coarse_state'])
    fine=RimRefiner().cuda().eval();fine.load_state_dict(bank['state'])
    plan,data,cal=collect(coarse,'r6');raw={s:evaluate(fine,a,cal,bank['max_correction_pixels']) for s,a in data.items()}
    features=[];targets=[];names=list(dict.fromkeys(r['name'] for r in raw['train']['rows']))
    for name in names:
        uv=np.mean([r['uv'] for r in raw['train']['rows'] if r['name']==name],axis=0)
        meta=next(m for m in data['train']['meta'] if m['name']==name)
        target=project(np.r_[meta['xy'],meta['z']],cal)
        features.append(np.r_[1.,(uv-512.)/512.]);targets.append(target-uv)
    x=torch.tensor(np.asarray(features),device='cuda',dtype=torch.float64);y=torch.tensor(np.asarray(targets),device='cuda',dtype=torch.float64)
    penalty=torch.diag(x.new_tensor([protocol['ridge_intercept_penalty'],protocol['ridge_slope_penalty'],protocol['ridge_slope_penalty']]))
    fitstart=time.perf_counter();coefficient=torch.linalg.solve(x.T@x+penalty,x.T@y);torch.cuda.synchronize();fit_seconds=time.perf_counter()-fitstart
    bank.update(affine_uv_correction=coefficient.cpu().tolist(),max_affine_correction_pixels=protocol['max_affine_correction_pixels'])
    bank['identity']=hashlib.sha256((bank['identity']+json.dumps(bank['affine_uv_correction'])).encode()).hexdigest()
    results={s:corrected(raw[s],a,cal,bank) for s,a in data.items()};gate=plan['component_gates']
    minimum=min(r['minimum_positive_score'] for r in results.values());maximum=max(r['maximum_synthetic_absent_score'] for r in results.values());threshold=(minimum+maximum)/2
    config=dict(parent['consensus'],score_floor=max(.01,threshold*.6));win=[]
    for s,a in data.items():win+=windows(results[s],a['meta'],cal,threshold,config)
    parity=[];end_to_end=[]
    for index,image in data['validation']['parity']:
        cpu,cs=locate_static(image,bank,device='cpu');gpu,gs=locate_static(image,bank,device='cuda')
        parity.append(float(np.linalg.norm(cpu-gpu)));end_to_end.append(float(np.linalg.norm(gpu-results['validation']['rows'][index]['uv'])))
    passed=all(r['max_error_mm']<gate['localization_max_error_mm'] for r in results.values()) and minimum-maximum>gate['localization_negative_margin'] and all(w['confirmed'] and w['error_mm']<gate['localization_max_error_mm'] for w in win) and max(parity)<.1 and max(end_to_end)<.01
    artifact=dict(bank=bank,calibration=cal,threshold=threshold,consensus=config,accepted=passed)
    torch.save(artifact,ROOT/'cup-neural-localization-attempt.pt')
    report=dict(passed=passed,scope='all_nine_declared_cup_sizes',learned_parameters=6,frozen_parameters=sum(p.numel() for p in coarse.parameters())+sum(p.numel() for p in fine.parameters()),
        fitting_episodes=names,optimizer_steps=0,closed_form_solves=1,fit_seconds=fit_seconds,total_seconds=time.perf_counter()-start,
        device=torch.cuda.get_device_name(),plan_sha256=sha(ROOT/'plan.json'),refit_protocol_sha256=sha(ROOT/'locator-refit-r6-plan.json'),
        parent_sha256=sha(source),audit_sha256=sha(ap),manual_review_sha256=sha(rp),module_sha256=sha(ROOT/'scale_rim_refiner.py'),source_sha256=sha(Path(__file__)),
        data_helper_sha256=sha(ROOT/'fit_scale_rim_refiner.py'),input_resolution=1024,native_crop_size=256,
        counts={s:dict(images=len(a['meta']),episodes=len({r['name'] for r in a['meta']})) for s,a in data.items()},
        training_check=results['train'],selected=results['validation'] if passed else None,best_attempt=results['validation'],temporal_windows=win,
        absence_margin=minimum-maximum,cpu_gpu_max_difference_pixels=max(parity),end_to_end_max_difference_pixels=max(end_to_end),
        maximum_fitted_correction_pixels=max(r['affine_correction_px'] for result in results.values() for r in result['rows']),
        attempt_checkpoint_sha256=sha(ROOT/'cup-neural-localization-attempt.pt'),
        method='Six affine pixel correction coefficients fitted by CUDA float64 ridge least squares on one mean per training episode, with fixed 0.1 slope penalty and a 2-native-pixel correction bound. Both CNNs remain frozen at the retained r4 candidate. Image-derived UV only at inference, whole-episode splits, unchanged frame/temporal/absence gates, native CPU/GPU checks. No RL.')
    if passed:
        torch.save(artifact,ROOT/'cup-neural-localization.pt');report['checkpoint_sha256']=sha(ROOT/'cup-neural-localization.pt')
        write('scope-selection.json',dict(passed=True,scope=report['scope'],checkpoint_sha256=report['checkpoint_sha256'],
            max_validation_error_mm=results['validation']['max_error_mm'],max_training_error_mm=results['train']['max_error_mm'],threshold=threshold,
            module_sha256=report['module_sha256'],refit_protocol_sha256=report['refit_protocol_sha256'],model_activated=False))
    write('cup-neural-localization.json',report)
    print(json.dumps(dict(passed=passed,validation_max_mm=results['validation']['max_error_mm'],training_max_mm=results['train']['max_error_mm'],max_correction_px=report['maximum_fitted_correction_pixels'],cpu_gpu_max_px=max(parity))),flush=True)
if __name__=='__main__':main()
