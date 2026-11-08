"""Fit six affine image-center corrections; retain the two CNNs unchanged."""
import copy,hashlib,json,sys,urllib.request
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/learning/A-perception-20261108-v1/staged','/workspace/src','/workspace']
from fit_perception_locator import collect,read,sha,ROOT
from select_perception_locator_ensemble import evaluate
from pour_cup_locator import CupLocator,correct_prediction
from pour_guard_vision import plane_point
from pour_temporal_localization import consensus

def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not h['recording']['active']
    assert not (ROOT/'cup-locator-correction.json').exists() and not (ROOT/'scope-selection.json').exists()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4)
    source=ROOT/'rounds/baseline-r0/cup-neural-localization.pt';parent=torch.load(source,map_location='cpu',weights_only=False);bank=copy.deepcopy(parent['bank'])
    assert bank['kind']=='cup_heatmap_ensemble' and bank['max_disagreement_px']==6. and 'affine_uv_correction' not in bank
    models=[]
    for member in bank['members']:
        model=CupLocator().cuda().eval();model.load_state_dict(member['state']);models.append(model)
    plan,data,cal=collect(dense=True);assert plan['perception_round']==1
    raw={s:evaluate(models,d) for s,d in data.items()};train=raw['train'];features=[];targets=[]
    names=sorted({r['name'] for r in train if r['domain']=='native_detail' and r['baseline_cup']})
    for name in names:
        ids=[i for i,r in enumerate(train) if r['name']==name];uv=np.mean([train[i]['uv'] for i in ids],axis=0)
        target=data['train']['u'][ids].double().mean(0)
        features.append(np.r_[1.,(uv-192.)/192.]);targets.append(target-torch.tensor(uv,device='cuda',dtype=torch.float64))
    x=torch.tensor(np.asarray(features),device='cuda',dtype=torch.float64);y=torch.stack(targets)
    penalty=torch.diag(torch.tensor([1e-8,.1,.1],device='cuda',dtype=torch.float64))
    coefficient=torch.linalg.solve(x.T@x+penalty,x.T@y)
    bank.update(affine_uv_correction=coefficient.cpu().tolist(),max_correction_px=3.)
    rows={}
    for split,values in raw.items():
        rows[split]=[]
        for r,meta in zip(values,data[split]['meta']):
            uv,score=correct_prediction((r['uv'],r['score']),bank)
            error=float(np.linalg.norm(plane_point(uv,meta['z'],meta['camera'])[:2]-meta['xy'])*1000)
            rows[split].append(dict(r,raw_error_mm=r['error_mm'],uv=uv.tolist(),score=score,error_mm=error,correction_px=float(np.linalg.norm(uv-r['uv']))))
    native=[r for values in rows.values() for r in values if r['domain']=='native_detail'];primary=[r for r in native if r['baseline_cup']]
    minimum=min(r['score'] for r in primary);maximum=max(r['synthetic_absent_score'] for r in native);threshold=(minimum+maximum)/2
    config=dict(parent['consensus'],score_floor=max(.01,threshold*.6));windows=[]
    for split,values in rows.items():
        for name in sorted({r['name'] for r in values if r['domain']=='native_detail' and r['baseline_cup']}):
            sequence=sorted([r for r in values if r['name']==name and r['group']=='obs'],key=lambda r:r['index']);history=[];accepted=None
            for r in sequence:
                history.append((r['uv'],r['score']));accepted=consensus(history,threshold,config)
                if accepted is not None:break
            meta=next(m for m in data[split]['meta'] if m['name']==name)
            error=None if accepted is None else float(np.linalg.norm(plane_point(accepted['uv'],meta['z'],meta['camera'])[:2]-meta['xy'])*1000)
            windows.append(dict(name=name,split=split,confirmed=accepted is not None,error_mm=error,frames=len(history)))
    gate=plan['component_gates'];maximum_error=max(r['error_mm'] for r in primary)
    passed=maximum_error<gate['localization_max_error_mm'] and minimum-maximum>gate['localization_negative_margin'] and all(w['confirmed'] and w['error_mm']<gate['localization_max_error_mm'] for w in windows)
    report=dict(passed=passed,scope='baseline_cup_only',source_checkpoint_sha256=sha(source),fitted_parameters=6,training_episodes=names,
        correction=bank['affine_uv_correction'],max_correction_px=3.,frame_max_error_mm=maximum_error,
        frame_max_by_split={s:max(r['error_mm'] for r in values if r['domain']=='native_detail' and r['baseline_cup']) for s,values in rows.items()},
        minimum_positive_score=minimum,maximum_synthetic_absent_score=maximum,threshold=threshold,windows=windows,rows=rows,
        source_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'staged/pour_cup_locator.py'),
        method='Six affine pixel-offset coefficients fitted by ridge least squares on one mean observation per admitted original-cup training episode. Two RGB CNNs remain unchanged. Includes nine explicitly consumed failed r0 calibration records; new r1 checks are unused. Runtime inputs are image-derived centers only; corrections above 3 pixels are rejected. Same 3 mm frame/temporal gates.')
    if passed:
        torch.save(dict(bank=bank,calibration=cal,threshold=threshold,consensus=config),ROOT/'cup-neural-localization.pt')
        report['checkpoint_sha256']=sha(ROOT/'cup-neural-localization.pt')
        (ROOT/'scope-selection.json').write_text(json.dumps(dict(passed=True,scope='baseline_cup_only',checkpoint_sha256=report['checkpoint_sha256'],
            max_validation_error_mm=report['frame_max_by_split']['validation'],max_training_error_mm=report['frame_max_by_split']['train'],
            max_temporal_error_mm=max(w['error_mm'] for w in windows),threshold=threshold,source_sha256=sha(Path(__file__)),method=report['method']),indent=2)+'\n')
    (ROOT/'cup-locator-correction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','windows','training_episodes','method')}))
if __name__=='__main__':main()
