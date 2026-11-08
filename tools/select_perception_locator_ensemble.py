"""Evaluate one fixed equal-weight ensemble; no reserved data or weight search."""
import hashlib,json,sys,urllib.request
from pathlib import Path
import numpy as np
import torch
sys.path[:0]=['/workspace/learning/A-perception-20261108-v1/staged','/workspace/src','/workspace']
from fit_perception_locator import collect,read,sha,ROOT
from pour_cup_locator import CupLocator,centers,combine_predictions
from pour_guard_vision import plane_point
from pour_temporal_localization import consensus

@torch.inference_mode()
def evaluate(models,data):
    predictions=[];negative_predictions=[]
    for start in range(0,len(data['x']),16):
        x=data['x'][start:start+16];negative=x.clone()
        for j,(u,v) in enumerate(data['u'][start:start+16].round().long().tolist()):negative[j,max(0,v-24):min(384,v+48),max(0,u-50):min(384,u+50)]=180
        for images,target in ((x,predictions),(negative,negative_predictions)):
            values=[centers(m(images)) for m in models]
            arrays=[(u.cpu().numpy(),s.cpu().numpy()) for u,s in values]
            target.extend(combine_predictions([(u[j],float(s[j])) for u,s in arrays],6.) for j in range(len(x)))
    rows=[]
    for meta,(u,s),(_,n) in zip(data['meta'],predictions,negative_predictions):
        xy=plane_point(u,meta['z'],meta['camera'])[:2]
        rows.append(dict(**{k:meta[k] for k in ('name','domain','baseline_cup','group','index')},uv=u.tolist(),score=s,synthetic_absent_score=n,
            error_mm=float(np.linalg.norm(xy-meta['xy'])*1000)))
    return rows

def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not h['recording']['active']
    assert not (ROOT/'cup-locator-ensemble.json').exists() and not (ROOT/'scope-selection.json').exists()
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4)
    sources=[ROOT/'neural-localization-supplement-r2/cup-neural-localization.pt',ROOT/'cup-neural-localization-attempt.pt']
    first=torch.load(sources[0],map_location='cpu',weights_only=False)['bank']['state']
    second=torch.load(sources[1],map_location='cpu',weights_only=False)['state'];models=[];members=[]
    for state in (first,second):
        model=CupLocator().cuda().eval();model.load_state_dict(state);models.append(model)
        identity=hashlib.sha256(b''.join(v.numpy().tobytes() for _,v in sorted(state.items()))).hexdigest()
        members.append(dict(kind='cup_heatmap',state=state,identity=identity))
    plan,data,cal=collect(dense=True);rows={s:evaluate(models,d) for s,d in data.items()}
    native=[r for rs in rows.values() for r in rs if r['domain']=='native_detail'];primary=[r for r in native if r['baseline_cup']]
    minimum=min(r['score'] for r in primary);maximum=max(r['synthetic_absent_score'] for r in native);threshold=(minimum+maximum)/2
    config=dict(window=20,votes=8,radius_px=1.5,fraction=.6,score_floor=max(.01,threshold*.6));windows=[]
    for split,values in rows.items():
        for name in sorted({r['name'] for r in values if r['domain']=='native_detail' and r['baseline_cup']}):
            sequence=sorted([r for r in values if r['name']==name and r['group']=='obs'],key=lambda r:r['index']);history=[];accepted=None
            for r in sequence:
                history.append((r['uv'],r['score']));accepted=consensus(history,threshold,config)
                if accepted is not None:break
            meta=next(m for m in data[split]['meta'] if m['name']==name)
            error=None if accepted is None else float(np.linalg.norm(plane_point(accepted['uv'],meta['z'],meta['camera'])[:2]-meta['xy'])*1000)
            windows.append(dict(name=name,split=split,confirmed=accepted is not None,error_mm=error,frames=len(history)))
    gate=plan['component_gates'];frame_max=max(r['error_mm'] for r in primary)
    passed=frame_max<gate['localization_max_error_mm'] and minimum-maximum>gate['localization_negative_margin'] and all(w['confirmed'] and w['error_mm']<gate['localization_max_error_mm'] for w in windows)
    report=dict(passed=passed,scope='baseline_cup_only',sources={str(p):sha(p) for p in sources},frame_max_error_mm=frame_max,
        frame_max_by_split={s:max(r['error_mm'] for r in v if r['domain']=='native_detail' and r['baseline_cup']) for s,v in rows.items()},
        minimum_positive_score=minimum,maximum_synthetic_absent_score=maximum,threshold=threshold,windows=windows,rows=rows,
        source_sha256=sha(Path(__file__)),module_sha256=sha(ROOT/'staged/pour_cup_locator.py'),
        method='One fixed equal-weight average of two supervised RGB center estimates. Minimum confidence, rejection above 6 pixels model disagreement. No weight search, fitting or reserved images; unchanged 3 mm frame and temporal gates. Resized autonomy remains gated.')
    if passed:
        torch.save(dict(bank=dict(kind='cup_heatmap_ensemble',members=members,max_disagreement_px=6.),calibration=cal,threshold=threshold,consensus=config),ROOT/'cup-neural-localization.pt')
        report['checkpoint_sha256']=sha(ROOT/'cup-neural-localization.pt')
        (ROOT/'scope-selection.json').write_text(json.dumps(dict(passed=True,scope='baseline_cup_only',checkpoint_sha256=report['checkpoint_sha256'],
            max_validation_error_mm=report['frame_max_by_split']['validation'],max_training_error_mm=report['frame_max_by_split']['train'],
            max_temporal_error_mm=max(w['error_mm'] for w in windows),threshold=threshold,source_sha256=sha(Path(__file__)),method=report['method']),indent=2)+'\n')
    (ROOT/'cup-locator-ensemble.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','windows')}))
if __name__=='__main__':main()
