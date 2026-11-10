"""Reproduce initial-window fusion feasibility; this is not a tracker or acceptance."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
sys.path[:0]=['/workspace/src',str(ROOT)]
from pour_guard_vision import plane_point
from pour_temporal_localization import consensus
def read(n):return json.loads((ROOT/n).read_text())
def main():
    artifact=torch.load(ROOT/'rim-d2.pt',map_location='cpu',weights_only=False)
    fine=read('rim-d2.json')['selected'];old=read('rim-reuse-probe.json')['results'];coarse=read('coarse-d2.json')['selected'];rows=[]
    for alpha in (0.,.25,.5,.75,1.):
        case={}
        for split,key in (('train','training'),('validation','validation')):
            history={};confirmed={};errors=[]
            for x,y,m in zip(old[split]['rows'],fine[key]['rows'],coarse[key]['rows']):
                assert (x['name'],x['group'],x['index'])==(y['name'],y['group'],y['index'])==(m['name'],m['group'],m['index'])
                if x['group']!='obs':continue
                uv=np.asarray(x['uv'])*(1-alpha)+np.asarray(y['uv'])*alpha
                errors.append(float(np.linalg.norm(plane_point(uv,m['z'],artifact['calibration'])[:2]-m['xy'])*1000))
                h=history.setdefault(x['name'],[]);h.append((uv.tolist(),min(x['score'],y['score'])))
                if x['name'] not in confirmed:
                    result=consensus(h,artifact['threshold'],artifact['consensus'])
                    if result is not None:confirmed[x['name']]=float(np.linalg.norm(plane_point(result['uv'],m['z'],artifact['calibration'])[:2]-m['xy'])*1000)
            case[split]=dict(initial_frame_max_mm=max(errors),window_max_mm=max(confirmed.values()),confirmed=len(confirmed),bad_windows={k:v for k,v in confirmed.items() if v>=3})
        rows.append(dict(alpha=alpha,**case))
    original=read('rim-temporal-feasibility.json')
    assert rows==original['rows'],'Replay differs from the recorded diagnostic'
    print(json.dumps(dict(reproduced=True,accepted=False,rows=rows)))
if __name__=='__main__':main()
