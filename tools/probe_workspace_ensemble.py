"""Select confidence maps from saved consumed-window detections, on CPU."""
import json
import numpy as np
import torch
from pour_workspace_study import ROOT,PARENT
from pour_temporal_localization import consensus
from pour_guard_vision import bottle_base_from_neck

saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
old=json.loads((ROOT/'fit-attempt0/localization-candidates.json').read_text())
new=json.loads((ROOT/'fit-attempt1/localization-candidates.json').read_text())
observations={e['name']:e for e in old['observations']}
old_negative={e['name']:e for e in old['synthetic_negatives']}
new_negative={e['name']:e for e in new['synthetic_negatives']};results=[]
for anchor in (.343780,.38,.40,.45,.50):
    for slope in (.5,.75,1.,1.25,1.5):
        errors=[];misses=[];bad=[];negative=[]
        for e in new['observations']:
            history=[];accepted=None
            for i,d in enumerate(e['detections']):
                o=observations[e['name']]['detections'][i]['source'];n=d['source']
                score=.7+slope*(o['score']-anchor)
                history.append((o['uv'],score) if score>n['score'] else (n['uv'],n['score']))
                accepted=consensus(history,.7,dict(window=20,votes=8,radius_px=1.,fraction=.6,score_floor=.30))
                if accepted:
                    error=float(np.linalg.norm(bottle_base_from_neck(accepted['uv'],saved['calibration'])[:2]-e['source_xyz'][:2])*1000)
                    errors.append(error)
                    if error>2:bad.append((e['name'],error))
                    break
            if not accepted:misses.append(e['name'])
            negative.append(max(.7+slope*(old_negative[e['name']]['score']-anchor),new_negative[e['name']]['score']))
        row=dict(anchor=anchor,slope=slope,max_error=max(errors),misses=misses,bad=bad,max_negative=max(negative));results.append(row)
        if not misses and not bad and max(negative)<.7:print('ELIGIBLE',row,flush=True)
(ROOT/'localization-ensemble-probe.json').write_text(json.dumps(results,indent=2)+'\n')
