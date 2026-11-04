"""CPU calibration of view-local penalties, with a retained failed window."""
import json
import numpy as np
import torch
from pour_workspace_data import sequences,cover_workspace
from pour_workspace_study import PARENT,ROOT
from pour_guard_vision import make_bank,locate,project,bottle_base_from_neck
from pour_temporal_localization import consensus

torch.set_num_threads(2)
saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
seq=sequences();cover_workspace(saved)
train=[e for e in seq if e['split']=='train' and e['name']==e['reference_case']]
anchors=[project(e['source_xyz']+[0,0,.135],saved['calibration']) for e in train]
bank=make_bank([e['images'][0] for e in train],anchors,'source',device='cpu',aggregation='max',absolute_gradients=True,anchor_radius_px=28.)
bank['bounds']=saved['source_bank']['bounds'];report={}
for penalty in (.05,.10,.20):
    bank['anchor_penalty']=penalty;rows=[];failed=[]
    for e in seq:
        im=e['images'][0];uv,score=locate(im,bank,device='cpu')
        row=dict(name=e['name'],score=score,error_mm=float(np.linalg.norm(bottle_base_from_neck(uv,saved['calibration'])[:2]-e['source_xyz'][:2])*1000))
        image=im.copy();x,y=np.rint(project(e['source_xyz']+[0,0,.135],saved['calibration'])).astype(int)
        image[max(0,y-14):min(384,y+38),max(0,x-24):min(384,x+24)]=np.median(image[10:40,10:40],axis=(0,1)).astype(np.uint8)
        n,s=locate(image,bank,device='cpu');row['negative_score']=s;rows.append(row)
        if e['name']=='C30_A_test_02':
            for i,image in enumerate(e['images']):
                uv,score=locate(image,bank,device='cpu');error=float(np.linalg.norm(bottle_base_from_neck(uv,saved['calibration'])[:2]-e['source_xyz'][:2])*1000)
                failed.append(dict(index=i,uv=uv.tolist(),score=score,error_mm=error))
    report[str(penalty)]=dict(initial=rows,failed_window=failed)
    print('penalty',penalty,'initial max error',max(x['error_mm'] for x in rows),'min positive',min(x['score'] for x in rows),'max negative',max(x['negative_score'] for x in rows),'failed window max',max(x['error_mm'] for x in failed),flush=True)
    (ROOT/'localization-view-support-probe.json').write_text(json.dumps(report,indent=2)+'\n')
