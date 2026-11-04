"""CPU-only matching ablation on consumed references, never W30 task tests."""
import json
import numpy as np
import torch
from pour_workspace_data import sequences,cover_workspace
from pour_workspace_study import PARENT,ROOT
from pour_guard_vision import make_bank,locate,project,bottle_base_from_neck

torch.set_num_threads(2)
saved=torch.load(PARENT/'best.pt',map_location='cpu',weights_only=False)
seq=sequences();cover_workspace(saved)
train=[e for e in seq if e['split']=='train' and e['name']==e['reference_case']]
anchors=[project(e['source_xyz']+[0,0,.135],saved['calibration']) for e in train]
report={}
for absolute,aggregation,radius in [(False,'max',28.),(False,'top3',35.),(True,'median',None),(True,'max',28.)]:
    key=f'absolute{absolute}-{aggregation}-radius{radius}'
    bank=make_bank([e['images'][0] for e in train],anchors,'source',device='cpu',
        aggregation=aggregation,absolute_gradients=absolute,anchor_radius_px=radius)
    bank['bounds']=saved['source_bank']['bounds'];rows=[]
    for e in seq:
        im=e['images'][0];uv,score=locate(im,bank,device='cpu')
        point=bottle_base_from_neck(uv,saved['calibration'])
        row=dict(name=e['name'],split=e['split'],uv=uv.tolist(),score=score,error_mm=float(np.linalg.norm(point[:2]-e['source_xyz'][:2])*1000))
        image=im.copy();x,y=np.rint(project(e['source_xyz']+[0,0,.135],saved['calibration'])).astype(int)
        image[max(0,y-14):min(384,y+38),max(0,x-24):min(384,x+24)]=np.median(image[10:40,10:40],axis=(0,1)).astype(np.uint8)
        n,s=locate(image,bank,device='cpu');row.update(negative_score=s,negative_uv=n.tolist());rows.append(row)
    report[key]=rows
    print(key,'max_error',max(r['error_mm'] for r in rows),'min_score',min(r['score'] for r in rows),'max_neg',max(r['negative_score'] for r in rows),flush=True)
    print('bad',[(r['name'],r['error_mm'],r['score'],r['negative_score']) for r in rows if r['error_mm']>=2 or r['score']<=max(x['negative_score'] for x in rows)],flush=True)
    (ROOT/'localization-batched-probe.json').write_text(json.dumps(report,indent=2)+'\n')
