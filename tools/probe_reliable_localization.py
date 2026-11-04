"""CPU diagnostic on already consumed development; no controller mutation."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from validate_pour_recording import h5py,decode
from pour_guard_vision import make_bank,locate,project,bottle_base_from_neck
from pour_temporal_localization import consensus
ROOT=Path('/workspace/learning/A-reliable-20261103-v1')
torch.set_num_threads(2)
saved=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
bundle=torch.load(ROOT/'reliable-cache-v1.pt',map_location='cpu',weights_only=False)
seq=bundle['localization_sequences'];validation={'L30_A_test_01','L30_A_test_06','L30_A_test_11','L30_A_test_15','L30_A_test_21','L30_A_test_27'}
train=[e for e in seq if e['name'] not in validation]
new=make_bank([e['images'][0] for e in train],[project(e['source_xyz']+[0,0,.135],saved['calibration']) for e in train],'source',device='cpu')
new['bounds']=saved['source_bank']['bounds'];banks={'parent':saved['source_bank'],'refit24':new}
fail=next(Path('/workspace/outputs/fountain').glob('*/*C30_A_dev_19_r0*/episode.hdf5'))
with h5py.File(fail,'r') as f:
 d=f['data/demo_0'];images=[decode(x,'jpeg95') for x in d['obs/task_image'][:]];xyz=d['ground_truth/bottle_pose'][0,:3]
out=dict(checkpoint=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest(),diagnostic_only=True,variants={})
for name,bank in banks.items():
 rows=[]
 for i,image in enumerate(images):
  uv,score=locate(image,bank,device='cpu');error=float(np.linalg.norm(bottle_base_from_neck(uv,saved['calibration'])[:2]-xyz[:2])*1000)
  rows.append(dict(index=i,uv=uv.tolist(),score=score,error_mm=error))
 positives=[];neg=[]
 for e in seq:
  image=e['images'][0];uv,score=locate(image,bank,device='cpu');error=float(np.linalg.norm(bottle_base_from_neck(uv,saved['calibration'])[:2]-e['source_xyz'][:2])*1000)
  positives.append(dict(name=e['name'],validation=e['name'] in validation,score=score,error_mm=error))
  image=image.copy();uv=project(e['source_xyz']+[0,0,.135],saved['calibration']);x,y=np.rint(uv).astype(int)
  image[max(0,y-14):min(384,y+38),max(0,x-24):min(384,x+24)]=np.median(image[10:40,10:40],axis=(0,1)).astype(np.uint8)
  uv,score=locate(image,bank,device='cpu');neg.append(score)
 out['variants'][name]=dict(kernel_count=len(bank['kernels']),failed_development=rows,initial_views=positives,synthetic_negative_scores=neg)
 print(name,'failed scores',min(x['score'] for x in rows),max(x['score'] for x in rows),'max error',max(x['error_mm'] for x in rows),'initial max error',max(x['error_mm'] for x in positives),'negative max',max(neg),flush=True)
(ROOT/'localization-refit-probe.json').write_text(json.dumps(out,indent=2)+'\n')
