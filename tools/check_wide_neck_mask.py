"""Check production neck-mask normalization against the recorded CPU probe."""
import hashlib,importlib.util,json,sys
from pathlib import Path
STUDY=Path('/workspace/learning/A-wide-20261105-v1')
sys.path[:0]=[str(STUDY/'staged'),'/workspace/src']
import numpy as np
import torch
from validate_pour_recording import h5py,decode
import pour_guard_vision as original

torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('neck_candidate','/tmp/pour_guard_vision_candidate.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
report=json.loads((STUDY/'partial-neck-mask-cpu-probe.json').read_text())
names=set(report['template_names']);entries=[]
for path in sorted(Path('/workspace/outputs/fountain').glob('*/*T30_A_demo_*/episode.hdf5')):
    with h5py.File(path) as f:
        s=json.loads(f.attrs['spec'])
        if s['name'] not in names:continue
        assert s['study_split']=='train' and f.attrs['status']=='complete'
        d=f['data/demo_0'];cal=dict(K=f['calibration/task/intrinsics'][:],T=d['obs/task_world_T_camera'][0],table_z=1.04)
        xyz=d['ground_truth/bottle_pose'][0,:3]
        entries.append(dict(name=s['name'],image=decode(d['obs/task_image'][0],'jpeg95'),anchor=original.project(xyz+[0,0,.135],cal),xyz=xyz,cal=cal))
assert {e['name'] for e in entries}==names
images=[e['image'] for e in entries];anchors=[e['anchor'] for e in entries];checks=[]
with torch.inference_mode():
    for absolute in (True,False):
        base=original.make_bank(images,anchors,'source',device='cpu',aggregation='max',absolute_gradients=absolute,anchor_radius_px=32.)
        for box in (None,(10,8,38,40),(12,8,36,36),(14,8,34,34)):
            new=candidate.make_bank(images,anchors,'source',device='cpu',aggregation='max',absolute_gradients=absolute,anchor_radius_px=32.,source_mask_box=box)
            expected=base['kernels'].clone();mask=base['mask'].clone()
            if box:
                x0,y0,x1,y1=box;keep=torch.zeros_like(mask);keep[:,:,y0:y1,x0:x1]=1
                norm=((expected/mask.clamp(min=1e-12)).square()*keep).sum((1,2,3),keepdim=True).sqrt().clamp(min=1e-6)
                expected=expected*keep/norm;mask*=keep
            error=float((new['kernels']-expected).abs().max())
            assert error<1e-6 and torch.equal(new['mask'],mask)
            new['bounds']=[24,24,360,350]
            e=next(e for e in entries if e['name']=='T30_A_demo_train_05')
            uv,score=candidate.locate(e['image'],new,device='cpu')
            mm=float(np.linalg.norm(candidate.bottle_base_from_neck(uv,e['cal'])[:2]-e['xyz'][:2])*1000)
            if box:assert mm<2.,(absolute,box,mm)
            checks.append(dict(absolute_gradients=absolute,box=box,kernel_max_abs_error=error,case=e['name'],localization_error_mm=mm))
output=STUDY/'neck-mask-production-check.json';assert not output.exists()
output.write_text(json.dumps(dict(checks=checks,production_source_sha256=hashlib.sha256(Path('/tmp/pour_guard_vision_candidate.py').read_bytes()).hexdigest(),scope='CPU normalization equivalence and repaired training-image regression only; no held-out task result.'),indent=2)+'\n')
print(json.dumps(checks),flush=True)
