"""CPU-only lifted-neck check on the fixed partial training-reference set."""
import argparse,hashlib,json,sys
from pathlib import Path
STUDY=Path('/workspace/learning/A-wide-20261105-v1')
sys.path[:0]=[str(STUDY/'staged'),'/workspace/src']
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from validate_pour_recording import h5py,decode
from pour_guard_vision import make_bank,locate,project

parser=argparse.ArgumentParser();parser.add_argument('--held-templates',action='store_true');args=parser.parse_args()
torch.set_num_threads(1);names=set(json.loads((STUDY/'partial-neck-mask-cpu-probe.json').read_text())['template_names']);entries=[]
for path in sorted(Path('/workspace/outputs/fountain').glob('*/*T30_A_demo_*/episode.hdf5')):
    with h5py.File(path) as f:
        spec=json.loads(f.attrs['spec'])
        if spec['name'] not in names:continue
        assert spec['study_split']=='train' and f.attrs['status']=='complete'
        d=f['data/demo_0'];cal=dict(K=f['calibration/task/intrinsics'][:],T=d['obs/task_world_T_camera'][0],table_z=1.04)
        base=d['ground_truth/bottle_pose'][0,:3];initial=project(base+[0,0,.135],cal);expected=project(base+[0,0,.135+.044],cal)
        phase=np.array([x.decode() for x in d['ground_truth/phase'][:]])
        indices=np.flatnonzero((phase=='Lifting and checking grip')&d['ground_truth/grasp_verified'][:])[-6:]
        assert len(indices)==6
        frames=[]
        for i in indices:
            pose=d['ground_truth/bottle_pose'][i];neck=pose[:3]+Rotation.from_quat(pose[3:]).apply([0,0,.135])
            frames.append(dict(index=int(i),t=float(d['next_time'][i]),image=decode(d['next_obs/task_image'][i],'jpeg95'),true_uv=project(neck,cal)))
        entries.append(dict(name=spec['name'],image=decode(d['obs/task_image'][0],'jpeg95'),anchor=initial,expected=expected,frames=frames))
assert {e['name'] for e in entries}==names
reports=[]
with torch.inference_mode():
    for absolute in (True,False):
        for box in ((10,8,38,40),(12,8,36,36),(14,8,34,34)):
            images=[e['frames'][-1]['image'] if args.held_templates else e['image'] for e in entries]
            anchors=[e['frames'][-1]['true_uv'] if args.held_templates else e['anchor'] for e in entries]
            bank=make_bank(images,anchors,'source',device='cpu',aggregation='max',absolute_gradients=absolute,anchor_radius_px=32.,source_mask_box=box)
            rows=[]
            for e in entries:
                for frame in e['frames']:
                    uv,score=locate(frame['image'],bank,tracking=True,expected_uv=e['expected'],device='cpu')
                    expected=e['expected']-e['anchor'];actual=uv-e['anchor']
                    passes=score>.23 and np.linalg.norm(expected)>=8 and np.dot(actual,expected)>.65*np.dot(expected,expected) and np.linalg.norm(uv-e['expected'])<12
                    rows.append(dict(name=e['name'],index=frame['index'],t=frame['t'],score=score,uv=uv.tolist(),true_uv=frame['true_uv'].tolist(),
                        visual_predicate=bool(passes),neck_error_px=float(np.linalg.norm(uv-frame['true_uv']))))
            result=dict(absolute_gradients=absolute,mask_box=box,frames=len(rows),visual_predicate_passes=sum(r['visual_predicate'] for r in rows),
                max_neck_error_px=max(r['neck_error_px'] for r in rows),rows=rows)
            reports.append(result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
out=STUDY/('partial-held-bank-tracking-probe.json' if args.held_templates else 'partial-lifted-neck-tracking-probe.json');assert not out.exists()
out.write_text(json.dumps(dict(scope='Six held-bottle frames per already-consumed training reference; component check only, not held-out or autonomous performance. Ground truth labels held-template anchors and offline errors, never autonomous runtime inputs.',
    held_templates=args.held_templates,templates=sorted(names),candidates=reports,source_sha256=hashlib.sha256((STUDY/'staged/pour_guard_vision.py').read_bytes()).hexdigest()),indent=2)+'\n')
