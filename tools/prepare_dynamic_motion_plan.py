"""Declare moving train/validation/reserved identities before fitting new views.

The draft permits CPU clearance work while high seeds are being measured. It
cannot be queued: the active protocol must later bind reviewed high-seed hashes.
"""
import copy,hashlib,json
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=read(ROOT/'protocol.json');base=read(ROOT/'plan.json')
    finger=read(Path('/workspace/learning/A-perception-20261108-v1/gripper-initialization.json'))
    for split in ('train','validation','holdout'):
        cases=[]
        for i,(h,d) in enumerate((h,d) for h in (.9,1.,1.1) for d in (.9,1.,1.1)):
            refs=[s for s in base['collect'] if s['cup_height_scale']==h and s['cup_diameter_scale']==d and s['name'].startswith('SC_A_static_'+('train' if split=='train' else 'validation')+'_')]
            assert len(refs)==(4 if split=='train' else 2)
            for regime in ('low','high'):
                source=next(s for s in refs if (s['calibration_initial_height']==1.)==(regime=='high'))
                s=copy.deepcopy(source)
                s.update(name=f'SD_A_motion_{split}_r0_{i:02d}_{regime}',study_split=split,max_seconds=3.,
                    calibration_hold_angle_deg=0.,calibration_finger_initial_deg=finger['angles_deg'],
                    initial_ml=20. if regime=='low' else 0.,coverage_role='moving_'+regime,
                    calibration_motion=dict(maximum_angle_deg=65. if regime=='low' else 82.,lateral_amplitude_m=.02,start_s=.5,duration_s=2.),
                    high_seed_binding_pending=regime=='high')
                if split=='holdout':
                    s['cup_world'][0]+=.008 if i%2 else -.008
                    s['cup_world'][1]+=.009 if i%2 else -.009
                    s['source_xy'][0]+=.006 if s['source_xy'][0]>0 else -.006
                cases.append(s)
        p[split]=cases
    p.update(version=3,moving_source_finger_sha256=sha(Path('/workspace/learning/A-perception-20261108-v1/gripper-initialization.json')),
        moving_frame_evaluation=dict(start_s=.5,end_s=2.5,stride=2,label='Native same-timestamp cup surface height; all moving frames scored, not just settled final frames'),
        draft_only=True)
    with (ROOT/'motion-draft.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n')
    print(json.dumps(dict(train=len(p['train']),validation=len(p['validation']),reserved=len(p['holdout']),draft_sha256=sha(ROOT/'motion-draft.json'))))
if __name__=='__main__':main()
