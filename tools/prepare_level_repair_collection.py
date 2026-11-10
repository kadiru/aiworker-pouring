"""Draft additive training coverage; retain all old/validation/reserved specs."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/workspace/src')
from pour_perception_geometry import calibration_seed_points
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    old=read(ROOT/'protocol.json');assert old['version']==3
    p=copy.deepcopy(old);extra=[]
    for i in range(9):
        source=next(s for s in old['train'] if s['name']==f'SD_A_motion_train_r0_{i:02d}_high')
        for j,(xy,source_xy) in enumerate([([.52,-.29,1.04],[.14934177945010857,-.0025900826859361903]),([.55,-.32,1.04],[.14934177945010857,-.0025900826859361903])]):
            for k,height in enumerate((.24,.38)):
                s=copy.deepcopy(source);s.update(name=f'SD_A_motion_train_r1_{i:02d}_{j}_{k}',initial_ml=20.,cup_world=xy,source_xy=source_xy,
                    calibration_initial_height=height,coverage_role='moving_low_intermediate_repair',study_split='train',layout_reference=f'new_repair_layout_{j}',swapped=True,
                    layout_label=f'Low-fill repair layout {j}',calibration_motion=dict(maximum_angle_deg=65.,lateral_amplitude_m=.02,start_s=.5,duration_s=2.))
                assert sha(Path(s['calibration_particle_seed']))==s['calibration_particle_seed_sha256']
                points=calibration_seed_points(s,np.load(s['calibration_particle_seed'])['cup_local_points'])
                s['cup_initial_ml']=len(points)*s['spacing']**3*1e6;extra.append(s)
    p['train']+=extra;p.update(version=4,repair_train_names=[s['name'] for s in extra],previous_protocol_sha256=sha(ROOT/'protocol.json'),
        repair_rationale='36 new training-only moving views: all nine cup sizes at two new layouts and cut fractions .24/.38, labels measured natively. Existing 171 training/36 validation episodes retained; all 18 reserved specs and gates unchanged. Frozen t1 tracker is required for every new frame. No new autonomous scenes.')
    assert all(p[k]==old[k] for k in old if k not in ('version','train'))
    with (ROOT/'protocol-repair-v4-draft.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n')
    print(json.dumps(dict(extra_training=len(extra),total_moving_train=len(p['train']),draft_sha256=sha(ROOT/'protocol-repair-v4-draft.json'),activated=False)))
if __name__=='__main__':main()
