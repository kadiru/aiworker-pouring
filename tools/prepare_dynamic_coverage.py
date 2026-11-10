"""Declare extra low-fill training data before any new simulation images exist."""
import copy,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/workspace/src')
from pour_perception_geometry import calibration_seed_points
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not any(Path('/workspace/outputs/fountain').glob('*/*SD_A_*/parameters.json'))
    protocol=read(ROOT/'protocol.json');plan=read(ROOT/'plan.json');cases=[]
    for i,(h,d) in enumerate((h,d) for h in (.9,1.,1.1) for d in (.9,1.,1.1)):
        refs=[s for s in plan['collect'] if s['name'].startswith('SC_A_static_train_') and s['cup_height_scale']==h and s['cup_diameter_scale']==d]
        assert len(refs)==4
        for j,source in enumerate(refs[:2]):
            s=copy.deepcopy(source);s.update(name=f'SD_A_low_r0_{i:02d}_{j:02d}',calibration_initial_height=.29,study_split='train',coverage_role='new_low_training')
            assert sha(Path(s['calibration_particle_seed']))==s['calibration_particle_seed_sha256']
            points=calibration_seed_points(s,np.load(s['calibration_particle_seed'])['cup_local_points'])
            s['cup_initial_ml']=len(points)*s['spacing']**3*1e6;cases.append(s)
    for s in protocol['seed']:s.update(study_split='train',coverage_role='high_seed_training')
    shutil.copy2(ROOT/'protocol.json',ROOT/'protocol-v1.json')
    protocol.update(version=2,low_train=cases,low_fill_rationale='Independent color fits fail low-fill generalization. Add 18 measured cut-0.29 training scenes at two existing training layouts per size; no validation/test identity changes.',motion_definition='Start upright beside cup, smoothly tilt to 65 degrees with colored source or 82 degrees empty, then return; transverse sinusoidal translation. Authored fixture only. Travel is maximum pairwise observed displacement.')
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(low_training=len(cases),high_seed_training=len(protocol['seed']),protocol_sha256=sha(ROOT/'protocol.json'))))
if __name__=='__main__':main()
