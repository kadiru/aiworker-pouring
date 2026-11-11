"""Draft training-only interpolation coverage, retaining all reserved fixtures."""
import copy,json
from pathlib import Path
import numpy as np
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged
from pour_perception_geometry import calibration_seed_points

def main():
    protected_unchanged();old=read(ROOT/'protocol.json');assert old['version']==5
    assert not read(ROOT/'pair-r8.json')['passed']
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    p=copy.deepcopy(old);extra=[]
    for i in range(9):
        for j in range(2):
            parent=next(s for s in old['train'] if s['name']==f'SD_A_static_train_r2_{i:02d}_{j}_0')
            for k,cut in enumerate((.30,.34)):
                s=copy.deepcopy(parent);s.update(name=f'SD_A_static_train_r3_{i:02d}_{j}_{k}',
                    calibration_initial_height=cut,coverage_role='static_midfill_interpolation',study_split='train')
                assert not s.get('calibration_motion')
                assert sha(Path(s['calibration_particle_seed']))==s['calibration_particle_seed_sha256']
                points=calibration_seed_points(s,np.load(s['calibration_particle_seed'])['cup_local_points'])
                s['cup_initial_ml']=len(points)*s['spacing']**3*1e6;extra.append(s)
    p['train']+=extra;p.update(version=6,midfill_train_names=[s['name'] for s in extra],
        previous_protocol_sha256=sha(ROOT/'protocol.json'),
        midfill_rationale='Training-only .30/.34 seed cuts between existing .24/.38 cuts at the same two training layouts across all nine sizes. Addresses development low-fill interpolation errors without reusing validation or reserved frames. Actual levels must be measured and reviewed; no synthetic labels.')
    assert all(p[k]==old[k] for k in ('validation','holdout','gates'))
    with (ROOT/'protocol-midfill-v6-draft.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n')
    print(json.dumps(dict(drafted_training_cases=len(extra),activated=False)),flush=True)

if __name__=='__main__':main()
