"""Frozen finite scene plans for the A-only learning study."""
import json
from pathlib import Path
import numpy as np
from bottle_pour import bottle_spec, START_X

ROOT=Path('/workspace/learning/A-20261102-v1')


def scene(base, name, cup_dx, cup_dy, source_dx, source_dy, source_ml, cup_ml=0, split='train'):
    spec=bottle_spec(base,small=source_ml<60)
    return dict(spec,name=name,robot=True,protocol=7,record_cameras=True,control_period=.05,
                opacity=1.,roughness=.22,initial_ml=float(source_ml),cup_initial_ml=float(cup_ml),
                cup_world=[.49+cup_dx,-.15+cup_dy,1.04],source_xy=[START_X+source_dx,source_dy],
                expected_stop='cup_near_full' if source_ml+cup_ml>=70 else 'bottle_empty',
                max_seconds=32.,target_fraction=.90,return_flow_seconds=.28,
                gentle=True,mouth_z=.052,study_split=split,image_encoding='jpeg95',
                save_particle_trajectory=False,save_scene=False)


def corrective(base):
    rng=np.random.default_rng(20261104)
    result=[]
    for i,(source,cup) in enumerate(zip([28,95,40,85,18,100,35,90],[0,14,8,0,14,8,0,14])):
        d=rng.uniform(-1,1,4)*[.008,.008,.008,.004]
        s=scene(base,f'A_train_noise_{i:02}',*d,source,cup,'train')
        result.append(dict(s,protocol=8,noisy_expert=True,arm_noise_deg=[.25,.4,.55][i%3],
                           noise_seed=20261104+i,corrective_prefix=True,prefix_seconds=3.3))
    return result


def plan(base):
    rng=np.random.default_rng(20261102)
    scenes=[]
    for i in range(20):
        d=rng.uniform(-1,1,4)*[.010,.010,.012,.006]
        full=i%2==0
        ml=([85,95,100][i%3] if full else [18,28,40][i%3])
        cup=[0,8,14][i%3]
        split='train' if i<16 else 'validation'
        scenes.append(scene(base,f'A_{split}_{i:02}',*d,ml,cup,split))
    tests=[]
    for i,ml in enumerate([24,90,34,100,22,92]):
        d=rng.uniform(-1,1,4)*[.009,.009,.010,.005]
        tests.append(scene(base,f'A_test_{i:02}',*d,ml,[5,11][i%2],'test'))
    extra_rng=np.random.default_rng(20261103)
    extras=[scene(base,f'A_train_{20+i:02}',*(extra_rng.uniform(-1,1,4)*[.010,.010,.012,.006]),
                  ml,cup,'train') for i,(ml,cup) in enumerate([(85,14),(100,0),(18,14),(40,0)])]
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'frozen-plan.json'
    if path.exists():
        frozen=json.loads(path.read_text())
        if any(s['initial_ml']>100 for part in frozen.values() for s in part):
            (ROOT/'frozen-plan-original.json').write_text(path.read_text())
            for part in frozen.values():
                for s in part: s['initial_ml']=min(100.,s['initial_ml'])
            path.write_text(json.dumps(frozen,indent=2)+'\n')
        names={s['name'] for s in frozen['train']}
        if any(s['name'] not in names for s in extras):
            (ROOT/'frozen-plan-before-fill-crosses.json').write_text(path.read_text())
            frozen['train'].extend(s for s in extras if s['name'] not in names)
            path.write_text(json.dumps(frozen,indent=2)+'\n')
        recovery=corrective(base)
        names={s['name'] for s in frozen['train']}
        if any(s['name'] not in names for s in recovery):
            (ROOT/'frozen-plan-before-corrective.json').write_text(path.read_text())
            frozen['train'].extend(s for s in recovery if s['name'] not in names)
            path.write_text(json.dumps(frozen,indent=2)+'\n')
        if any(s.get('noisy_expert') and not s.get('corrective_prefix') for s in frozen['train']):
            for s in frozen['train']:
                if s.get('noisy_expert'): s.update(corrective_prefix=True,prefix_seconds=3.3)
            path.write_text(json.dumps(frozen,indent=2)+'\n')
        return frozen
    frozen=dict(train=[s for s in scenes if s['study_split']=='train']+extras+corrective(base),
                validation=[s for s in scenes if s['study_split']=='validation'],test=tests)
    path.write_text(json.dumps(frozen,indent=2)+'\n')
    return frozen


def calibration(base):
    return [scene(base,'A_calibrate_full',0,0,0,0,95,0,'calibration'),
            scene(base,'A_calibrate_empty',.005,-.005,-.008,.004,28,8,'calibration')]
