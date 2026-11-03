"""Independent frozen recovery study. Original six tests never enter fitting."""
import json
from pathlib import Path
import numpy as np
from pour_study import scene,plan as original_plan,ROOT as ORIGINAL_ROOT

ROOT=Path('/workspace/learning/A-recovery-20261102-v1')


def plan(base):
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'frozen-plan.json'
    if path.exists():return json.loads(path.read_text())
    old=original_plan(base)
    # The viewer's selected fountain may differ from the accepted learning liquid.
    # Inherit the frozen v5 reference physics, never the currently selected preview.
    base=old['train'][1]
    frozen=dict(train=list(old['train']),validation=list(old['validation']),test=[])
    for split,seed,pairs in (
        ('train',20261105,[(90,8),(28,5),(85,14),(35,0),(98,0),(22,11)]),
        ('validation',20261106,[(92,5),(32,11)]),
        ('test',20261107,[(26,4),(94,10),(38,6),(88,12),(20,9),(99,3)])):
        rng=np.random.default_rng(seed)
        for i,(source,cup) in enumerate(pairs):
            delta=rng.uniform(-1,1,4)*[.008,.008,.009,.004]
            spec=scene(base,f'R_A_{split}_{i:02}',*delta,source,cup,split)
            if split!='test':
                spec.update(expert_recovery=True,policy_evaluation=True,takeover_seconds=8.,
                    checkpoint=str(ORIGINAL_ROOT/'best-bc-v5.pt'),max_seconds=38.,grasp_deadline=8.,protocol=9,
                    target_fraction=.91)
            frozen[split].append(spec)
    path.write_text(json.dumps(frozen,indent=2)+'\n')
    return frozen
