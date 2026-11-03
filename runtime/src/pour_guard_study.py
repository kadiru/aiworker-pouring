"""New reserved scenes for visual grasp confirmation and settled-level stopping."""
import json
from pathlib import Path
import numpy as np
from pour_study import scene,ROOT as ORIGINAL

ROOT=Path('/workspace/learning/A-guard-20261103-v1')
PREFIX='G_A_'


def plan():
    ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/'frozen-plan.json'
    if p.exists():return json.loads(p.read_text())
    base=json.loads((ORIGINAL/'frozen-plan.json').read_text())['train'][1];out={}
    for split,seed,pairs in (
        ('validation',20261121,[(94,11),(32,5)]),
        ('test',20261122,[(26,4),(91,12),(39,8),(87,14),(20,3),(98,6)])):
        rng=np.random.default_rng(seed);out[split]=[]
        for i,(source,cup) in enumerate(pairs):
            d=rng.uniform(-1,1,4)*[.008,.008,.009,.004]
            s=scene(base,f'{PREFIX}{split}_{i:02}',*d,source,cup,split)
            s.update(protocol=11,policy_evaluation=True,expert_recovery=False,
                policy_family='visual_guard_hierarchy',checkpoint=str(ROOT/'best.pt'),
                max_seconds=40.,grasp_deadline=12.,goal_definition='surface_height_or_source_empty',
                isosurface_max_subgrids=32768,isosurface_max_vertices=524288,isosurface_max_triangles=1048576)
            assert s['spacing']==.0018 and s['dt']==.0025
            out[split].append(s)
    prior=json.loads(Path('/workspace/learning/A-stop-20261103-v1/frozen-plan.json').read_text())
    out['regression']=[]
    for name,index in (('grasp',1),('full',3)):
        s=dict(prior['test'][index],name=PREFIX+'regression_'+name,study_split='regression',protocol=11,
            policy_family='visual_guard_hierarchy',checkpoint=str(ROOT/'best.pt'),max_seconds=40.,grasp_deadline=12.)
        out['regression'].append(s)
    p.write_text(json.dumps(out,indent=2)+'\n');return out
