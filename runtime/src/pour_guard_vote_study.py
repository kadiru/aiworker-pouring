"""Independent follow-up scenes; the completed G_A round remains immutable."""
import json
from pathlib import Path
import numpy as np
from pour_study import scene,ROOT as ORIGINAL

ROOT=Path('/workspace/learning/A-guard-20261103-v2')
PREFIX='H_A_'
TEST_SEED=20261126


def plan():
    ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/'frozen-plan.json'
    if p.exists():return json.loads(p.read_text())
    prior=json.loads(Path('/workspace/learning/A-guard-20261103-v1/frozen-plan.json').read_text())
    out=dict(regression=[],validation=[],test=[],test_seed=TEST_SEED)
    for label,index in (('grasp',0),('empty',4),('full',1)):
        s=dict(prior['test'][index],name=PREFIX+'regression_'+label,study_split='regression',protocol=12,
            checkpoint=str(ROOT/'best.pt'))
        out['regression'].append(s)
    base=json.loads((ORIGINAL/'frozen-plan.json').read_text())['train'][1]
    rng=np.random.default_rng(TEST_SEED)
    for i,(source,cup) in enumerate(((27,5),(90,13),(37,8),(88,14),(21,3),(98,6))):
        d=rng.uniform(-1,1,4)*[.008,.008,.009,.004]
        s=scene(base,f'{PREFIX}test_{i:02}',*d,source,cup,'test')
        s.update(protocol=12,policy_evaluation=True,expert_recovery=False,
            policy_family='visual_guard_hierarchy',checkpoint=str(ROOT/'best.pt'),max_seconds=40.,grasp_deadline=12.,
            goal_definition='surface_height_or_source_empty',isosurface_max_subgrids=32768,
            isosurface_max_vertices=524288,isosurface_max_triangles=1048576)
        assert s['spacing']==.0018 and s['dt']==.0025
        out['test'].append(s)
    p.write_text(json.dumps(out,indent=2)+'\n');return out
