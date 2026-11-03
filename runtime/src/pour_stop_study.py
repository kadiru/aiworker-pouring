"""Fresh, immutable scene split for visual-stop and terminal-sequence study."""
import json
from pathlib import Path
import numpy as np
from pour_study import ROOT as ORIGINAL_ROOT, scene

ROOT=Path('/workspace/learning/A-stop-20261103-v1')


def plan():
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'frozen-plan.json'
    if path.exists():return json.loads(path.read_text())
    base=json.loads((ORIGINAL_ROOT/'frozen-plan.json').read_text())['train'][1]
    frozen={}
    for split,seed,pairs in (
        ('validation',20261112,[(96,7),(30,9)]),
        ('test',20261113,[(23,6),(93,4),(37,10),(89,13),(19,2),(97,9)])):
        rng=np.random.default_rng(seed);cases=[]
        for i,(source,cup) in enumerate(pairs):
            delta=rng.uniform(-1,1,4)*[.008,.008,.009,.004]
            s=scene(base,f'S_A_{split}_{i:02}',*delta,source,cup,split)
            s.update(protocol=10,policy_evaluation=True,expert_recovery=False,
                policy_family='visual_stop_hierarchy',checkpoint=str(ROOT/'best.pt'),
                max_seconds=32.,grasp_deadline=8.,goal_definition='surface_height_or_source_empty',
                isosurface_max_subgrids=32768,isosurface_max_vertices=524288,isosurface_max_triangles=1048576)
            assert s['spacing']==.0018 and s['dt']==.0025
            cases.append(s)
        frozen[split]=cases
    path.write_text(json.dumps(frozen,indent=2)+'\n');return frozen
