"""Two additional layouts, preserving the original twelve and the same actor."""
import copy
import hashlib
import json
from pathlib import Path
from pour_layout_variety import PARENT, CHECKPOINT_SHA

ROOT=Path('/workspace/learning/A-layout-variety-extra-20261105-v1')
PREFIX='VX2_A_'

def make_plan(base):
    cases=[];layouts=[]
    pairs=[('Outer diagonal',(.60,-.21),(.52,-.35)),
           ('Outer diagonal: positions swapped',(.52,-.35),(.60,-.21))]
    for i,(label,cup,bottle) in enumerate(pairs,start=12):
        spec=copy.deepcopy(base)
        spec.update(name=f'{PREFIX}test_{i:02}',protocol=17,layout_index=i,layout_label=label,
            cup_world=[*cup,1.04],source_xy=[bottle[1]-cup[1],cup[0]-bottle[0]],
            expected_stop='cup_near_full' if i%2==0 else 'bottle_empty',initial_ml=94. if i%2==0 else 30.,cup_initial_ml=6.,
            checkpoint=str(PARENT/'best.pt'),policy_evaluation=True,expert_recovery=False,record_cameras=True,record_overview=True)
        cases.append(spec);layouts.append(dict(name=spec['name'],label=label,cup=list(cup),bottle=list(bottle)))
    return dict(test=cases,layouts_world_xy=layouts,checkpoint_sha256=CHECKPOINT_SHA,
        scope='Two additional once-only stress layouts after the initial finger-envelope audit flagged three of the original twelve. Same frozen actor; no training or threshold changes. Original failures retained.',
        policy='One attempt per layout, no teacher fallback or retries; video includes all original twelve plus these two.')

def plan():return json.loads((ROOT/'frozen-plan.json').read_text())

def unstarted():
    selection=json.loads((ROOT/'selection.json').read_text())
    assert hashlib.sha256((PARENT/'best.pt').read_bytes()).hexdigest()==CHECKPOINT_SHA
    assert hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest()==selection['scene_plan_sha256']
    for name,digest in selection['runtime_sources'].items():
        assert hashlib.sha256(Path('/workspace/src',name).read_bytes()).hexdigest()==digest,name
    started={json.loads(p.read_text())['name'] for p in Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}*/parameters.json')}
    return [s for s in plan()['test'] if s['name'] not in started]
