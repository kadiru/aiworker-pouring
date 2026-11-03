"""Thirty reserved cases for the unchanged material-A controller; no fitting."""
import copy
import hashlib
import json
from pathlib import Path
import random

ROOT=Path('/workspace/learning/A-robust-20261103-v1')
PARENT=Path('/workspace/learning/A-guard-20261103-v2')
PREFIX='T30_A_'
SEED=20261130
CHECKPOINT_SHA='b80ff58388e8f2516975f1925b20aa98c5720ab395bba3fe04dfb508fac64b1b'
RUNTIME_SOURCES=('pour_guard_actor.py','pour_guard_train.py','pour_guard_models.py','pour_guard_vision.py',
    'pour_guard_trajectory.py','pour_guard_study.py','pour_guard_data.py','pour_stop_actor.py',
    'pour_stop_models.py','pour_stop_sequence.py','run_fountain.py','robot_pour.py','pour_recording.py',
    'bottle_pour.py','pour_fluid.py','pour_geometry.py','vessel_appearance.py','pour_guard_vote_study.py','pour_robust_study.py')


def make_plan(base):
    rng=random.Random(SEED);targets=['bottle_empty']*15+['cup_near_full']*15;rng.shuffle(targets);cases=[]
    for i,target in enumerate(targets):
        source=rng.randint(20,40) if target=='bottle_empty' else rng.randint(85,100)
        cup=rng.randint(3,10) if target=='bottle_empty' else rng.randint(3,14)
        s=copy.deepcopy(base)
        s.update(name=f'{PREFIX}test_{i:02}',protocol=13,study_split='test',checkpoint=str(ROOT/'best.pt'),vessel_appearance='A',
            initial_ml=float(source),cup_initial_ml=float(cup),expected_stop=target,
            cup_world=[.49+rng.uniform(-.008,.008),-.15+rng.uniform(-.008,.008),1.04],
            source_xy=[-.12+rng.uniform(-.009,.009),rng.uniform(-.004,.004)])
        assert s['spacing']==.0018 and s['dt']==.0025 and s['vessel_appearance']=='A'
        assert s['policy_evaluation'] and not s['expert_recovery'] and s['max_seconds']==40.
        cases.append(s)
    return dict(test=cases,test_seed=SEED,count=30,stratification='15 full / 15 empty, shuffled before execution',
        scope='Existing calibrated A workspace: cup +/-8mm XY, source +/-9mm X and +/-4mm Y; no material or physics change')


def plan():
    p=ROOT/'frozen-plan.json'
    if p.exists():return json.loads(p.read_text())
    base=json.loads((PARENT/'frozen-plan.json').read_text())['test'][1]
    result=make_plan(base);ROOT.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');return result


def unstarted():
    frozen=json.loads((ROOT/'final-selection.json').read_text())
    assert hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest()==CHECKPOINT_SHA==frozen['selected_checkpoint_sha256']
    assert hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest()==frozen['scene_plan_sha256']
    for name,h in frozen['runtime_sources'].items():assert hashlib.sha256(Path('/workspace/src',name).read_bytes()).hexdigest()==h,name
    seen={json.loads(p.read_text())['name'] for p in Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}*/parameters.json')}
    return [s for s in plan()['test'] if s['name'] not in seen]
