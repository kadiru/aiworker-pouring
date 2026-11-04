"""Threefold position domain with separate demonstrations, development and tests."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
from pour_reliable_study import RUNTIME_SOURCES as PARENT_SOURCES

ROOT = Path('/workspace/learning/A-workspace-20261104-v1')
PARENT = Path('/workspace/learning/A-reliable-20261103-v1')
PARENT_SHA = '40b5506bc4bceb51f1bf8ab8a5df93525933da0b03dea4c7e3fc66b63ba62101'
PREFIX = 'W30_A_'
SEED = 20261224
RANGES = (.024, .024, .027, .012)
RUNTIME_SOURCES = tuple(dict.fromkeys(PARENT_SOURCES + (
    'pour_workspace_study.py', 'pour_workspace_data.py', 'pour_workspace_train.py',
    'pour_workspace_motion.py', 'pour_workspace_path.py', 'pour_stop_data.py', 'pour_learning.py')))


def scene(base, name, delta, target, rng, split, expert=False):
    s = copy.deepcopy(base)
    s.update(name=name, protocol=15, study_split=split, vessel_appearance='A',
        cup_world=[.49+delta[0], -.15+delta[1], 1.04],
        source_xy=[-.12+delta[2], delta[3]], expected_stop=target,
        initial_ml=float(rng.randint(20,40) if target=='bottle_empty' else rng.randint(85,100)),
        cup_initial_ml=float(rng.randint(3,10) if target=='bottle_empty' else rng.randint(3,14)),
        checkpoint=str(ROOT/'best.pt'), policy_evaluation=not expert, expert_recovery=False,
        max_seconds=40., grasp_deadline=12.)
    if expert:
        s.pop('policy_family',None); s.pop('checkpoint',None)
        s.update(target_fraction=.88, expert_slow_rate_deg_s=5.)
    return s


def outside_old(rng):
    while True:
        d=[rng.uniform(-r,r) for r in RANGES]
        if max(abs(x)/(r/3) for x,r in zip(d,RANGES))>=1.25:
            return d


def make_plan(previous):
    base=previous['test'][0]; rng=random.Random(20261222); demos=[]
    for i,signs in enumerate(itertools.product((-1,1),repeat=4)):
        d=[a*b for a,b in zip(signs,RANGES)]
        # Parity balances full/empty at both ends of each position dimension.
        target='cup_near_full' if sum(a>0 for a in signs)%2==0 else 'bottle_empty'
        demos.append(scene(base,f'{PREFIX}demo_train_{i:02}',d,target,rng,'train',True))
    for i in range(4):
        demos.append(scene(base,f'{PREFIX}demo_validation_{i:02}',outside_old(rng),
            'cup_near_full' if i%2==0 else 'bottle_empty',rng,'validation',True))
    rng=random.Random(20261223); development=[]
    for i in range(8):
        s=scene(base,f'{PREFIX}dev_{i:02}',outside_old(rng),
            'cup_near_full' if i%2==0 else 'bottle_empty',rng,'regression')
        s['development_case']=i; development.append(s)
    rng=random.Random(SEED); targets=['cup_near_full']*15+['bottle_empty']*15; rng.shuffle(targets)
    tests=[scene(base,f'{PREFIX}test_{i:02}',outside_old(rng),target,rng,'test') for i,target in enumerate(targets)]
    return dict(demonstrations=demos,development=development,test=tests,test_seed=SEED,
        half_ranges_m=RANGES,position_center=dict(cup_world=[.49,-.15,1.04],source_xy=[-.12,0.]),
        test_policy='30 once after final selection, 15 full/15 empty, no fitting/tuning/retries during tests',
        scope='Material A; threefold half-ranges in all four position coordinates. Every development/test scene extends beyond the old domain by at least25% in one coordinate. Other physical/fill parameters and primary score unchanged.')


def plan():
    ROOT.mkdir(parents=True,exist_ok=True); p=ROOT/'frozen-plan.json'
    if not p.exists():p.write_text(json.dumps(make_plan(json.loads((PARENT/'frozen-plan.json').read_text())),indent=2)+'\n')
    return json.loads(p.read_text())


def seen():
    return [json.loads(p.read_text()) for p in Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}*/parameters.json')]


def reference_specs():
    """Retain original attempts; optional repairs have new, explicit names."""
    original=plan()['demonstrations'];by_name={s['name']:s for s in original}
    path=ROOT/'reference-repair-plan.json'
    repairs=json.loads(path.read_text())['demonstrations'] if path.exists() else []
    names=set(by_name)
    for s in repairs:
        base=by_name[s['reference_case']]
        assert s['name']==base['name']+f"_repair_r{s['reference_repair_round']}"
        assert s['name'] not in names;names.add(s['name'])
        assert s['reference_repair_round']>=1 and s['approach_clearance_m']==.04
        for k,v in base.items():
            if k!='name':assert s[k]==v,(s['name'],k)
    return original+repairs


def demonstrations():
    assert not (ROOT/'final-selection.json').exists()
    started={s['name'] for s in seen()}
    return [s for s in reference_specs() if s['name'] not in started]


def development():
    assert not (ROOT/'final-selection.json').exists()
    r=max((s.get('development_round',-1) for s in seen()),default=-1)+1
    return [dict(s,name=s['name']+f'_r{r}',development_round=r) for s in plan()['development']]


def unstarted():
    f=json.loads((ROOT/'final-selection.json').read_text())
    assert hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest()==f['selected_checkpoint_sha256']
    assert hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest()==f['scene_plan_sha256']
    for n,h in f['runtime_sources'].items():assert hashlib.sha256(Path('/workspace/src',n).read_bytes()).hexdigest()==h,n
    started={s['name'] for s in seen()}
    return [s for s in plan()['test'] if s['name'] not in started]
