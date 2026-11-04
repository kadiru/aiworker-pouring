"""Corrective development and thirty new reserved material-A cases."""
import copy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path('/workspace/learning/A-reliable-20261103-v1')
PARENT = Path('/workspace/learning/A-settled-20261103-v1')
PREFIX = 'C30_A_'
SEED = 20261217
PARENT_SHA = '18f0064ea9f1503488ba606a863a8047bf1f3cd8ff05dfb95a81b53e24121527'
RUNTIME_SOURCES = ('pour_guard_actor.py', 'pour_guard_models.py', 'pour_guard_vision.py',
    'pour_guard_trajectory.py', 'pour_stop_actor.py', 'pour_stop_models.py', 'pour_stop_sequence.py',
    'run_fountain.py', 'robot_pour.py', 'pour_recording.py', 'bottle_pour.py', 'pour_fluid.py',
    'pour_geometry.py', 'vessel_appearance.py', 'pour_settled_model.py', 'pour_settled_data.py',
    'pour_settled_train.py', 'pour_settled_study.py', 'pour_temporal_localization.py',
    'pour_reliable_train.py', 'pour_reliable_study.py', 'pour_reliable_data.py')


def make_plan(previous):
    development = []
    for i in (19, 26, 29, 1, 14, 0, 10):
        s = copy.deepcopy(previous['test'][i])
        s.update(name=f'{PREFIX}dev_{i:02}', study_split='regression', protocol=14,
            checkpoint=str(ROOT / 'best.pt'), development_case=i)
        development.append(s)
    rng = random.Random(SEED)
    targets = ['bottle_empty'] * 15 + ['cup_near_full'] * 15
    rng.shuffle(targets)
    tests = []
    for i, target in enumerate(targets):
        s = copy.deepcopy(previous['test'][0])
        s.update(name=f'{PREFIX}test_{i:02}', study_split='test', protocol=14,
            checkpoint=str(ROOT / 'best.pt'), expected_stop=target,
            initial_ml=float(rng.randint(20, 40) if target == 'bottle_empty' else rng.randint(85, 100)),
            cup_initial_ml=float(rng.randint(3, 10) if target == 'bottle_empty' else rng.randint(3, 14)),
            cup_world=[.49 + rng.uniform(-.008, .008), -.15 + rng.uniform(-.008, .008), 1.04],
            source_xy=[-.12 + rng.uniform(-.009, .009), rng.uniform(-.004, .004)], vessel_appearance='A')
        tests.append(s)
    return dict(development=development, test=tests, test_seed=SEED,
        test_policy='30 once after final selection, 15 full/15 empty, no fitting or tuning during tests',
        scope='Same material A and calibrated position/fill ranges as prior T30 batch')


def plan():
    ROOT.mkdir(parents=True, exist_ok=True)
    p = ROOT / 'frozen-plan.json'
    if not p.exists():
        prior = json.loads((PARENT / 'frozen-plan.json').read_text())
        p.write_text(json.dumps(make_plan(prior), indent=2) + '\n')
    return json.loads(p.read_text())


def seen():
    return [json.loads(p.read_text()) for p in Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}*/parameters.json')]


def development():
    assert not (ROOT / 'final-selection.json').exists(), 'Final controller already frozen'
    prior = seen()
    round_id = max((s.get('development_round', -1) for s in prior), default=-1) + 1
    cases = [dict(s, name=s['name'] + f'_r{round_id}', development_round=round_id) for s in plan()['development']]
    # Prioritize the surge and nonflow counterexamples in development.
    # This changes development scheduling only, never the reserved test plan.
    return sorted(cases, key=lambda s: {19:0,26:1,29:2}.get(s['development_case'],2))


def unstarted():
    frozen = json.loads((ROOT / 'final-selection.json').read_text())
    assert hashlib.sha256((ROOT / 'best.pt').read_bytes()).hexdigest() == frozen['selected_checkpoint_sha256']
    assert hashlib.sha256((ROOT / 'frozen-plan.json').read_bytes()).hexdigest() == frozen['scene_plan_sha256']
    for n, sha in frozen['runtime_sources'].items():
        assert hashlib.sha256(Path('/workspace/src', n).read_bytes()).hexdigest() == sha, n
    started = {s['name'] for s in seen()}
    return [s for s in plan()['test'] if s['name'] not in started]
