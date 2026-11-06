"""Wider and exactly swapped fixtures, with disjoint reference/dev/test scenes."""
import copy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path('/workspace/learning/A-wide-20261105-v1')
OUTPUTS = Path('/workspace/outputs/fountain')
SOURCES = Path('/workspace/src')
PARENT = Path('/workspace/learning/A-workspace-20261104-v1')
PREFIX = 'XW30_A_'
# The four pilot references began before a historical T30 test-name collision
# was discovered. Retain the reference names and recordings; only dev/test IDs
# use the unused namespace. Every reference is also required to be protocol 17.
REFERENCE_PREFIX = 'T30_A_'
PARENT_SHA = 'a909abb318ecc980ad8450484ca98ba07990f9501cd641744b5ab6544732f51b'
TASK_FOCAL_M = .030
ANCHORS = (
    ((.49, -.15), (.49, -.27)),
    ((.54, -.12), (.44, -.30)),
    ((.43, -.12), (.54, -.31)),
    ((.58, -.16), (.58, -.30)),
    ((.60, -.21), (.52, -.35)),
)


def candidates(base, split, seed):
    """Yield paired layouts for deterministic geometry-only screening."""
    rng = random.Random(seed)
    for candidate in range(400):
        a, b = ANCHORS[candidate % len(ANCHORS)]
        jitter = .012
        a = [v + rng.uniform(-jitter, jitter) for v in a]
        b = [v + rng.uniform(-jitter, jitter) for v in b]
        pair = []
        for swap in (False, True):
            cup, bottle = (b, a) if swap else (a, b)
            s = copy.deepcopy(base)
            target = 'cup_near_full' if (candidate + int(swap)) % 2 == 0 else 'bottle_empty'
            s.update(protocol=17, study_split=split, candidate_index=candidate,
                swapped=swap, layout_family=candidate % len(ANCHORS),
                layout_label=f'Family {candidate % len(ANCHORS) + 1}' + (' / swapped' if swap else ''),
                cup_world=[*cup, 1.04], source_xy=[bottle[1] - cup[1], cup[0] - bottle[0]],
                expected_stop=target, initial_ml=float(rng.randint(88, 100) if target == 'cup_near_full' else rng.randint(24, 38)),
                cup_initial_ml=float(rng.randint(4, 10)), record_cameras=True, record_overview=True,
                task_camera_focal_m=TASK_FOCAL_M, approach_clearance_m=.07,
                policy_evaluation=split in ('development', 'test'), expert_recovery=False,
                checkpoint=str(ROOT / 'best.pt'), policy_family='visual_guard_hierarchy',
                max_seconds=45., grasp_deadline=14., save_scene=False, save_particle_trajectory=False)
            if split in ('train', 'validation'):
                s.pop('policy_family', None)
                s.pop('checkpoint', None)
                s.update(reference_hold_seconds=1., reference_approach_extension_s=1.25,
                         expert_slow_rate_deg_s=5., target_fraction=.88)
            pair.append(s)
        yield pair


def plan():
    return json.loads((ROOT / 'frozen-plan.json').read_text())


def seen():
    rows=[]
    for prefix in (PREFIX,REFERENCE_PREFIX+'demo_'):
        for p in OUTPUTS.glob(f'*/*{prefix}*/parameters.json'):
            s=json.loads(p.read_text())
            if s.get('protocol')==17:rows.append(s)
    return rows


def references(limit=None):
    assert not (ROOT / 'final-selection.json').exists()
    started = {s['name'] for s in seen()}
    rows = [s for s in plan()['demonstrations'] if s['name'] not in started]
    repairs = ROOT / 'reference-repair-plan.json'
    if repairs.exists():
        rows += [s for s in json.loads(repairs.read_text())['demonstrations'] if s['name'] not in started]
    return rows[:limit] if limit is not None else rows


def repairs():
    assert not (ROOT / 'final-selection.json').exists()
    started={s['name'] for s in seen()}
    return [s for s in json.loads((ROOT/'reference-repair-plan.json').read_text())['demonstrations']
            if s['name'] not in started]


def development():
    assert not (ROOT / 'final-selection.json').exists()
    round_id = max((s.get('development_round', -1) for s in seen()), default=-1) + 1
    return [dict(s, name=s['name'] + f'_r{round_id}', development_round=round_id)
            for s in plan()['development']]


def tests():
    frozen = json.loads((ROOT / 'final-selection.json').read_text())
    assert hashlib.sha256((ROOT / 'best.pt').read_bytes()).hexdigest() == frozen['selected_checkpoint_sha256']
    assert hashlib.sha256((ROOT / 'frozen-plan.json').read_bytes()).hexdigest() == frozen['scene_plan_sha256']
    for name, digest in frozen['runtime_sources'].items():
        assert hashlib.sha256((SOURCES/name).read_bytes()).hexdigest() == digest, name
    started = {s['name'] for s in seen()}
    return [s for s in plan()['test'] if s['name'] not in started]
