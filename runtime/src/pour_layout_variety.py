"""Twelve explicit table layouts; once-only evaluation of the frozen v18 actor."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/learning/A-layout-variety-20261105-v1')
PARENT = Path('/workspace/learning/A-workspace-20261104-v1')
PREFIX = 'V12_A_'
CHECKPOINT_SHA = 'a909abb318ecc980ad8450484ca98ba07990f9501cd641744b5ab6544732f51b'

# World XY, metres. Paired rows deliberately exchange the actual object positions.
LAYOUTS = (
    ('Familiar arrangement', (.49, -.15), (.49, -.27)),
    ('Exact front/back swap', (.49, -.27), (.49, -.15)),
    ('Side by side: bottle nearer robot', (.55, -.22), (.42, -.22)),
    ('Side by side: positions swapped', (.42, -.22), (.55, -.22)),
    ('Wide diagonal', (.54, -.12), (.44, -.30)),
    ('Wide diagonal: positions swapped', (.44, -.30), (.54, -.12)),
    ('Opposite diagonal', (.43, -.12), (.54, -.31)),
    ('Opposite diagonal: positions swapped', (.54, -.31), (.43, -.12)),
    ('Both nearer robot', (.40, -.16), (.40, -.30)),
    ('Both farther from robot', (.58, -.16), (.58, -.30)),
    ('Far end of table', (.50, -.04), (.46, -.17)),
    ('Near end of table', (.48, -.30), (.44, -.38)),
)


def make_plan(base):
    cases = []
    for i, (label, cup, bottle) in enumerate(LAYOUTS):
        spec = copy.deepcopy(base)
        spec.update(name=f'{PREFIX}test_{i:02}', protocol=16, study_split='test',
            layout_label=label, layout_index=i, cup_world=[*cup, 1.04],
            source_xy=[bottle[1]-cup[1], cup[0]-bottle[0]],
            expected_stop='cup_near_full' if i % 2 == 0 else 'bottle_empty',
            initial_ml=94. if i % 2 == 0 else 30., cup_initial_ml=6.,
            checkpoint=str(PARENT/'best.pt'), policy_evaluation=True,
            expert_recovery=False, record_cameras=True, record_overview=True,
            max_seconds=40., grasp_deadline=12.)
        cases.append(spec)
    return dict(test=cases, layouts_world_xy=[dict(name=s['name'], label=label,
        cup=list(cup), bottle=list(bottle)) for s, (label, cup, bottle) in zip(cases, LAYOUTS)],
        checkpoint_sha256=CHECKPOINT_SHA,
        policy='Current frozen v18 actor. Twelve once-only autonomous tests, no fitting, threshold changes, teacher fallback or retries. All outcomes appear in the video.',
        camera_note='Task/wrist cameras and actor inputs unchanged. Extra overview camera is recording-only.',
        scope='Deliberate larger layout stress test, not an iid generalization-rate estimate. Six full targets and six empty targets; material A and existing physics.')


def plan():
    return json.loads((ROOT/'frozen-plan.json').read_text())


def unstarted():
    frozen = json.loads((ROOT/'selection.json').read_text())
    assert hashlib.sha256((PARENT/'best.pt').read_bytes()).hexdigest() == CHECKPOINT_SHA
    assert hashlib.sha256((ROOT/'frozen-plan.json').read_bytes()).hexdigest() == frozen['scene_plan_sha256']
    for name, digest in frozen['runtime_sources'].items():
        assert hashlib.sha256(Path('/workspace/src', name).read_bytes()).hexdigest() == digest, name
    started = set()
    for path in Path('/workspace/outputs/fountain').glob(f'*/*{PREFIX}*/parameters.json'):
        spec = json.loads(path.read_text())
        assert spec['name'] not in started, 'Duplicate layout attempt'
        started.add(spec['name'])
    return [s for s in plan()['test'] if s['name'] not in started]
