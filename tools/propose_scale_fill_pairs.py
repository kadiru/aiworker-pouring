"""Prepare, but never queue, missing low/full pairs at training layouts."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
sys.path[:0] = ['/workspace/src', '/workspace']
from pour_perception_geometry import calibration_seed_points

ROOT = Path('/workspace/learning/A-scale-20261109-v1')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / 'fill-pair-supplement-proposal.json'
    assert not output.exists()
    plan = read(ROOT / 'plan.json')
    manifest = read(ROOT / 'candidate.json')
    audit = read(ROOT / 'collect-r0-audit.json')
    assert sha(ROOT / 'plan.json') == manifest['plan_sha256'] == audit['plan_sha256']
    assert audit['complete'] and all(r['label']['accepted'] for r in audit['records'])
    cases, inventory = [], []
    for s in plan['collect']:
        if s['study_split'] != 'train':
            continue
        assert s['calibration_only'] and not s['policy_evaluation']
        seed = Path(s['calibration_particle_seed'])
        assert sha(seed) == s['calibration_particle_seed_sha256']
        with np.load(seed, allow_pickle=False) as f:
            points = f['cup_local_points'].copy()
        for cut in (.22, 1.):
            if s['calibration_initial_height'] == cut:
                continue
            new = copy.deepcopy(s)
            new.update(name=s['name'].replace('SC_A_static_train_', 'SC_A_pair_train_') +
                       ('_low' if cut == .22 else '_high'),
                       calibration_initial_height=cut, proposed_pair_of=s['name'])
            selected = calibration_seed_points(new, points)
            new['cup_initial_ml'] = len(selected) * new['spacing']**3 * 1e6
            cases.append(new)
            inventory.append(dict(name=new['name'], particles=len(selected),
                                  initial_positions_sha256=hashlib.sha256(selected.tobytes()).hexdigest()))
    assert len(cases) == 54
    assert not {s['name'] for s in cases} & {s['name'] for s in plan['collect'] + plan['holdout']}
    report = dict(proposal_only=True, queued=False, model_fitted=False, new_training_episodes=54,
        resulting_train_episodes=90, unchanged_validation_episodes=18, unchanged_holdout_episodes=9,
        cases=cases, inventories=inventory, parent_plan_sha256=sha(ROOT / 'plan.json'),
        parent_audit_sha256=sha(ROOT / 'collect-r0-audit.json'), helper_sha256=sha(Path(__file__)),
        rationale='Each size will have low and near-full observations at all four existing training layouts, retaining intermediate-fill scenes. This addresses observed within-size fill/layout confounding; it is not a proven cure.',
        required_before_use=['Archive completed fitting evidence and declare a new immutable collection plan',
            'Verify idle runtime, isolation and geometry for the declared cases',
            'Record each new scene once; audit native recording, actual stable labels and loss',
            'Inspect new camera views and bind review to the updated whole-episode audit',
            'Refit from training data only; retain validation and unused holdout identities'],
        limitations='Seed cuts are initial conditions, not measured labels. This proposal does not collect data, change the active plan, consume holdouts or accept resized autonomy.')
    with output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    assert sha(ROOT / 'plan.json') == manifest['plan_sha256']
    print(json.dumps({k: report[k] for k in ('proposal_only', 'queued', 'new_training_episodes',
                                           'resulting_train_episodes', 'unchanged_validation_episodes')}))


if __name__ == '__main__':
    main()
