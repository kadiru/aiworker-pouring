"""Archive the rejected stage and declare exactly 54 additional training scenes."""
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
import numpy as np
sys.path[:0] = ['/workspace/src', '/workspace']
from pour_perception_geometry import calibration_seed_points
from scale_pair_plan import extend
ROOT = Path('/workspace/learning/A-scale-20261109-v1')


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v): p.write_text(json.dumps(v, indent=2) + '\n')


def main():
    h = json.load(urllib.request.urlopen('http://192.168.68.60:8212/health', timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    m = read(ROOT/'candidate.json'); plan = read(ROOT/'plan.json'); proposal = read(ROOT/'fill-pair-supplement-proposal.json')
    closure = read(ROOT/'static-fit-closure.json'); audit = read(ROOT/'collect-r0-audit.json')
    assert closure['completed_fits'] == 3 and not closure['accepted_locator']
    assert sha(ROOT/'plan.json') == m['plan_sha256'] == proposal['parent_plan_sha256'] == closure['plan_sha256']
    assert sha(ROOT/'best.pt') == m['checkpoint_sha256'] and not m['trained']
    assert sha(ROOT/'collect-r0-audit.json') == proposal['parent_audit_sha256']
    assert audit['complete'] and all(r['label']['accepted'] for r in audit['records'])
    assert all(sha(Path(p)) == v for p, v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n) == v for n, v in m['runtime_sources'].items())
    new = extend(plan, proposal)
    started = {read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in proposal['cases'] + plan['holdout']}
    inventories = {r['name']:r for r in proposal['inventories']}
    for s in proposal['cases']:
        seed = Path(s['calibration_particle_seed']); assert sha(seed) == s['calibration_particle_seed_sha256']
        with np.load(seed, allow_pickle=False) as f: points = calibration_seed_points(s, f['cup_local_points'])
        assert hashlib.sha256(points.tobytes()).hexdigest() == inventories[s['name']]['initial_positions_sha256']
        assert len(points) == inventories[s['name']]['particles']
    archive = ROOT/'collection-r0'; archive.mkdir()
    names = ('plan.json','candidate.json','collect-geometry.json','holdout-geometry.json',
             'collect-r0-audit.json','collect-r0-manual-review.json','static-fit-closure.json',
             'fit_scale_locator.py','run_scale_locator_fit.py','fit_scale_level.py',
             'select_scale_static_components.py','evaluate_scale_static_holdout.py','scale_data_admission.py')
    saved = {}
    for name in names:
        shutil.copy2(ROOT/name, archive/name); saved[name] = sha(archive/name)
        assert saved[name] == sha(ROOT/name)
    rejected = ROOT/'locator-rounds/r2'; rejected.mkdir()
    for name in ('cup-neural-localization.json','cup-neural-localization-attempt.pt',
                 'cup-locator-data-manifest.json','fit_scale_locator.py','run_scale_locator_fit.py',
                 'scale_data_admission.py','scale_locator_augmentation.py'):
        shutil.copy2(ROOT/name, rejected/name); assert sha(ROOT/name) == sha(rejected/name)
    write(archive/'index.json', dict(files=saved, rejected_r2=str(rejected)))
    for name in ('cup-neural-localization.json','cup-neural-localization-attempt.pt','cup-locator-data-manifest.json'):
        assert sha(ROOT/name) == sha(rejected/name); (ROOT/name).unlink()
    write(ROOT/'plan.json', new)
    m.update(plan_sha256=sha(ROOT/'plan.json'), stage='paired_fill_collection',
             collection_audit='collect-r1-audit.json', collection_review='collect-r1-manual-review.json',
             collection_round='r1', trained=False, accepted_locator=False)
    write(ROOT/'candidate.json', m)
    write(ROOT/'pair-collection-preparation.json', dict(new_cases=54, training=90, validation=18,
        unchanged_holdouts=9, plan_sha256=m['plan_sha256'], previous_plan_sha256=saved['plan.json'],
        proposal_sha256=sha(ROOT/'fill-pair-supplement-proposal.json'),
        helper_sha256=sha(Path(__file__)), plan_helper_sha256=sha(ROOT/'scale_pair_plan.py'),
        archive_index_sha256=sha(archive/'index.json'), geometry_required=True, recordings_generated=False))
    print(json.dumps(dict(new_cases=54, training=90, validation=18, unused_holdouts=9, plan_sha256=m['plan_sha256'])))


if __name__ == '__main__': main()
