"""Separate, once-only side-return development and reserved evaluation."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/learning/A-side-return-20261108-v1')
PREFIX = 'SR_A_'

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def pending(part):
    manifest = read(ROOT/'candidate.json')
    assert sha(ROOT/'best.pt') == manifest['checkpoint_sha256']
    assert sha(ROOT/'plan.json') == manifest['plan_sha256']
    assert all(sha(Path('/workspace/src')/n) == d for n,d in manifest['runtime_sources'].items())
    assert all(sha(Path(n)) == d for n,d in manifest['protected_parent_files'].items())
    plan = read(ROOT/'plan.json')
    if part == 'development':assert manifest['stage']=='development_candidate', 'Complete candidate checks before development'
    if (ROOT/'final-selection.json').exists():
        assert part == 'test', 'Reserved evaluation is frozen'
    if part == 'test':
        selection = read(ROOT/'final-selection.json')
        for key in ('checkpoint_sha256','plan_sha256','runtime_sources'):
            assert selection[key] == manifest[key]
        assert selection['development_passes'] == 6
        assert selection['development_acceptances'] == 6
        assert sha(ROOT/'development-outcomes.json') == selection['development_outcomes_sha256']
        development = read(ROOT/'development-outcomes.json')
        assert development['counts']['completed'] == development['counts']['accepted'] == 6
        assert sha(ROOT/'development-manual-review.json') == selection['development_manual_review_sha256']
        assert read(ROOT/'development-manual-review.json')['accepted']
    started = {read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*'+PREFIX+'*/parameters.json')}
    cases = plan[part]
    assert not {s['name'] for s in cases} & started, 'Never retry or silently resume a started split'
    for s in cases:
        for key in ('cup_height_scale','cup_diameter_scale','bottle_height_scale','bottle_diameter_scale','bottle_neck_scale'):
            assert s.get(key,1.) == 1., 'Size generalization remains gated'
    return cases
