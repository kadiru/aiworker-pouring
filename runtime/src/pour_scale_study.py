"""Separate size calibration; no autonomous use before explicit acceptance."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/learning/A-scale-20261109-v1')
PREFIX = 'SC_A_'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pending(part):
    manifest = json.loads((ROOT / 'candidate.json').read_text())
    assert sha(ROOT / 'best.pt') == manifest['checkpoint_sha256']
    assert sha(ROOT / 'plan.json') == manifest['plan_sha256']
    assert all(sha(Path('/workspace/src') / n) == h for n, h in manifest['runtime_sources'].items())
    assert all(sha(Path(n)) == h for n, h in manifest['protected_parent_files'].items())
    plan = json.loads((ROOT / 'plan.json').read_text())
    assert part in ('pilot', 'collect', 'holdout'), 'Autonomous size work is not accepted yet'
    if part == 'collect':
        assert manifest.get('seed_and_camera_accepted'), 'Review physical seeds and cameras first'
    if part == 'holdout':
        selection = json.loads((ROOT / 'component-selection.json').read_text())
        assert all(selection[k] == manifest[k] for k in ('checkpoint_sha256', 'plan_sha256', 'runtime_sources'))
        assert selection['all_size_component_gates_passed']
    cases = plan[part]
    assert cases and len({s['name'] for s in cases}) == len(cases)
    assert all(s.get('calibration_only') and not s.get('policy_evaluation') for s in cases)
    started = {json.loads(p.read_text())['name'] for p in Path('/workspace/outputs/fountain').glob('*/*' + PREFIX + '*/parameters.json')}
    # Collection may exceed the server's 32-case cap; only unstarted IDs may queue.
    return [s for s in cases if s['name'] not in started]
