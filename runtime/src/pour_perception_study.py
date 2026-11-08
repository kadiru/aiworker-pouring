"""Isolated perception calibration with explicit splits and once-only episodes."""
import hashlib
import json
from pathlib import Path
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
PREFIX='PV_A_'

def check_freeze(selection,manifest):
    for key in ('checkpoint_sha256','plan_sha256','runtime_sources'):
        assert selection[key]==manifest[key],f'Frozen perception {key} changed'

def pending(part):
    manifest=json.loads((ROOT/'candidate.json').read_text())
    assert hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    for name,digest in manifest['runtime_sources'].items():
        assert hashlib.sha256((Path('/workspace/src')/name).read_bytes()).hexdigest()==digest,name
    plan=json.loads((ROOT/'plan.json').read_text())
    assert hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest()==manifest['plan_sha256']
    if part in ('development','test'):
        assert manifest.get('trained') and manifest.get('component_gates_passed')
    if (ROOT/'final-selection.json').exists():
        assert part=='test','No calibration, fitting data or development after final freeze'
    if part=='test':
        check_freeze(json.loads((ROOT/'final-selection.json').read_text()),manifest)
    if part=='holdout':
        check_freeze(json.loads((ROOT/'component-selection.json').read_text()),manifest)
        part='perception_holdout'
    started={json.loads(p.read_text())['name'] for p in Path('/workspace/outputs/fountain').glob('*/*'+PREFIX+'*/parameters.json')}
    return [s for s in plan[part] if s['name'] not in started]
