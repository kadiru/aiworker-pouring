"""Separate size curriculum; completed FN30 outcomes are immutable."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/learning/A-size-20261108-v1')
PREFIX = 'SZ_A_'
OUTPUTS = Path('/workspace/outputs/fountain')


def pending(part):
    plan = json.loads((ROOT/'plan.json').read_text())
    manifest = json.loads((ROOT/'candidate.json').read_text())
    assert hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest() == manifest['checkpoint_sha256']
    for name, digest in manifest['runtime_sources'].items():
        assert hashlib.sha256((Path('/workspace/src')/name).read_bytes()).hexdigest() == digest, name
    started = {json.loads(p.read_text())['name'] for p in OUTPUTS.glob('*/*'+PREFIX+'*/parameters.json')}
    return [s for s in plan[part] if s['name'] not in started]
