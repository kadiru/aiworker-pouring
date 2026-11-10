"""New study IDs and explicit gates; never reuses completed static test IDs."""
import hashlib,json
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def pending(part):
    m=read(ROOT/'candidate.json');p=read(ROOT/'protocol.json')
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    assert all(sha(Path(n))==v for n,v in m['protected_parent_files'].items())
    if part in ('seed_pilot','seed_remaining'):
        pilot=set(p['seed_pilot_names']);cases=[s for s in p['seed'] if (s['name'] in pilot)==(part=='seed_pilot')]
        if part=='seed_remaining':assert read(ROOT/'seed-pilot-review.json')['accepted']
    elif part=='low_train':cases=p[part]
    else:
        assert m.get('high_seeds_accepted'), 'High-fill physical seeds require review first'
        cases=p[part]
        if part=='holdout':
            accepted=read(ROOT/'component-selection.json');assert accepted['all_required_development_gates_passed']
            assert accepted['protocol_sha256']==sha(ROOT/'protocol.json')
            assert all(sha(ROOT/n)==v for n,v in accepted['frozen_files'].items())
        if part=='autonomy':
            accepted=read(ROOT/'autonomy-selection.json');assert m.get('autonomy_accepted') and accepted['all_gates_passed']
            assert accepted['protocol_sha256']==sha(ROOT/'protocol.json')
            assert all(sha(ROOT/n)==v for n,v in accepted['frozen_files'].items())
    assert cases and len({s['name'] for s in cases})==len(cases)
    assert all(s['name'].startswith('SD_A_') for s in cases)
    assert all(s.get('calibration_only') and not s.get('policy_evaluation') for s in cases) if part!='autonomy' else all(s.get('policy_evaluation') for s in cases)
    geo=read(ROOT/(part+'-geometry.json'));assert geo['passed'] and geo['complete'] and geo['protocol_sha256']==sha(ROOT/'protocol.json')
    assert {s['name'] for s in cases}=={r['name'] for r in geo['cases']}
    started={read(q)['name'] for q in Path('/workspace/outputs/fountain').glob('*/*SD_A_*/parameters.json')}
    return [s for s in cases if s['name'] not in started]
