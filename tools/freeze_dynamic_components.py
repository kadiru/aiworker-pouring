"""Freeze a passing development pair before any reserved moving collection."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    assert not (ROOT/'component-selection.json').exists() and not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    m=read(ROOT/'candidate.json');p=read(ROOT/'protocol.json');pair=read(ROOT/'pair-r4.json');cfg=read(ROOT/'pair-r4-protocol.json');assert pair['passed'] and not m['autonomy_accepted']
    assert pair['protocol_sha256']==sha(ROOT/'pair-r4-protocol.json') and cfg['protocol_sha256']==sha(ROOT/'protocol.json')
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items())
    assert all(sha(Path(n))==v for n,v in m['protected_parent_files'].items()) and all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    locator=cfg['locator'];assert read(ROOT/locator.replace('.pt','.json'))['passed']
    for kind in ('rgb','color'):
        report=read(ROOT/('pair-r4-'+kind+'.json'));assert report['passed'] and report['checkpoint_sha256']==sha(ROOT/('pair-r4-'+kind+'.pt'))
    assert read(ROOT/'dynamic-localization-check.json')['passed'] and read(ROOT/'level-localization-check.json')['passed']
    files=list(cfg['frozen_files'])+['pair-r4.json','pair-r4-protocol.json','pair-r4-rgb.pt','pair-r4-color.pt','pair-r4-rgb.json','pair-r4-color.json','dynamic-localization-check.json','level-localization-check.json','evaluate_dynamic_holdout.py']
    result=dict(all_required_development_gates_passed=True,protocol_sha256=sha(ROOT/'protocol.json'),locator=locator,rgb='pair-r4-rgb.pt',color='pair-r4-color.pt',holdout_names=[s['name'] for s in p['holdout']],evaluation_helper_sha256=sha(ROOT/'evaluate_dynamic_holdout.py'),frozen_files={n:sha(ROOT/n) for n in sorted(set(files))},autonomy_accepted=False,scope='Permits only once-only reserved perception fixtures. Autonomous task use needs separate runtime, motion and task acceptance.')
    with (ROOT/'component-selection.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(frozen=True,reserved_scenes=len(p['holdout']),autonomy_accepted=False)))
if __name__=='__main__':main()
