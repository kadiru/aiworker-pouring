"""Activate preflighted static replacement without changing robot runtime."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    assert not read(ROOT/'pair-r9.json')['passed']
    old=read(ROOT/'protocol.json');new=read(ROOT/'protocol-midfill-v6-draft.json');geo=read(ROOT/'protocol-midfill-v6-draft-train-geometry.json')
    assert old['version']==5 and new['version']==6 and new['previous_protocol_sha256']==sha(ROOT/'protocol.json')
    assert all(new[k]==old[k] for k in ('validation','holdout','gates'))
    assert geo['passed'] and geo['complete'] and geo['protocol_sha256']==sha(ROOT/'protocol-midfill-v6-draft.json')
    assert not (ROOT/'component-selection.json').exists() and not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    m=read(ROOT/'candidate.json');assert all(sha(Path(n))==v for n,v in m['protected_parent_files'].items()) and all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    shutil.copy2(ROOT/'protocol.json',ROOT/'protocol-v5-before-midfill.json');shutil.copy2(ROOT/'train-geometry.json',ROOT/'train-geometry-v5-before-midfill.json')
    (ROOT/'protocol.json').write_bytes((ROOT/'protocol-midfill-v6-draft.json').read_bytes());(ROOT/'train-geometry.json').write_bytes((ROOT/'protocol-midfill-v6-draft-train-geometry.json').read_bytes())
    report=dict(protocol_sha256=sha(ROOT/'protocol.json'),previous_protocol_sha256=sha(ROOT/'protocol-v5-before-midfill.json'),extra_static_training=36,tracker_sha256=sha(ROOT/'tracker-t2.pt'),runtime_sources_unchanged=True,validation_reserved_and_gates_unchanged=True,autonomy_accepted=False)
    with (ROOT/'midfill-activation.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
