"""Activate the prepared development round after measured support checks."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
assert not (ROOT/'final-selection.json').exists() and not (ROOT/'candidate-selection-r1.json').exists()
h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
assert not h['queue'] and not h['recording']['active'] and not h.get('error')
m=read(ROOT/'candidate.json');assert m['stage']=='support_comparison_prepared' and m['candidate_round']==1
assert sha(ROOT/'best.pt')==m['checkpoint_sha256'] and sha(ROOT/'plan.json')==m['plan_sha256']
assert all(sha(Path('/workspace/src')/n)==d for n,d in m['runtime_sources'].items())
assert all(sha(Path(p))==d for p,d in m['protected_parent_files'].items())
report=read(ROOT/'support-outcomes.json');assert report['passed'] and len(report['cases'])==2
audit=read(ROOT/'recording-audit.json');assert not audit['started_incomplete'] and sha(ROOT/'recording-audit.json')==report['audit_sha256']
progress=read(ROOT/'support-r0/progress.json');assert progress['status']=='complete' and progress['completed']==2 and (ROOT/'support-r0/runtime-after.json').exists()
plan=read(ROOT/'plan.json');started={read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*SR_A_*/parameters.json')}
assert not started&{s['name'] for s in plan['development']+plan['test']}
assert sha(ROOT/'rounds/development-r0/best.pt')==m['checkpoint_sha256']
m.update(stage='development_candidate',support_outcomes_sha256=sha(ROOT/'support-outcomes.json'),
    method='Retained learned models, stop threshold, fluid coefficients and side trajectory from r0. Opt-in bottle-bottom rigid support friction; new development required. No new fit.')
write(ROOT/'candidate.json',m)
write(ROOT/'candidate-selection-r1.json',dict(candidate=m,checkpoint_unchanged=True,
    parent_candidate_sha256=sha(ROOT/'rounds/development-r0/candidate.json'),support_result=report,helper_sha256=sha(Path(__file__))))
print(json.dumps(dict(stage=m['stage'],checkpoint_sha256=m['checkpoint_sha256'],plan_sha256=m['plan_sha256'])))
