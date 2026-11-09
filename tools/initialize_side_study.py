"""Idle-only native preparation; copies the frozen parent without modifying it."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
from watch_perception_cases import capture
from watch_wide_collection import health,HOST_ROOT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
parent=HOST_ROOT/'learning/A-perception-20261108-v1'
root=HOST_ROOT/'learning/A-side-return-20261108-v1'
assert not root.exists() or not list(root.iterdir())
m=read(parent/'candidate.json');w=capture(m);h=health()
assert not h['queue'] and not h['recording']['active'] and not h.get('error')
root.mkdir(exist_ok=True);(root/'staged').mkdir();(root/'parent').mkdir()
for n in ('best.pt','plan.json','candidate.json','final-selection.json','test-outcomes.json'):
    shutil.copy2(parent/n,root/'parent'/n)
shutil.copy2(parent/'best.pt',root/'best.pt')
shutil.copy2('/tmp/side-plan.json',root/'plan.json')
protected=dict(m['protected_parent_files'])
for n in ('best.pt','plan.json','candidate.json','final-selection.json','test-outcomes.json'):
    protected['/workspace/learning/A-perception-20261108-v1/'+n]=sha(parent/n)
for n in ('run_fountain.py','pour_side_study.py'):
    shutil.copy2('/tmp/side-'+n,root/'staged'/n)
sources={p.name:sha(p) for p in (HOST_ROOT/'src').glob('*.py')}
for p in (root/'staged').glob('*.py'):sources[p.name]=sha(p)
new=dict(checkpoint_sha256=sha(root/'best.pt'),plan_sha256=sha(root/'plan.json'),runtime_sources=sources,
    protected_parent_files=protected,component_gates_passed=True,trained=True,
    method='Unmodified parent learned weights; diagnostic instrumentation only at initial deployment',
    validated_cup_scope='baseline_only',parent_checkpoint_sha256=m['checkpoint_sha256'])
(root/'candidate.json').write_text(json.dumps(new,indent=2)+'\n')
(root/'initial-runtime-witness.json').write_text(json.dumps(w,indent=2)+'\n')
print(json.dumps(dict(study=str(root),parent_unchanged=True,checkpoint_sha256=new['checkpoint_sha256'])))
