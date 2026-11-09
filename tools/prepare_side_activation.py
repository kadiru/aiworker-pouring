"""Idle-only archive and manifest preparation before an owned runtime restart."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
from watch_side_cases import ROOT,capture,read
from watch_wide_collection import health,CID,HOST_ROOT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--phase',choices=('impact',),required=True);a=p.parse_args()
assert not (ROOT/'final-selection.json').exists()
m=read(ROOT/'candidate.json');w=capture(m);h=health();assert not h['queue'] and not h['recording']['active'] and not h.get('error')
progress=read(ROOT/'diagnostic-r0/progress.json');assert progress['status']=='complete' and progress['completed']==4
assert (ROOT/'diagnostic-r0/runtime-after.json').exists()
audit=read(ROOT/'recording-audit.json');assert len(audit['records'])==4 and not audit['started_incomplete']
files=['run_fountain.py','robot_pour.py','pour_side_study.py','pour_side_return.py','pour_side_metrics.py','pour_guard_actor.py']
plan=read(Path('/tmp/side-balanced-plan.json'));impact=read(ROOT/'impact-plan-v2.json');assert len(impact['impact'])==12
assert plan['diagnostic']==read(ROOT/'plan.json')['diagnostic']
plan.update(version=3,impact=impact['impact'],impact_seed_sha256=impact['seed_sha256'])
for s in plan['development']+plan['test']:s['development_round']=0
code='''
import ast,hashlib,json,shutil,sys
from pathlib import Path
root=Path('/workspace/learning/A-side-return-20261108-v1');p=json.load(sys.stdin)
archive=root/'rounds/instrumentation';archive.mkdir(parents=True,exist_ok=False)
for n in ('candidate.json','plan.json','best.pt','recording-audit.json'):shutil.copy2(root/n,archive/n)
for n in p['files']:ast.parse((root/'staged'/n).read_text())
(root/'plan.json').write_text(json.dumps(p['plan'],indent=2)+'\\n')
m=p['manifest'];m['plan_sha256']=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()
for n in p['files']:m['runtime_sources'][n]=hashlib.sha256((root/'staged'/n).read_bytes()).hexdigest()
m.update(stage='impact_comparison_prepared',method='Unmodified learned parent weights; explicitly non-autonomous fluid impact comparison; side controller unselected')
(root/'candidate.json').write_text(json.dumps(m,indent=2)+'\\n')
(root/'impact-preparation-witness.json').write_text(json.dumps(p['witness'],indent=2)+'\\n')
print(json.dumps(dict(plan_sha256=m['plan_sha256'],checkpoint_sha256=m['checkpoint_sha256'],files=p['files'])))
'''
subprocess.run(['docker','exec','-i',CID,'/isaac-sim/python.sh','-c',code],input=json.dumps(dict(files=files,plan=plan,manifest=m,witness=w)),text=True,check=True,timeout=60)
print('Prepared only. The exact owned deployment helper must activate these files before any queue command.')
