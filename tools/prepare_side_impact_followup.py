"""Idle-only declaration of the two narrow cohesion/iteration fixtures."""
import copy,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
from watch_side_cases import ROOT,capture,read
from watch_wide_collection import health,CID

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'final-selection.json').exists()
m=read(ROOT/'candidate.json');w=capture(m);h=health()
assert not h['queue'] and not h['recording']['active'] and not h.get('error')
progress=read(ROOT/'impact-r0/progress.json')
assert progress['status']=='complete' and progress['completed']==12
assert (ROOT/'impact-r0/runtime-after.json').exists()
report=read(ROOT/'impact-outcomes.json');assert len(report['cases'])==12 and report['baseline_reproduces_loss']
assert next(r for r in report['cases'] if r['name']=='SR_A_impact_02_water_cohesion')['zero_measured_loss']
plan=read(ROOT/'plan.json');assert 'impact_followup' not in plan
base=next(s for s in plan['impact'] if s['name']=='SR_A_impact_02_water_cohesion')
rows=[]
for i,iterations in enumerate((16,32)):
    s=copy.deepcopy(base);s.update(name=f'SR_A_impact_followup_{i:02}_cohesion{iterations}',study_split='impact_followup',iterations=iterations)
    rows.append(s)
plan.update(version=plan['version']+1,impact_followup=rows,
    impact_followup_rationale='Repeat cohesion=0.01 with 16 versus 32 iterations. The first comparison contains the drops with cohesion alone; iterations=32 alone prevents the late level drop but does not contain the drops. Combined behavior is untested until this follow-up.')
payload=dict(plan=plan,manifest=m,witness=w,report_sha256=sha(ROOT/'impact-outcomes.json'))
code='''
import ast,hashlib,json,shutil,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path('/workspace/learning/A-side-return-20261108-v1')
archive=root/'rounds/impact-r0';archive.mkdir(parents=True,exist_ok=False)
for n in ('candidate.json','plan.json','best.pt','recording-audit.json','impact-outcomes.json'):shutil.copy2(root/n,archive/n)
source=root/'staged/run_fountain.py';ast.parse(source.read_text())
(root/'plan.json').write_text(json.dumps(p['plan'],indent=2)+'\\n')
m=p['manifest'];m['plan_sha256']=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()
m['runtime_sources']['run_fountain.py']=hashlib.sha256(source.read_bytes()).hexdigest()
m['stage']='impact_followup_prepared'
(root/'candidate.json').write_text(json.dumps(m,indent=2)+'\\n')
(root/'impact-followup-preparation.json').write_text(json.dumps(p,indent=2)+'\\n')
print(json.dumps(dict(plan_sha256=m['plan_sha256'],checkpoint_sha256=m['checkpoint_sha256'])))
'''
subprocess.run(['docker','exec','-i',CID,'/isaac-sim/python.sh','-c',code],input=json.dumps(payload),text=True,check=True,timeout=60)
