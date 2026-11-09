"""Archive a reviewed development round and prepare support checks/new pours.

Never invoke during an active split or before its recording/video delivery is
finished. Does not deploy, queue, change learned weights, or loosen acceptance.
"""
import copy,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
from watch_side_cases import ROOT,capture,read
from watch_wide_collection import health,CID
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (ROOT/'final-selection.json').exists()
m=read(ROOT/'candidate.json');w=capture(m);h=health()
assert not h['queue'] and not h['recording']['active'] and not h.get('error')
assert m['candidate_round']==0
progress=read(ROOT/'development-r0/progress.json')
assert progress['status']=='complete' and progress['completed']==6 and (ROOT/'development-r0/runtime-after.json').exists()
plan=read(ROOT/'plan.json');outcomes=read(ROOT/'development-outcomes.json');review=read(ROOT/'development-manual-review.json')
expected={s['name'] for s in plan['development']}
assert outcomes['status']=='complete' and {r['name'] for r in outcomes['cases']}==expected
assert review['manual_review']=='inspected' and {r['name'] for r in review['cases']}==expected
assert review['checkpoint_sha256']==m['checkpoint_sha256']
for row in review['cases']:
    assert row['manual_review']=='inspected'
    for p,d in row['reviewed_files'].items():assert sha(Path(p.replace('/workspace','/home/kadir/aiworker-pouring-4070',1)))==d
audit=read(ROOT/'recording-audit.json');assert not audit['started_incomplete']
checks=read(ROOT/'support-cpu-checks.json');assert checks['passed'] and checks['source_sha256']==sha(ROOT/'staged/pour_rigid_support.py')
assert 'support' not in plan and 'development_history' not in plan
contact=dict(static=.4,dynamic=.3);support=[]
for i,config in enumerate((None,contact)):
    spec=copy.deepcopy(plan['development'][0]);spec.update(name=f'SR_A_support_{i:02}_'+('baseline' if config is None else 'base_friction'),
        study_split='support',policy_evaluation=False,calibration_only=True,initial_ml=37.,cup_initial_ml=5.,max_seconds=2.,
        calibration_bottle_velocity_m_s=[-.02,0.,0.],trajectory_stride=1,trajectory_start_s=0.,
        evidence_scope='Non-autonomous bottle-on-table slide at an authored 20 mm/s initial velocity, about twice the observed release drift speed. No policy, attachment, velocity clamp or state reset after initialization.')
    if config is not None:spec['bottle_support_friction']=config
    support.append(spec)
plan['development_history']=copy.deepcopy(plan['development'])
for i,s in enumerate(plan['development']):
    s.update(name=f'SR_A_development_r1_{i:02}',development_round=1,bottle_support_friction=contact)
for s in plan['test']:s['bottle_support_friction']=contact
plan.update(version=plan['version']+1,support=support,bottle_support_friction=contact,
    repair_scope='New development attempts at the same six declared layouts after the retained r0 placement failures; all thirty test realizations remain unused. Acceptance thresholds unchanged.')
started={read(p)['name'] for p in (ROOT.parents[1]/'outputs/fountain').glob('*/*SR_A_*/parameters.json')}
assert not started&{s['name'] for s in plan['development']+plan['test']+support}
files=['run_fountain.py','robot_pour.py','pour_rigid_support.py','pour_side_study.py']
code='''
import ast,hashlib,json,shutil,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path('/workspace/learning/A-side-return-20261108-v1')
archive=root/'rounds/development-r0';archive.mkdir(parents=True,exist_ok=False);(archive/'src').mkdir()
for n in ('best.pt','candidate.json','candidate-selection-r0.json','plan.json','recording-audit.json','development-outcomes.json','development-visual-review.json','development-manual-review.json'):
    shutil.copy2(root/n,archive/n)
for n in p['manifest']['runtime_sources']:shutil.copy2(Path('/workspace/src')/n,archive/'src'/n)
for n in p['files']:ast.parse((root/'staged'/n).read_text())
(root/'plan.json').write_text(json.dumps(p['plan'],indent=2)+'\\n')
m=p['manifest'];m['plan_sha256']=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()
for n in p['files']:m['runtime_sources'][n]=hashlib.sha256((root/'staged'/n).read_bytes()).hexdigest()
m.update(stage='support_comparison_prepared',candidate_round=1,bottle_support_friction=p['contact'])
(root/'candidate.json').write_text(json.dumps(m,indent=2)+'\\n')
(root/'support-preparation.json').write_text(json.dumps(p,indent=2)+'\\n')
print(json.dumps(dict(plan_sha256=m['plan_sha256'],checkpoint_sha256=m['checkpoint_sha256'],stage=m['stage'])))
'''
subprocess.run(['docker','exec','-i',CID,'/isaac-sim/python.sh','-c',code],input=json.dumps(dict(plan=plan,manifest=m,witness=w,contact=contact,files=files)),text=True,check=True,timeout=60)
