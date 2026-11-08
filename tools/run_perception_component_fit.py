"""One bounded component fit, inside the idle owned 4070 container only."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-perception-20261108-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--component',choices=('locator','correction','cup','settled'),required=True);p.add_argument('--round',required=True);p.add_argument('--refine-from');p.add_argument('--cnn-from');p.add_argument('--member',type=int,choices=(0,1));a=p.parse_args()
    assert a.round.replace('-','').isalnum()
    out=Path('/tmp')/('aiworker-perception-'+a.component+'-'+a.round);out.mkdir(exist_ok=False)
    manifest=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');audit=read(ROOT/'recording-audit.json')
    assert not manifest['trained'] and not (ROOT/'component-selection.json').exists()
    names={s['name'] for k in ('collect','level_supplement','consumed_perception_training') for s in plan.get(k,[])}|set(plan.get('perception_training_pilots',[]))
    assert names<={r['name'] for r in audit['records']} and not audit['started_incomplete']
    assert all(r.get('label',{}).get('accepted') for r in audit['records'] if r['name'] in names)
    tasks={s['name'] for s in plan.get('consumed_task_development',[])}
    assert tasks<={r['name'] for r in audit['records']}
    assert sha(ROOT/'plan.json')==manifest['plan_sha256'] and sha(ROOT/'best.pt')==manifest['checkpoint_sha256']
    assert all(sha(w.HOST_ROOT/'src'/n)==d for n,d in manifest['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT/Path(p).relative_to('/workspace'))==d for p,d in manifest['protected_parent_files'].items())
    witness=w.isolation();h=w.health();assert not h['queue'] and not h.get('recording',{}).get('active',False) and not h.get('error')
    own={int(x.strip()) for x in subprocess.check_output(['docker','top',w.CID,'-eo','pid'],text=True).splitlines()[1:]}
    assert all(int(r.split(',')[1]) in own for r in witness['gpu_processes'].splitlines() if r.startswith(w.GPU))
    report={'locator':'cup-neural-localization.json','correction':'cup-locator-correction.json','cup':'cup-component.json','settled':'cup-component.json'}[a.component]
    assert not (ROOT/report).exists(),'Archive previous reports before starting a new fit'
    helper={'locator':'fit_perception_locator.py','correction':'fit_perception_locator_correction.py','cup':'fit_perception_cup.py','settled':'fit_perception_settled.py'}[a.component]
    extra=[]
    if a.refine_from:
        assert a.component=='locator'
        initial=(ROOT/a.refine_from).resolve();assert initial.is_relative_to(ROOT) and initial.is_file() and initial.suffix=='.pt'
        extra=['--refine-from',a.refine_from]
    if a.member is not None:
        assert a.component=='locator' and a.refine_from;extra+=['--member',str(a.member)]
    if a.cnn_from:
        assert a.component=='cup' and not a.refine_from
        initial=(ROOT/a.cnn_from).resolve();assert initial.is_relative_to(ROOT) and initial.is_file() and initial.suffix=='.pt'
        extra=['--cnn-from',a.cnn_from]
    identity=sha(ROOT/helper)
    (out/'runtime-before.json').write_text(json.dumps(dict(witness,helper_sha256=identity),indent=2)+'\n')
    with (out/'fit.log').open('x') as log:
        result=subprocess.run(['docker','exec',w.CID,'nice','-n','10','/isaac-sim/python.sh','/workspace/learning/A-perception-20261108-v1/'+helper]+extra,stdout=log,stderr=subprocess.STDOUT)
    (out/'runtime-after.json').write_text(json.dumps(w.isolation(),indent=2)+'\n')
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert sha(ROOT/helper)==identity
    completion=dict(exit_code=result.returncode,helper_sha256=identity,report=read(ROOT/report) if (ROOT/report).exists() else None)
    (out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(dict(exit_code=result.returncode,passed=completion['report'].get('passed') if completion['report'] else None,output=str(out))),flush=True)
    assert result.returncode==0
if __name__=='__main__':main()
