"""Read-only closure of an unaccepted development study; no job control."""
import hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-scale-dynamic-20261109-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    m=read(ROOT/'candidate.json');h=w.health();witness=w.isolation()
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    assert not m['autonomy_accepted'] and not (ROOT/'component-selection.json').exists() and not (ROOT/'autonomy-selection.json').exists()
    assert sha(ROOT/'best.pt')==m['checkpoint_sha256']
    protected={n:sha(w.HOST_ROOT/Path(n).relative_to('/workspace'))==value for n,value in m['protected_parent_files'].items()}
    sources={n:sha(w.HOST_ROOT/'src'/n)==value for n,value in m['runtime_sources'].items()}
    assert all(protected.values()) and all(sources.values())
    reserved=list((w.HOST_ROOT/'outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'));assert not reserved
    jobs={}
    for p in sorted(ROOT.glob('*/completion.json')):
        d=read(p);item=dict(completion=d,completion_sha256=sha(p))
        for name in ('runtime-before.json','run.log'):
            q=p.parent/name
            if q.exists():item[name+'_sha256']=sha(q)
        jobs[p.parent.name]=item
    processes=subprocess.check_output(['docker','top',w.CID,'-eo','pid,args'],text=True)
    assert not any(('fit_dynamic_' in line or 'probe_dynamic_rim_' in line) and '/isaac-sim' in line for line in processes.splitlines())
    cases=[]
    for part in ('seed_pilot','seed_remaining','low_train','train','validation'):
        audit=read(ROOT/(part+'-r0-audit.json'));review=read(ROOT/(part+'-r0-manual-review.json'))
        assert audit['complete'] and review['accepted'] and review['audit_sha256']==sha(ROOT/(part+'-r0-audit.json'))
        cases.extend(r['name'] for r in audit['records'])
    assert len(cases)==len(set(cases))==63
    reports={n:dict(sha256=sha(ROOT/n),passed=read(ROOT/n).get('passed',read(ROOT/n).get('passed_for_refinement'))) for n in ('coarse-d1.json','coarse-d2.json','rim-d1.json','rim-d2.json','rim-reuse-probe.json','rim-edge-probe.json','pair-r3-aborted.json')}
    report=dict(status='development_accuracy_failed_no_autonomy',runtime=witness,health={k:h.get(k) for k in ('run','phase','queue','recording','error','learning_error')},active_checkpoint_sha256=m['checkpoint_sha256'],protected_parent_files_verified=len(protected),runtime_sources_verified=len(sources),new_calibration_scenes=63,reserved_scenes_started=0,autonomy_accepted=False,no_successor_models_activated=True,component_jobs_running=False,jobs=jobs,reports=reports,helper_sha256=sha(Path(__file__)),scope='Closure of this development pass, not successful completion of the requested perception repair. No foreign job signaled or reconfigured; point-in-time resource witnesses do not establish uninterrupted execution of other workloads.')
    with (ROOT/'development-closure.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('jobs','reports','health','runtime')}),flush=True)
if __name__=='__main__':main()
