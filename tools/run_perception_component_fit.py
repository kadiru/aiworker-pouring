"""One bounded component fit, inside the idle owned 4070 container only."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-perception-20261108-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--component',choices=('locator','cup'),required=True);p.add_argument('--round',required=True);a=p.parse_args()
    assert a.round.replace('-','').isalnum()
    out=Path('/tmp')/('aiworker-perception-'+a.component+'-'+a.round);out.mkdir(exist_ok=False)
    manifest=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');audit=read(ROOT/'recording-audit.json')
    assert not manifest['trained'] and not (ROOT/'component-selection.json').exists()
    names={s['name'] for k in ('collect','level_supplement') for s in plan.get(k,[])}|set(plan.get('perception_training_pilots',[]))
    assert names<={r['name'] for r in audit['records']} and not audit['started_incomplete']
    assert all(r.get('label',{}).get('accepted') for r in audit['records'] if r['name'] in names)
    assert sha(ROOT/'plan.json')==manifest['plan_sha256'] and sha(ROOT/'best.pt')==manifest['checkpoint_sha256']
    assert all(sha(w.HOST_ROOT/'src'/n)==d for n,d in manifest['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT/Path(p).relative_to('/workspace'))==d for p,d in manifest['protected_parent_files'].items())
    witness=w.isolation();h=w.health();assert not h['queue'] and not h.get('recording',{}).get('active',False) and not h.get('error')
    own={int(x.strip()) for x in subprocess.check_output(['docker','top',w.CID,'-eo','pid'],text=True).splitlines()[1:]}
    assert all(int(r.split(',')[1]) in own for r in witness['gpu_processes'].splitlines() if r.startswith(w.GPU))
    report='cup-neural-localization.json' if a.component=='locator' else 'cup-component.json'
    assert not (ROOT/report).exists(),'Archive previous reports before starting a new fit'
    helper='fit_perception_locator.py' if a.component=='locator' else 'fit_perception_cup.py'
    identity=sha(ROOT/helper)
    (out/'runtime-before.json').write_text(json.dumps(dict(witness,helper_sha256=identity),indent=2)+'\n')
    with (out/'fit.log').open('x') as log:
        result=subprocess.run(['docker','exec',w.CID,'nice','-n','10','/isaac-sim/python.sh','/workspace/learning/A-perception-20261108-v1/'+helper],stdout=log,stderr=subprocess.STDOUT)
    (out/'runtime-after.json').write_text(json.dumps(w.isolation(),indent=2)+'\n')
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert sha(ROOT/helper)==identity
    completion=dict(exit_code=result.returncode,helper_sha256=identity,report=read(ROOT/report) if (ROOT/report).exists() else None)
    (out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps(dict(exit_code=result.returncode,passed=completion['report'].get('passed') if completion['report'] else None,output=str(out))),flush=True)
    assert result.returncode==0
if __name__=='__main__':main()
