"""Finite guarded host runner for a selected perception or autonomous split."""
import argparse,datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-perception-20261108-v1'
sys.path.insert(0,str(w.HOST_ROOT/'src'))
from pour_robust_report import completed_row
from pour_wide_report import strict_success,reasons
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def capture(manifest):
    witness=w.isolation()
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert all(sha(w.HOST_ROOT/'src'/n)==d for n,d in manifest['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT/Path(p).relative_to('/workspace'))==d for p,d in manifest['protected_parent_files'].items())
    return dict(witness,checkpoint_sha256=manifest['checkpoint_sha256'],plan_sha256=manifest['plan_sha256'],runtime_sources=manifest['runtime_sources'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=('holdout','development','test'),required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();assert not a.output.exists()
    manifest=read(ROOT/'candidate.json');selection=read(ROOT/('final-selection.json' if a.part=='test' else 'component-selection.json'))
    assert manifest['trained'] and manifest['component_gates_passed']
    for key in ('checkpoint_sha256','plan_sha256','runtime_sources'):assert selection[key]==manifest[key]
    if a.part=='development':
        heldout=read(ROOT/'perception-holdout.json');assert heldout['passed'] and heldout['checkpoint_sha256']==manifest['checkpoint_sha256']
    plan=read(ROOT/'plan.json');expected={s['name']:s for s in plan['perception_holdout' if a.part=='holdout' else a.part]}
    assert len(expected)==(9 if a.part=='holdout' else 6)
    if a.part=='holdout':
        geometry=read(ROOT/'baseline-holdout-geometry.json')
        assert geometry['complete'] and geometry['passed'] and {r['name'] for r in geometry['cases']}==expected.keys()
    outputs=w.HOST_ROOT/'outputs/fountain'
    existing={read(p)['name'] for p in outputs.glob('*/*PV_A_*/parameters.json')}
    assert not expected.keys()&existing,'Never retry or silently resume a started split'
    h=w.health();assert not h['queue'] and not h.get('recording',{}).get('active',False) and not h.get('error')
    a.output.mkdir();(a.output/'runtime-before.json').write_text(json.dumps(capture(manifest),indent=2)+'\n')
    w.command('perception_'+a.part);seen={};deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        assert subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',w.CID],text=True).strip()=='true','Priority stop respected; no auto-restart'
        assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
        assert all(sha(w.HOST_ROOT/'src'/n)==d for n,d in manifest['runtime_sources'].items())
        for p in outputs.glob('*/*PV_A_*/metrics.json'):
            spec=read(p.parent/'parameters.json');name=spec['name']
            if name not in expected or name in seen:continue
            assert all(spec[k]==v for k,v in expected[name].items())
            if a.part=='holdout':row=dict(name=name,status='complete',path=str(p.parent))
            else:
                row=completed_row(p.parent,manifest['checkpoint_sha256'])
                row.update(strict_pass=strict_success(row),failure_reasons=reasons(row))
                assert row['strict_pass']==(not row['failure_reasons'])
            seen[name]=row
            print(json.dumps({k:v for k,v in row.items() if k in ('name','strict_pass','failure_reasons','stop_reason','cup_surface_height_fraction','source_remaining_ml','final_outside_ml')}),flush=True)
        h=w.health();assert not h.get('error') and not h.get('learning_error')
        progress=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),part=a.part,
            status='complete' if seen.keys()==expected.keys() else 'running',planned=len(expected),completed=len(seen),
            strict_passes=None if a.part=='holdout' else sum(r['strict_pass'] for r in seen.values()),
            cases=list(seen.values()),health={k:h.get(k) for k in ('phase','sim_time','recording','error')})
        (a.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        if seen.keys()==expected.keys() and not h['queue'] and not h['recording']['active']:break
        time.sleep(20)
    else:raise TimeoutError('Finite observer expired; recordings retained; no automatic restart')
    (a.output/'runtime-after.json').write_text(json.dumps(capture(manifest),indent=2)+'\n')
    subprocess.run(['docker','exec',w.CID,'/isaac-sim/python.sh','/workspace/learning/A-perception-20261108-v1/check_perception_stage.py'],check=True)
    print('SPLIT_COMPLETE; no later split or fitting started.',flush=True)
if __name__=='__main__':main()
