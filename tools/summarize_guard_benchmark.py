"""Verify the committed native-six report, fixed sources, and external backups."""
import csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/guard-A'


def read(p):return json.loads(p.read_text())


def main():
    r=read(E/'guard-results.json');m=read(E/'model-summary.json');selection=read(E/'final-selection.json')
    assert r['test_status']=='complete'
    assert r['checkpoint_sha256']==m['checkpoint_sha256']==selection['selected_checkpoint_sha256']
    assert hashlib.sha256((E/'best.pt').read_bytes()).hexdigest()==r['checkpoint_sha256']
    assert hashlib.sha256((E/'frozen-plan.json').read_bytes()).hexdigest()==r['scene_plan_sha256']==m['scene_plan_sha256']
    for n,h in m['sources'].items():assert hashlib.sha256((ROOT/'runtime/src'/n).read_bytes()).hexdigest()==h,n
    rows=r['groups']['test']['rows'];expected={s['name'] for s in read(E/'frozen-plan.json')['test']}
    assert len(rows)==6 and {x['name'] for x in rows}==expected
    assert all(x['checkpoint_sha256']==r['checkpoint_sha256'] and not x['teacher_fallback'] for x in rows)
    integrity=read(E/'recording-integrity.json');files=[];sidecars=[];partials=[]
    for entries,dest in ((integrity['files'],files),(integrity['sidecar_files'],sidecars)):
        for x in entries:
            p=ROOT/'data/material-A-guard'/x['relative_path']
            assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],str(p)
            dest.append(dict(x,local_path=str(p)))
    for x in integrity['partial_aborted_excluded']:
        assert x['status']=='aborted','An active recording remains'
        if 'relative_path' in x:
            p=ROOT/'data/material-A-guard'/x['relative_path']
            assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
            partials.append(dict(x,local_path=str(p)))
    assert len(sidecars)==4*len(files)
    index=dict(episodes=len(files),valid_recordings=integrity['valid_recordings'],all_native_checksums_match=True,
        files=files,sidecar_files=sidecars,partial_aborted_excluded=partials)
    (ROOT/'data/material-A-guard/backup-index.json').write_text(json.dumps(index,indent=2)+'\n')
    summary=dict(status='complete',evidence_kind='native Isaac Sim visual-servo and supervised-imitation autonomous evaluation',
        checkpoint_sha256=r['checkpoint_sha256'],model=m,selection=selection,primary_goal=r['primary_definition'],legacy_goal=r['legacy_definition'],
        volume_units=r['volume_units'],test={k:v for k,v in r['groups']['test'].items() if k!='rows'},rows=rows,
        verified_native_backups=len(files),aborted_development_retained=len(partials))
    (E/'final-benchmark.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (E/'final-benchmark.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in x.items()} for x in rows)
    print(json.dumps(dict(status='complete',test=summary['test'],verified_native_backups=len(files),aborted_development_retained=len(partials)),indent=2))


if __name__=='__main__':main()
