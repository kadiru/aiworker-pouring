"""Verify local frozen-suite provenance and native backups; write concise results."""
import csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence/stop-A'


def read(p):return json.loads(p.read_text())


def main():
    result=read(E/'stop-results.json');model=read(E/'model-summary.json');selection=read(E/'final-selection.json')
    assert result['test_status']=='complete'
    sha=hashlib.sha256((E/'best.pt').read_bytes()).hexdigest()
    assert sha==model['checkpoint_sha256']==result['checkpoint_sha256']==selection['selected_checkpoint_sha256']
    assert hashlib.sha256((E/'frozen-plan.json').read_bytes()).hexdigest()==result['scene_plan_sha256']==model['scene_plan_sha256']
    for name,value in model['sources'].items():assert hashlib.sha256((ROOT/'runtime/src'/name).read_bytes()).hexdigest()==value,name
    tests=result['groups']['test'];rows=tests['rows'];expected={s['name'] for s in read(E/'frozen-plan.json')['test']}
    assert len(rows)==6 and {r['name'] for r in rows}==expected
    assert all(r['checkpoint_sha256']==sha and not r['teacher_fallback'] for r in rows)
    integrity=read(E/'recording-integrity.json');files=[]
    for r in integrity['files']:
        p=ROOT/'data/material-A-stop'/r['relative_path']
        assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
        files.append(dict(r,local_path=str(p)))
    sidecars=[]
    for r in integrity['sidecar_files']:
        p=ROOT/'data/material-A-stop'/r['relative_path']
        assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
        sidecars.append(dict(r,local_path=str(p)))
    assert len(sidecars)==4*len(files)
    index=dict(episodes=len(files),valid_recordings=integrity['valid_recordings'],all_native_checksums_match=True,sidecar_files=sidecars,
        partial_aborted_excluded=integrity['partial_aborted_excluded'],files=files)
    (ROOT/'data/material-A-stop/backup-index.json').write_text(json.dumps(index,indent=2)+'\n')
    summary=dict(status='complete',evidence_kind='native Isaac Sim autonomous learned/programmed hierarchy',
        checkpoint_sha256=sha,model=model,selection=selection,
        primary_goal=result['primary_definition'],legacy_goal=result['legacy_definition'],goal_calibration_note=result['goal_calibration_note'],
        volume_units=result['volume_units'],test={k:v for k,v in tests.items() if k!='rows'},rows=rows,
        verified_native_backups=len(files))
    (E/'final-benchmark.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (E/'final-benchmark.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    print(json.dumps(dict(status=summary['status'],test=summary['test'],verified_native_backups=len(files)),indent=2))


if __name__=='__main__':main()
