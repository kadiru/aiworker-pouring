"""Verify all frozen sources and every native backup before publishing counts."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/robust-A';DATA=ROOT/'data/material-A-robust'


def read(p):return json.loads(p.read_text())


def main():
    r=read(E/'robust-results.json');selection=read(E/'final-selection.json');before=read(E/'runtime-before-tests.json');after=read(E/'runtime-final.json')
    assert r['status']=='complete' and len(r['cases'])==30
    sha=hashlib.sha256((E/'best.pt').read_bytes()).hexdigest()
    assert sha==r['checkpoint_sha256']==selection['selected_checkpoint_sha256']==before['selected_checkpoint_sha256']==after['selected_checkpoint_sha256']
    assert hashlib.sha256((E/'frozen-plan.json').read_bytes()).hexdigest()==selection['scene_plan_sha256']==r['scene_plan_sha256']
    assert before['sources']==after['sources']==selection['runtime_sources']
    for n,h in selection['runtime_sources'].items():assert hashlib.sha256((ROOT/'runtime/src'/n).read_bytes()).hexdigest()==h,n
    assert before['container']==after['container'] and before['limits']==after['limits'] and after['guard_processes']
    assert not after['health']['queue'] and not after['health']['recording']['active'] and not after['health']['error'] and not after['health']['learning_error']
    assert before['asset_sha256']==after['asset_sha256']=='82ac39486428e51d8be58996ea817fd4b25a40c262936e2d2cc8823673e1387d'
    expected={x['name'] for x in read(E/'frozen-plan.json')['test']};assert {x['name'] for x in r['cases']}==expected
    assert all(x['checkpoint_sha256']==sha and not x['teacher_fallback'] for x in r['cases'])
    integrity=read(E/'recording-integrity.json');assert len(integrity['files'])==30 and len(integrity['sidecar_files'])==120
    for x in integrity['files']+integrity['sidecar_files']:
        p=DATA/x['relative_path'];assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],str(p)
    assert all(x['valid'] for x in integrity['files'])
    (DATA/'backup-index.json').write_text(json.dumps(dict(all_native_checksums_match=True,**integrity),indent=2)+'\n')
    (E/'final-benchmark.json').write_text(json.dumps(dict(**r,verified_native_backups=30,verified_sidecars=120),indent=2)+'\n')
    print(json.dumps(dict(status='complete',counts=r['counts'],native_backups=30,sidecars=120),indent=2))


if __name__=='__main__':main()
