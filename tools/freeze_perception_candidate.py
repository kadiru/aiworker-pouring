"""Freeze a reviewed six-pass development candidate; never queue or tune tests."""
import argparse,datetime,hashlib,json,shutil,sys,urllib.request
from pathlib import Path
sys.path.insert(0,'/workspace/src')
from pour_perception_study import check_freeze
ROOT=Path('/workspace/learning/A-perception-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);a=p.parse_args()
    assert not (ROOT/'final-selection.json').exists()
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    manifest=read(ROOT/'candidate.json');check_freeze(read(ROOT/'component-selection.json'),manifest)
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'] and sha(ROOT/'plan.json')==manifest['plan_sha256']
    assert all(sha(Path('/workspace/src')/n)==d for n,d in manifest['runtime_sources'].items())
    assert all(sha(Path(p))==d for p,d in manifest['protected_parent_files'].items())
    plan=read(ROOT/'plan.json');development=read(ROOT/'development-outcomes.json');holdout=read(ROOT/'perception-holdout.json')
    audit=read(ROOT/'recording-audit.json');review=read(a.review)
    assert not audit['started_incomplete'] and development['status']=='complete'
    expected={s['name'] for s in plan['development']};assert len(expected)==6
    assert {r['name'] for r in development['cases']}==expected
    assert all(r['strict_pass'] and r['withdrawal']['passed'] for r in development['cases'])
    assert holdout['passed'] and holdout['checkpoint_sha256']==development['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert development['runtime_sources']==manifest['runtime_sources']
    assert review['checkpoint_sha256']==manifest['checkpoint_sha256'] and review['manual_review']=='inspected'
    assert {r['name'] for r in review['cases']}==expected
    indexed={r['name']:r for r in audit['records']}
    for r in review['cases']:
        assert r['source_sha256']==indexed[r['name']]['sha256'] and r['manual_review']=='inspected'
        assert r['reviewed_files']
        for path,digest in r['reviewed_files'].items():assert sha(Path(path))==digest
    reserved={s['name'] for s in plan['test']}
    started={read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*PV_A_*/parameters.json')}
    assert reserved.isdisjoint(started)
    archive=ROOT/'frozen-candidate';archive.mkdir();(archive/'src').mkdir()
    for n in ('best.pt','candidate.json','plan.json','component-selection.json','perception-holdout.json','development-outcomes.json','recording-audit.json'):
        shutil.copy2(ROOT/n,archive/n)
    for n in manifest['runtime_sources']:shutil.copy2(Path('/workspace/src')/n,archive/'src'/n)
    selection=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **{k:manifest[k] for k in ('checkpoint_sha256','plan_sha256','runtime_sources')},
        evidence={n:sha(ROOT/n) for n in ('component-selection.json','perception-holdout.json','development-outcomes.json','recording-audit.json')},
        visual_review_sha256=sha(a.review),helper_sha256=sha(Path(__file__)),
        scope='Six once-only fresh autonomous realizations of prior layout families, after six development passes and nine independent static perception checks. Baseline vessel sizes only; no fitting/retries or runtime changes during reserved evaluation. Original benchmark scores remain unchanged.')
    payload=json.dumps(selection,indent=2)+'\n';(archive/'final-selection.json').write_text(payload);(ROOT/'final-selection.json').write_text(payload)
    print(json.dumps(dict(frozen=True,checkpoint_sha256=manifest['checkpoint_sha256'],reserved_unstarted=len(reserved))))
if __name__=='__main__':main()
