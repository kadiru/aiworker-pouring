"""Freeze six reviewed accepted development pours; never queue or tune tests."""
import datetime,hashlib,json,shutil,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (ROOT/'final-selection.json').exists()
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    m=read(ROOT/'candidate.json');assert m['component_gates_passed']
    assert sha(ROOT/'best.pt')==m['checkpoint_sha256'] and sha(ROOT/'plan.json')==m['plan_sha256']
    assert all(sha(Path('/workspace/src')/n)==d for n,d in m['runtime_sources'].items())
    assert all(sha(Path(p))==d for p,d in m['protected_parent_files'].items())
    plan=read(ROOT/'plan.json');development=read(ROOT/'development-outcomes.json')
    audit=read(ROOT/'recording-audit.json');review=read(ROOT/'development-manual-review.json')
    assert read(ROOT/'support-outcomes.json')['passed']
    assert not audit['started_incomplete'] and development['status']=='complete'
    expected={s['name'] for s in plan['development']};assert len(expected)==6
    assert {r['name'] for r in development['cases']}==expected
    assert development['counts']['completed']==development['counts']['strict_passes']==development['counts']['accepted']==6
    assert all(r['accepted'] and r['strict_pass'] and r['withdrawal']['passed'] and r['side_return']['passed'] and r['stopping_margin_passed'] for r in development['cases'])
    for key in ('checkpoint_sha256','plan_sha256','runtime_sources'):assert development[key]==m[key]
    assert review['checkpoint_sha256']==m['checkpoint_sha256'] and review['manual_review']=='inspected' and review['accepted']
    assert {r['name'] for r in review['cases']}==expected
    indexed={r['name']:r for r in audit['records']}
    for r in review['cases']:
        assert r['source_sha256']==indexed[r['name']]['sha256'] and r['manual_review']=='inspected'
        assert len(r['frames'])==12 and len(r['return_frames'])==12 and r['reviewed_files']
        for path,digest in r['reviewed_files'].items():assert sha(Path(path))==digest
    reserved={s['name'] for s in plan['test']};assert len(reserved)==30
    assert sum(s['expected_stop']=='cup_near_full' for s in plan['test'])==15
    started={read(p)['name'] for p in Path('/workspace/outputs/fountain').glob('*/*SR_A_*/parameters.json')}
    assert reserved.isdisjoint(started)
    archive=ROOT/'frozen-candidate';archive.mkdir();(archive/'src').mkdir()
    evidence=('development-outcomes.json','development-manual-review.json','recording-audit.json','impact-outcomes.json','impact_followup-outcomes.json','support-outcomes.json')
    for n in ('best.pt','candidate.json','plan.json',*evidence):shutil.copy2(ROOT/n,archive/n)
    for n in m['runtime_sources']:shutil.copy2(Path('/workspace/src')/n,archive/'src'/n)
    selection=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **{k:m[k] for k in ('checkpoint_sha256','plan_sha256','runtime_sources')},
        development_passes=6,development_acceptances=6,
        development_outcomes_sha256=sha(ROOT/'development-outcomes.json'),
        development_manual_review_sha256=sha(ROOT/'development-manual-review.json'),
        evidence={n:sha(ROOT/n) for n in evidence},helper_sha256=sha(Path(__file__)),
        scope='Thirty fresh once-only realizations of existing layout families, balanced 15 full / 15 empty, after six reviewed accepted development pours. Baseline vessel sizes. No fitting, retries, threshold/physics/runtime changes or teacher fallback during reserved evaluation. Historical failures are preserved.')
    payload=json.dumps(selection,indent=2)+'\n';(archive/'final-selection.json').write_text(payload);(ROOT/'final-selection.json').write_text(payload)
    print(json.dumps(dict(frozen=True,checkpoint_sha256=m['checkpoint_sha256'],reserved_unstarted=len(reserved))))
if __name__=='__main__':main()
