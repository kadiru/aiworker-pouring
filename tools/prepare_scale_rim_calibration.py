"""Archive r5; declare bounded train-only pixel calibration of retained r4."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-scale-20261109-v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
    h=w.health();witness=w.isolation();m=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json')
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    assert not m['trained'] and sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(w.HOST_ROOT/'src'/n)==v for n,v in m['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT/Path(p).relative_to('/workspace'))==v for p,v in m['protected_parent_files'].items())
    started={read(p)['name'] for p in (w.HOST_ROOT/'outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in plan['holdout']}
    report=read(ROOT/'cup-neural-localization.json');completion=read(ROOT/'locator-fit-r5/completion.json')
    assert not report['passed'] and completion['exit_code']==0 and completion['report_sha256']==sha(ROOT/'cup-neural-localization.json')
    archive=ROOT/'locator-rounds/r5';archive.mkdir()
    names=('cup-neural-localization.json','cup-neural-localization-attempt.pt','cup-locator-data-manifest.json',
        'fit_scale_rim_refiner.py','scale_rim_refiner.py','scale_rim_augmentation.py','run_scale_locator_fit.py','run_scale_rim_pipeline.py',
        'candidate.json','locator-refit-r5-plan.json')
    hashes={}
    for n in names:
        shutil.copy2(ROOT/n,archive/n);hashes[n]=sha(archive/n);assert hashes[n]==sha(ROOT/n)
    write(archive/'index.json',dict(files=hashes,runtime=witness,accepted=False))
    base=ROOT/'locator-rounds/r4/cup-neural-localization-attempt.pt'
    protocol=dict(round='r6',plan_sha256=m['plan_sha256'],audit_sha256=sha(ROOT/m['collection_audit']),
        review_sha256=sha(ROOT/m['collection_review']),parent='locator-rounds/r4/cup-neural-localization-attempt.pt',parent_sha256=sha(base),
        ridge_slope_penalty=.1,ridge_intercept_penalty=1e-8,max_affine_correction_pixels=2.,
        fitting='Six coefficients in delta=[1,(u-512)/512,(v-512)/512] @ B. One mean per training episode, equal episode weighting; CUDA float64 ridge solve.',
        unchanged=['108 episode identities and splits','all accuracy thresholds','nine unused holdouts','active robot and runtime sources'],
        required=['both split frame maxima under 3 mm','absence margin','all twelve-frame temporal confirmations','end-to-end inference parity','subsequent level gates before reserved tests'],
        scope='Known-size static perception only; no autonomous activation.',runtime_before=witness)
    write(ROOT/'locator-refit-r6-plan.json',protocol)
    for n in names[:3]:assert sha(ROOT/n)==hashes[n];(ROOT/n).unlink()
    print(json.dumps(dict(prepared=True,protocol_sha256=sha(ROOT/'locator-refit-r6-plan.json'),r5_archived=True)))
if __name__=='__main__':main()
