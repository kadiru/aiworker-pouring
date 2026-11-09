"""Preserve r3 and declare one native-detail refinement without consuming tests."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, '/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-scale-20261109-v1'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f: json.dump(v,f,indent=2); f.write('\n')
def main():
    m=read(ROOT/'candidate.json'); plan=read(ROOT/'plan.json'); h=w.health(); witness=w.isolation()
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    assert not m['trained'] and sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(w.HOST_ROOT/'src'/n)==v for n,v in m['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT/Path(p).relative_to('/workspace'))==v for p,v in m['protected_parent_files'].items())
    started={read(p)['name'] for p in (w.HOST_ROOT/'outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in plan['holdout']}
    report=read(ROOT/'cup-neural-localization.json'); closure=read(ROOT/'locator-fit-r3/completion.json')
    assert not report['passed'] and closure['exit_code']==0 and closure['report_sha256']==sha(ROOT/'cup-neural-localization.json')
    archive=ROOT/'locator-rounds/r3'; archive.mkdir()
    names=('cup-neural-localization.json','cup-neural-localization-attempt.pt','cup-locator-data-manifest.json',
           'fit_scale_locator.py','run_scale_locator_fit.py','scale_data_admission.py','scale_locator_augmentation.py',
           'fit_scale_level.py','select_scale_static_components.py','evaluate_scale_static_holdout.py','run_scale_pair_pipeline.py','candidate.json')
    hashes={}
    for n in names:
        shutil.copy2(ROOT/n,archive/n); hashes[n]=sha(archive/n); assert hashes[n]==sha(ROOT/n)
    write(archive/'index.json',dict(files=hashes,runtime=witness,accepted=False))
    protocol=dict(round='r4',plan_sha256=m['plan_sha256'],audit_sha256=sha(ROOT/m['collection_audit']),
        review_sha256=sha(ROOT/m['collection_review']),coarse_checkpoint='locator-rounds/r3/cup-neural-localization-attempt.pt',
        coarse_checkpoint_sha256=hashes['cup-neural-localization-attempt.pt'],
        method='Frozen 384-pixel coarse CNN followed by a learned native-resolution 256-pixel rim crop heatmap. RGB only at inference; no truth-centered inference crops.',
        crop_size=256,epochs=160,learning_rate=.0003,seed=26110984,translation_pixels=24,
        max_correction_pixels=32.,native_resolution=1024,validation_interval=10,
        unchanged=['108 episode identities and splits','all accuracy thresholds','nine unused holdouts','active robot and runtime sources'],
        selection='Best validation maximum among candidates passing both split maxima, absence margin and twelve-frame temporal confirmation; CPU/GPU parity required.',
        scope='Known-size static perception only; no autonomous model activation.',runtime_before=witness)
    write(ROOT/'locator-refit-r4-plan.json',protocol)
    for n in names[:3]:
        assert sha(ROOT/n)==hashes[n]; (ROOT/n).unlink()
    print(json.dumps(dict(prepared=True,protocol_sha256=sha(ROOT/'locator-refit-r4-plan.json'),archived_r3=hashes)))
if __name__=='__main__':main()
