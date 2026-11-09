"""Archive rejected level fit; freeze accepted localization and a new protocol."""
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
    m=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');h=w.health();runtime=w.isolation()
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    assert not m['trained'] and m['accepted_locator'] and not (ROOT/'component-selection.json').exists()
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(w.HOST_ROOT/'src'/n)==v for n,v in m['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT/Path(p).relative_to('/workspace'))==v for p,v in m['protected_parent_files'].items())
    started={read(p)['name'] for p in (w.HOST_ROOT/'outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in plan['holdout']}
    loc=read(ROOT/'cup-neural-localization.json');level=read(ROOT/'static-level-fit.json')
    assert loc['passed'] and not level['passed'] and sha(ROOT/'cup-neural-localization.pt')==loc['checkpoint_sha256']
    assert read(ROOT/'level-fit-r6/completion.json')['report_sha256']==sha(ROOT/'static-level-fit.json')
    archive=ROOT/'level-rounds/r6';archive.mkdir(parents=True)
    names=('static-level-fit.json','static-level-candidate.pt','level-localization-check.json','fit_scale_level.py',
        'run_scale_locator_fit.py','select_scale_static_components.py','evaluate_scale_static_holdout.py','candidate.json')
    hashes={}
    for n in names:
        shutil.copy2(ROOT/n,archive/n);hashes[n]=sha(archive/n);assert hashes[n]==sha(ROOT/n)
    write(archive/'index.json',dict(files=hashes,accepted=False,runtime=runtime))
    frozen={n:sha(ROOT/n) for n in ('cup-neural-localization.pt','cup-neural-localization.json','scope-selection.json','scale_rim_refiner.py')}
    protocol=dict(round='r7',plan_sha256=m['plan_sha256'],parent_checkpoint_sha256=m['checkpoint_sha256'],
        audit_sha256=sha(ROOT/m['collection_audit']),manual_review_sha256=sha(ROOT/m['collection_review']),frozen_locator=frozen,
        feature_sets=['compact_geometry','spatial_geometry'],ridge_penalties=[.001,.01,.1,1.,10.,100.],
        kernel_length_scales=[.5,1.,2.,4.],kernel_ridge_penalties=[.0001,.001,.01,.1],
        training='Episode-mean features/targets, 90 equally weighted training episodes; CUDA float64 ridge and RBF kernel ridge. Whole-episode validation. No held-out images.',
        selection='Only candidates passing every original per-size train/validation gate; minimize worst per-size validation MAE + 0.3*near-full max error, then overall validation MAE.',
        rationale='The previous 139-feature residual MLP overfits training and lacks explicit cup dimensions; use declared geometry and a regularized readout of image features.',
        required=['unchanged all-size localization','unchanged per-size level gates','CPU/GPU parity below 1e-4','exact inference-feature path'],
        unchanged=['108 episodes and splits','all original thresholds','nine unused reserved scenes','active controller and runtime sources'],
        scope='Known-size stationary cup perception; dimensions are fixture inputs, not visually estimated.',runtime_before=runtime)
    write(ROOT/'level-refit-r7-plan.json',protocol)
    for n in names[:3]:assert sha(ROOT/n)==hashes[n];(ROOT/n).unlink()
    print(json.dumps(dict(prepared=True,protocol_sha256=sha(ROOT/'level-refit-r7-plan.json'),frozen_locator=frozen)))
if __name__=='__main__':main()
