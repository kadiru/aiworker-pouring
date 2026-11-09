"""Archive rejected native-rim r4 and predeclare the r5 augmentation revision."""
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
    report=read(ROOT/'cup-neural-localization.json');completion=read(ROOT/'locator-fit-r4/completion.json')
    assert not report['passed'] and completion['exit_code']==0 and completion['report_sha256']==sha(ROOT/'cup-neural-localization.json')
    archive=ROOT/'locator-rounds/r4';archive.mkdir()
    names=('cup-neural-localization.json','cup-neural-localization-attempt.pt','cup-locator-data-manifest.json',
        'fit_scale_rim_refiner.py','scale_rim_refiner.py','run_scale_locator_fit.py','run_scale_rim_pipeline.py',
        'fit_scale_level.py','evaluate_scale_static_holdout.py','select_scale_static_components.py','candidate.json','locator-refit-r4-plan.json')
    hashes={}
    for n in names:
        shutil.copy2(ROOT/n,archive/n);hashes[n]=sha(archive/n);assert hashes[n]==sha(ROOT/n)
    write(archive/'index.json',dict(files=hashes,runtime=witness,accepted=False))
    protocol=read(ROOT/'locator-refit-r4-plan.json')
    protocol.update(round='r5',seed=26110985,learning_rate=.0001,
        fine_initialization='locator-rounds/r4/cup-neural-localization-attempt.pt',
        fine_initialization_sha256=hashes['cup-neural-localization-attempt.pt'],
        augmentation=dict(kind='affine',rotation_degrees=12.,scale_min=.9,scale_max=1.1,aspect_log_range=.06,translation_pixels=24.),
        rationale='r4 validation best 3.0905 mm at epoch 20, training maximum 3.7451 mm; later training improves while validation worsens. Broaden image transformations to address observed overfit without changing splits or gates.',
        method='Same frozen r3 coarse and native-detail refiner; initialize from rejected r4 best-validation attempt and fit with label-consistent continuous affine RGB crop augmentation.',
        runtime_before=witness)
    write(ROOT/'locator-refit-r5-plan.json',protocol)
    for n in names[:3]:assert sha(ROOT/n)==hashes[n];(ROOT/n).unlink()
    print(json.dumps(dict(prepared=True,protocol_sha256=sha(ROOT/'locator-refit-r5-plan.json'),r4_archived=True)))
if __name__=='__main__':main()
