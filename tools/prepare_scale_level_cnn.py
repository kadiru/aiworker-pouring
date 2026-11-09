"""Archive rejected surface model and predeclare trainable size-aware CNN."""
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
    m=read(ROOT/'candidate.json');h=w.health();runtime=w.isolation();old=read(ROOT/'level-refit-r8-plan.json');plan=read(ROOT/'plan.json')
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    assert not m['trained'] and not (ROOT/'component-selection.json').exists()
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(ROOT/n)==v for n,v in old['frozen_locator'].items())
    assert all(sha(w.HOST_ROOT/'src'/n)==v for n,v in m['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT/Path(p).relative_to('/workspace'))==v for p,v in m['protected_parent_files'].items())
    started={read(p)['name'] for p in (w.HOST_ROOT/'outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in plan['holdout']}
    assert not read(ROOT/'static-level-fit.json')['passed']
    assert read(ROOT/'level-fit-r8/completion.json')['report_sha256']==sha(ROOT/'static-level-fit.json')
    archive=ROOT/'level-rounds/r8';archive.mkdir()
    names=('static-level-fit.json','static-level-candidate.pt','level-localization-check.json','fit_scale_level.py','fit_scale_level_readout.py',
        'scale_level_readout.py','scale_surface_geometry.py','run_scale_locator_fit.py','run_scale_level_pipeline.py','select_scale_static_components.py','evaluate_scale_static_holdout.py','candidate.json')
    hashes={}
    for n in names:shutil.copy2(ROOT/n,archive/n);hashes[n]=sha(archive/n);assert hashes[n]==sha(ROOT/n)
    write(archive/'index.json',dict(files=hashes,runtime=runtime))
    protocol=dict(round='r9',plan_sha256=m['plan_sha256'],parent_checkpoint_sha256=m['checkpoint_sha256'],
        audit_sha256=old['audit_sha256'],manual_review_sha256=old['manual_review_sha256'],frozen_locator=old['frozen_locator'],
        epochs=240,batch_size=64,learning_rate=.00005,weight_decay=.0001,seed=26110999,
        augmentation=dict(translation_pixels=1.5,scale_min=.98,scale_max=1.02,rotation_degrees=1.5,brightness_min=.95,brightness_max=1.05),
        training='Train all retained CupEstimator weights plus a zero-initialized 11/16/1 tanh context correction; AdamW, equal episode/size sampling, MSE. Image augmentation leaves the physical level label unchanged.',
        context='Declared height, diameter, diameter/height, and image-localized XY; corrections depend also on the CNN level.',
        rationale='Both frozen-CNN readouts fail. Fine-tune RGB features to discriminate liquid level relative to the visible vessel while reducing sensitivity to small crop errors.',
        unchanged=old['unchanged'],required=old['required'],selection=old['selection'],scope=old['scope'],runtime_before=runtime)
    write(ROOT/'level-refit-r9-plan.json',protocol)
    for n in names[:3]:assert sha(ROOT/n)==hashes[n];(ROOT/n).unlink()
    print(json.dumps(dict(prepared=True,protocol_sha256=sha(ROOT/'level-refit-r9-plan.json'))))
if __name__=='__main__':main()
