"""Archive r7 and declare an 80-pixel-height geometric surface readout."""
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
    m=read(ROOT/'candidate.json');h=w.health();runtime=w.isolation();old=read(ROOT/'level-refit-r7-plan.json');plan=read(ROOT/'plan.json')
    assert not h['queue'] and not h['recording']['active'] and not h.get('error') and not h.get('learning_error')
    assert not m['trained'] and not (ROOT/'component-selection.json').exists()
    assert sha(ROOT/'plan.json')==m['plan_sha256'] and sha(ROOT/'best.pt')==m['checkpoint_sha256']
    assert all(sha(ROOT/n)==v for n,v in old['frozen_locator'].items())
    assert all(sha(w.HOST_ROOT/'src'/n)==v for n,v in m['runtime_sources'].items())
    assert all(sha(w.HOST_ROOT/Path(p).relative_to('/workspace'))==v for p,v in m['protected_parent_files'].items())
    started={read(p)['name'] for p in (w.HOST_ROOT/'outputs/fountain').glob('*/*SC_A_*/parameters.json')}
    assert not started & {s['name'] for s in plan['holdout']}
    assert not read(ROOT/'static-level-fit.json')['passed']
    assert read(ROOT/'level-fit-r7/completion.json')['report_sha256']==sha(ROOT/'static-level-fit.json')
    archive=ROOT/'level-rounds/r7';archive.mkdir()
    names=('static-level-fit.json','static-level-candidate.pt','level-localization-check.json','fit_scale_level.py','fit_scale_level_readout.py',
        'scale_level_readout.py','run_scale_locator_fit.py','run_scale_level_pipeline.py','select_scale_static_components.py','evaluate_scale_static_holdout.py','candidate.json')
    hashes={}
    for n in names:shutil.copy2(ROOT/n,archive/n);hashes[n]=sha(archive/n);assert hashes[n]==sha(ROOT/n)
    write(archive/'index.json',dict(files=hashes,runtime=runtime))
    protocol=dict(old,round='r8',crop_resolution=192,usable_height_pixels=80,surface_quantiles_percent=[2,5,10],
        surface_columns=13,projected_height_samples=101,feature_sets=['physical_surface'],
        rationale='Old 96-pixel crops use 40 pixels over usable cup height. Measure 192-pixel color envelopes, convert to heights using image-localized cup geometry/camera projection, then supervise remaining refraction/rendering bias.',runtime_before=runtime)
    write(ROOT/'level-refit-r8-plan.json',protocol)
    for n in names[:3]:assert sha(ROOT/n)==hashes[n];(ROOT/n).unlink()
    print(json.dumps(dict(prepared=True,protocol_sha256=sha(ROOT/'level-refit-r8-plan.json'))))
if __name__=='__main__':main()
