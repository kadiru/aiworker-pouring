"""Freeze the next fit only after new static recordings pass all admission."""
import hashlib,json
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    assert not (ROOT/'component-selection.json').exists() and not (ROOT/'pair-r7-protocol.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    p=read(ROOT/'protocol.json');assert p['version']==5
    c=read(ROOT/'level-cache-v3.json');assert c['passed'] and c['protocol_sha256']==sha(ROOT/'protocol.json')
    assert c['cache_sha256']==sha(ROOT/'level-cache-v3.npz') and c['metadata_sha256']==sha(ROOT/'level-cache-v3-meta.json')
    assert c['tracker_sha256']==sha(ROOT/'tracker-t1.pt') and c['tracker_module_sha256']==sha(ROOT/'dynamic_cup_tracker.py')
    assert c['admission_sha256']==sha(ROOT/'static-repair-tracker-admission.json') and read(ROOT/'static-repair-tracker-admission.json')['passed']
    old=read(ROOT/'pair-r4-protocol.json');archive=ROOT/'pair-r5-frozen-inputs'
    assert all(sha(archive/n)==v for n,v in old['frozen_files'].items())
    assert all(sha(ROOT/n)==v for n,v in old['frozen_files'].items() if n!='protocol.json')
    original=read(ROOT/'protocol-v3-before-level-repair.json');assert all(p[k]==original[k] for k in ('validation','holdout','gates'))
    m=read(ROOT/'candidate.json');assert all(sha(Path(n))==v for n,v in m['protected_parent_files'].items()) and all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    checks=read(ROOT/'level-repair-cpu-checks.json');assert checks['passed']
    assert all(sha(ROOT/n)==v for n,v in checks['sources'].items())
    files=list(old['frozen_files'])+['level-cache-v1.json','level-cache-v3.json','level-cache-v3.npz','level-cache-v3-meta.json',
        'train-r2-audit.json','train-r2-manual-review.json','static-repair-tracker-admission.json','static-repair-activation.json',
        'fit_level_repair_v3.py','prepare_level_repair_v3.py','repair_level_model.py','repair_level_model_v2.py','level-repair-cpu-checks.json',
        'pair-r5-rgb.pt','pair-r5-color.pt','pair-r5.json','pair-r5-protocol.json','cache_static_repair_collection.py']
    cfg=dict(round='r7',seed=261110207,locator='tracker-t1.pt',protocol_sha256=sha(ROOT/'protocol.json'),frozen_files={n:sha(ROOT/n) for n in sorted(set(files))},
        epochs=600,batch_size=64,lr=.0002,lr_steps=[301,451],jitter_pixels=2.5,top_quartile_weight=.25,wall_seconds=1800,max_episode_mae=.02,
        initialization=dict(rgb='pair-r5-rgb.pt',color='pair-r5-color.pt'),
        method='Independent supervised RGB/color-outline CNNs fine-tuned from separate r5 models; equal motion/size/episode sampling, MSE plus 0.25 worst-quartile MSE; AdamW. All original gates plus every-episode MAE <0.02. No cross-model loss, no RL.',
        scope='All original development episodes plus 36 reviewed static low/intermediate-fill training episodes. Retired r1 moving data excluded. Tracker, validation, reserved specs and original gates unchanged; no runtime activation.')
    with (ROOT/'pair-r7-protocol.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared='r7',counts=c['counts'],protocol_sha256=sha(ROOT/'pair-r7-protocol.json'))))
if __name__=='__main__':main()
