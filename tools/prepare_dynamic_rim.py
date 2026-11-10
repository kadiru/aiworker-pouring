"""Freeze the development-only moving-rim repair after actual review."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    m=read(ROOT/'candidate.json');assert m['high_seeds_accepted'] and not m['autonomy_accepted']
    assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    files=['protocol.json','plan.json','candidate.json','best.pt','cup-neural-localization.pt','scope-selection.json',
           'fit_dynamic_rim.py','fit_scale_rim_refiner.py','scale_rim_refiner.py','scale_data_admission.py','scale_rim_augmentation.py',m['collection_audit'],m['collection_review']]
    for part in ('seed_pilot','seed_remaining','low_train','train','validation'):
        ap=ROOT/(part+'-r0-audit.json');rp=ROOT/(part+'-r0-manual-review.json');a=read(ap);r=read(rp)
        assert a['complete'] and r['accepted'] and r['audit_sha256']==sha(ap)
        assert all(v['recording_validation']['valid'] and v['label']['accepted'] for v in a['records'])
        if part in ('train','validation'):assert all(v['motion']['passed'] for v in a['records'])
        files += [ap.name,rp.name]
    config=dict(seed=261110310,epochs=120,lr=.0003,lr_steps=[61,91],batch_size=16,
        augmentation=dict(translation_pixels=8,rotation_degrees=3,scale=[.98,1.02],brightness=[.95,1.05]),
        frame_selection='First 12 observation frames and final next-observation frame in all scenes; also every 20 Hz next-observation frame from 0.5 through 2.5 seconds in moving scenes.',
        frozen_files={n:sha(ROOT/n) for n in sorted(set(files))},
        scope='Separate supervised fine-rim repair. Frozen coarse CNN, no validation gradient updates, no reserved images, no active robot change. Unchanged 3 mm/absence/temporal/parity gates. Parent affine correction is replaced by direct fine-model learning, not applied twice.')
    with (ROOT/'rim-d0-protocol.json').open('x') as f:json.dump(config,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared=True,autonomy_accepted=False)))
if __name__=='__main__':main()
