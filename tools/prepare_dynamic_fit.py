"""Freeze reviewed development inputs; no held-out images or runtime activation."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path('/workspace/learning/A-scale-dynamic-20261109-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--round',choices=('r3','r4'),default='r3');p.add_argument('--rim-round',choices=('d0','d1','d2'),default='d1');a=p.parse_args()
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    protocol=read(ROOT/'protocol.json');m=read(ROOT/'candidate.json');assert m['high_seeds_accepted'] and not m['autonomy_accepted'] and not protocol['draft_only']
    assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items())
    assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    parts=['seed_pilot','seed_remaining','low_train','validation']
    aborted=False;locator='cup-neural-localization.pt';locator_selection='scope-selection.json'
    repair='rim-'+a.rim_round
    if a.round=='r4':
        aborted=(ROOT/'pair-r3-aborted.json').exists()
        prior=read(ROOT/('pair-r3-aborted.json' if aborted else 'pair-r3.json'));assert not prior['passed'];parts.append('train')
        if aborted:
            assert prior['optimizer_steps']==0 and read(ROOT/(repair+'.json'))['passed']
            locator=repair+'.pt';locator_selection=repair+'-selection.json'
            report=read(ROOT/(repair+'.json'));assert report['checkpoint_sha256']==sha(ROOT/locator)
            selection=dict(passed=True,scope='all_nine_declared_cup_sizes',checkpoint_sha256=sha(ROOT/locator),module_sha256=sha(ROOT/'scale_rim_refiner.py'),repair_report_sha256=sha(ROOT/(repair+'.json')),model_activated=False)
            with (ROOT/locator_selection).open('x') as f:json.dump(selection,f,indent=2);f.write('\n')
    files=['protocol.json','plan.json','candidate.json','best.pt','primary-level.pt','color-mask-r2.pt','cup-neural-localization.pt','scope-selection.json',
        'fit_dynamic_pair.py','dynamic_level_data.py','fit_scale_level.py','scale_level_cnn.py','scale_color_mask.py','scale_data_admission.py','scale_rim_refiner.py','scale_rim_augmentation.py','scale_level_readout.py']
    files += [m['collection_audit'],m['collection_review'],locator,locator_selection]
    if aborted:files+=[repair+'.json',repair+'-protocol.json','pair-r3-aborted.json']
    for part in parts:
        audit=part+'-r0-audit.json';review=part+'-r0-manual-review.json';r=read(ROOT/review);v=read(ROOT/audit)
        assert v['complete'] and r['accepted'] and r['audit_sha256']==sha(ROOT/audit)
        assert all(s['recording_validation']['valid'] and s['label']['accepted'] for s in v['records'])
        if part in ('validation','train'):assert all(s['motion']['passed'] for s in v['records'])
        files += [audit,review]
    rgb='primary-level.pt' if a.round=='r3' or aborted else 'pair-r3-rgb.pt';color='color-mask-r2.pt' if a.round=='r3' or aborted else 'pair-r3-color.pt'
    files += [rgb,color]
    config=dict(round=a.round,seed=261110203 if a.round=='r3' else 261110204,parts=parts,locator=locator,locator_selection=locator_selection,
        moving_frame_sampling='Every recorded 20 Hz frame from 0.5 through 2.5 seconds. Denser than the scene draft stride of two; expanded before any moving-view fit or model-error evaluation. Scene IDs and gates unchanged.',
        rgb_initialization=rgb,color_initialization=color,color_initialization_sha256=sha(ROOT/color),
        rgb=dict(epochs=240,lr=.00035,lr_steps=[121,181]),color=dict(epochs=300,lr=.0007,lr_steps=[151,241]),jitter_pixels=.35,
        protocol_sha256=sha(ROOT/'protocol.json'),frozen_files={n:sha(ROOT/n) for n in sorted(set(files))},
        scope='Train only on declared train images; select on separate development validation; retired and fresh reserved images excluded. No autonomous acceptance.')
    with (ROOT/('pair-'+a.round+'-protocol.json')).open('x') as f:json.dump(config,f,indent=2);f.write('\n')
    print(json.dumps(dict(round=a.round,parts=parts,autonomy_accepted=False)))
if __name__=='__main__':main()
