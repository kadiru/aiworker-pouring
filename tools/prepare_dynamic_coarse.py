"""Freeze coarse-detector repair on already reviewed development episodes."""
import argparse,json,urllib.request
from pathlib import Path
from prepare_dynamic_rim import ROOT,read,sha
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round',choices=('d1','d2'),default='d1');args=parser.parse_args();prefix='coarse-'+args.round
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10));assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    assert not (ROOT/'component-selection.json').exists() and not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    old=read(ROOT/'rim-d0-protocol.json');assert read(ROOT/'rim-d0-failed.json')['checkpoint_produced'] is False
    for n,v in old['frozen_files'].items():
        if n!='fit_dynamic_rim.py':assert sha(ROOT/n)==v
    m=read(ROOT/'candidate.json');assert all(sha(Path(p))==v for p,v in m['protected_parent_files'].items());assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
    files=list(old['frozen_files'])+['fit_dynamic_coarse.py','rim-d0-failed.json','rim-d0-crop-diagnostic.json']
    if args.round=='d2':
        previous=read(ROOT/'coarse-d1.json');assert not previous['passed_for_refinement'] and previous['checkpoint_sha256']==sha(ROOT/'coarse-d1.pt')
        files+=['coarse-d1.pt','coarse-d1.json','coarse-d1-protocol.json']
    cfg=dict(seed=261110411,epochs=100,lr=.00003,lr_steps=[51,76],maximum_native_crop_error_pixels=24.,absence_margin=read(ROOT/'protocol.json')['gates']['localization_negative_margin'],
        augmentation=dict(translation_pixels=12,synthetic_cup_absence_probability=.25,brightness=[.95,1.05]),
        frozen_files={n:sha(ROOT/n) for n in sorted(set(files))},scope='Supervised coarse heatmap focal loss plus 2x smooth-L1 offsets; equal size/episode sampling. Provisional crop coverage only, not relaxed final 3 mm acceptance. Reserved data excluded; old failed attempt retained.')
    if args.round=='d2':
        cfg.update(seed=261110413,epochs=120,lr=.00006,lr_steps=[61,91],initialization='coarse-d1.pt',negative_probability=.5,balance_motion_groups=True)
        cfg['augmentation']['synthetic_cup_absence_probability']=.5
        cfg['scope']='Supervised coarse repair from retained d1; equal static/moving group weight, then equal size/episode. Half the training samples use the unchanged absent-cup mask. Focal heatmap plus 2x smooth-L1 offset loss. Same accuracy/absence gates and validation identities; no reserved data or activation.'
    with (ROOT/(prefix+'-protocol.json')).open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared=True,autonomy_accepted=False)))
if __name__=='__main__':main()
