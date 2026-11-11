"""Ablate absolute-position shortcut after observing r8 development errors."""
import json
from pathlib import Path
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged

def main():
    protected_unchanged();assert not read(ROOT/'pair-r8.json')['passed']
    assert read(ROOT/'tracker-t2.json')['passed'] and read(ROOT/'level-cache-v4.json')['passed']
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    cfg=read(ROOT/'pair-r8-protocol.json')
    assert all(sha(ROOT/n)==v for n,v in cfg['frozen_files'].items())
    extra=['pair-r8.json','pair-r8-protocol.json','pair-r8-rgb.pt','pair-r8-color.pt',
           'pair-r8-rgb.json','pair-r8-color.json','fit_expanded_levels_no_xy.py','prepare_expanded_levels_no_xy.py']
    cfg['frozen_files'].update({n:sha(ROOT/n) for n in extra})
    cfg.update(round='r9',seed=261110609,lr=.0002,initialization=dict(rgb='pair-r8-rgb.pt',color='pair-r8-color.pt'),
               method='Independent supervised RGB/color-outline models, same regenerated crops and all original gates; absolute table XY context zeroed. Known geometry remains. AdamW MSE plus worst-quartile MSE; no cross-model loss or reserved data.',
               scope='Controlled follow-up to r8 low-fill generalization errors. Tracker-t2 and train/validation/holdout assignments remain unchanged. No runtime activation.')
    with (ROOT/'pair-r9-protocol.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared='r9',absolute_xy_input=False)),flush=True)

if __name__=='__main__':main()
