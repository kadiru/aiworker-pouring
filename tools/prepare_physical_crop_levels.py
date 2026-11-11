"""Freeze measured midfill extension after physical, visual and tracker admission."""
import json
from pathlib import Path
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged

def main():
    assert read(ROOT/'physical-crop-contracts.json')['passed']
    protected_unchanged();p=read(ROOT/'protocol.json');old=read(ROOT/'protocol-v5-before-midfill.json')
    assert p['version']==6 and all(p[k]==old[k] for k in ('validation','holdout','gates'))
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    c=read(ROOT/'level-cache-v5.json');assert c['passed'] and c['protocol_sha256']==sha(ROOT/'protocol.json')
    assert c['counts']==dict(train=dict(episodes=243,frames=3195),validation=dict(episodes=36,frames=918))
    assert c['cache_sha256']==sha(ROOT/'level-cache-v5.npz') and c['metadata_sha256']==sha(ROOT/'level-cache-v5-meta.json')
    assert c['tracker_sha256']==sha(ROOT/'tracker-t2.pt') and c['admission_sha256']==sha(ROOT/'midfill-tracker-admission.json')
    assert read(ROOT/'midfill-tracker-admission.json')['passed'] and read(ROOT/'train-r3-manual-review.json')['accepted']
    previous=read(ROOT/'pair-r10-protocol.json')
    assert all(sha(ROOT/n)==v for n,v in previous['frozen_files'].items())
    files=['protocol.json','protocol-v5-before-midfill.json','plan.json','candidate.json','tracker-t2.pt','tracker-t2.json',
           'dynamic_cup_tracker.py','level-cache-v5.npz','level-cache-v5-meta.json','level-cache-v5.json','level-cache-v4.json',
           'midfill-tracker-admission.json','train-r3-audit.json','train-r3-manual-review.json','midfill-activation.json',
           'fit_physical_crop_levels.py','prepare_physical_crop_levels.py','physical_level_augmentation.py','test_physical_level_augmentation.py','physical-crop-contracts.json','midfill-crop-diagnosis.json','repair_level_model.py','repair_level_model_v2.py',
           'fit_dynamic_pair.py','scale_level_cnn.py','scale_rim_augmentation.py','scale_color_mask.py',
           'pair-r10-rgb.pt','pair-r10-color.pt','pair-r10.json','pair-r10-protocol.json','cache_midfill_levels.py']
    cfg=dict(round='r11',seed=261110611,locator='tracker-t2.pt',protocol_sha256=sha(ROOT/'protocol.json'),
             frozen_files={n:sha(ROOT/n) for n in files},epochs=800,batch_size=64,lr=.00015,
             lr_steps=[401,601],xy_radius_m=.002,top_quartile_weight=.25,wall_seconds=1800,max_episode_mae=.02,
             initialization=dict(rgb='pair-r10-rgb.pt',color='pair-r10-color.pt'),
             method='Independent supervised RGB/color-outline CNNs, geometry only (no absolute XY context); AdamW MSE plus 0.25 worst-quartile MSE, balanced motion/size/episode sampling. Camera-consistent random XY crop perturbation within 2 mm, matching the observed localization envelope. Color mask/edges recomputed after perturbing RGB. No RL or cross-model loss.',
             scope='243 training / 36 validation episodes. Frozen tracker-t2; original accuracy gates and per-episode <2 percentage points retained. No reserved images or active-controller update.')
    with (ROOT/'pair-r11-protocol.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared='r11',counts=c['counts'])),flush=True)

if __name__=='__main__':main()
