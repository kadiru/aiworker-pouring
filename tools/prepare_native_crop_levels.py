"""Freeze native re-crop augmentation while retaining the original accuracy checks."""
import json
from pathlib import Path
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged


def main():
    protected_unchanged()
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    previous = read(ROOT/'pair-r12-protocol.json')
    assert all(sha(ROOT/n) == h for n, h in previous['frozen_files'].items())
    crops = read(ROOT/'native-shift-crops.json')
    assert crops['passed'] and crops['additional_native_crops'] == 6390
    assert crops['crop_sha256'] == sha(ROOT/'native-shift-crops.npz')
    assert crops['validation_crops_added'] == crops['reserved_crops_added'] == 0
    files = list(previous['frozen_files'])+['pair-r12-protocol.json', 'pair-r12.json',
        'pair-r12-rgb.pt', 'pair-r12-color.pt', 'pair-r12-rgb.json', 'pair-r12-color.json',
        'native-shift-crops.npz', 'native-shift-crops.json', 'native-shift-crop-protocol.json',
        'cache_native_shifted_levels.py', 'prepare_native_crop_levels.py', 'fit_native_crop_levels.py']
    cfg = dict(previous)
    cfg.update(round='r13', seed=261110615, frozen_files={n: sha(ROOT/n) for n in files},
               initialization=dict(rgb='pair-r12-rgb.pt', color='pair-r12-color.pt'),
               native_crop_choices=3, jitter_pixels=0,
               method='Independent supervised RGB/color-outline CNNs, AdamW MSE plus 0.25 worst-quartile MSE. Each balanced training sample chooses its original crop or one of two native-camera XY-perturbed crops. No second resampling; color features computed after selection. Original train/validation images and all acceptance gates remain unchanged.',
               scope='243 training / 36 validation episodes; 6390 alternative native training crops are correlated augmentations, not new episodes. No reserved images, true-center inference or active-controller change.')
    with (ROOT/'pair-r13-protocol.json').open('x') as f:
        json.dump(cfg, f, indent=2); f.write('\n')
    print(json.dumps(dict(prepared='r13', training_episodes=243, alternative_crops=6390)), flush=True)


if __name__ == '__main__':
    main()
