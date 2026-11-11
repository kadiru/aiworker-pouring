"""Train rim localization on all admitted midfill scenes, preserving earlier fits."""
import json
from pathlib import Path
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged
from perception_midfill_data import entries


def main():
    assert not (ROOT/'component-selection.json').exists()
    assert read(ROOT/'midfill-localization-data-contracts.json')['passed']
    assert not read(ROOT/'pair-r11.json')['passed'], 'Stop expanding once the existing repair passes'
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    data = entries(); protected_unchanged()
    files = ['protocol.json', 'plan.json', 'candidate.json', 'rim-d3.pt', 'rim-d3.json',
             'coarse-d2.pt', 'cup-neural-localization.pt', 'tracker-t2.pt', 'dynamic_cup_tracker.py',
             'perception_repair_data.py', 'perception_midfill_data.py', 'fit_midfill_rim.py',
             'prepare_midfill_rim.py', 'scale_rim_refiner.py', 'scale_rim_augmentation.py',
             'fit_scale_rim_refiner.py', 'train-r3-audit.json', 'train-r3-manual-review.json',
             'midfill-tracker-admission.json', 'midfill-crop-diagnosis.json', 'pair-r11.json',
             'test_perception_midfill_data.py', 'midfill-localization-data-contracts.json']
    cfg = dict(seed=261110612, epochs=200, lr=.0001, lr_steps=[101, 151], wall_seconds=2700,
               coarse_checkpoint='coarse-d2.pt', fine_initialization='rim-d3.pt', balance_motion_groups=True,
               frozen_files={n: sha(ROOT/n) for n in files},
               entries=[dict(name=s['name'], split=split, path=r['path'], sha256=r['sha256']) for s, r, split in data],
               scope='243 training / 36 validation episodes, 12 initial frames each. New midfill training recordings now supervise the rim model too. Original localization, absence and temporal gates unchanged; no reserved data or active-controller change.')
    with (ROOT/'rim-d4-protocol.json').open('x') as f:
        json.dump(cfg, f, indent=2); f.write('\n')
    print(json.dumps(dict(prepared='d4', training=243, validation=36)), flush=True)


if __name__ == '__main__':
    main()
