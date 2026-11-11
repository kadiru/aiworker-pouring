"""Project measured level/rim labels into existing predicted-origin crop coordinates."""
import json
import numpy as np
import torch
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged
from perception_midfill_data import entries
from physical_level_augmentation import crop_maps
from pour_guard_vision import project


def main():
    protected_unchanged()
    assert not (ROOT/'component-selection.json').exists()
    assert not read(ROOT/'pair-r13.json')['passed']
    assert read(ROOT/'landmark-geometry-contracts.json')['passed']
    meta = read(ROOT/'level-cache-v6-meta.json')
    cache = dict(np.load(ROOT/'level-cache-v6.npz'))
    with np.load(ROOT/'native-shift-crops.npz') as archive: offsets = archive['offsets_xy']
    tracker = torch.load(ROOT/'tracker-t3.pt', map_location='cpu', weights_only=False)
    cal = dict(tracker['calibration'], table_z=1.04)
    specs = {s['name']: s for s, r, split in entries()}
    arrays = {}
    for split, m in meta.items():
        xy = cache[split+'_xy'].astype(np.float64)
        choices = np.stack((xy, xy+offsets[:, 0], xy+offsets[:, 1]), 1) if split == 'train' else xy[:, None]
        labels = np.empty((len(xy), choices.shape[1], 2, 2), dtype=np.float64)
        for i, item in enumerate(m['meta']):
            s = specs[item['name']];g = m['geometry'][i];target = float(cache[split+'_y'][i])
            actual = np.asarray(s['cup_world'][:2])
            uv = np.stack([project(np.r_[actual, z], cal) for z in
                           (1.04+g['rim_z_m'], 1.04+g['bottom_z_m']+target*(g['rim_z_m']-g['bottom_z_m']))])
            A, b = crop_maps(torch.tensor(choices[i]), [g]*len(choices[i]), cal)
            labels[i] = torch.linalg.solve(A, (torch.tensor(uv)[None]-b[:, None]).transpose(1, 2)).transpose(1, 2).numpy()
        assert np.isfinite(labels).all() and labels.min() > 4 and labels.max() < 92
        arrays[split+'_points'] = labels.astype(np.float32)
        arrays[split+'_xy_choices'] = choices.astype(np.float32)
    np.savez_compressed(ROOT/'landmark-level-labels.npz', **arrays)
    files = ['protocol.json', 'plan.json', 'candidate.json', 'tracker-t3.pt', 'tracker-t3.json',
             'level-cache-v6.npz', 'level-cache-v6-meta.json', 'level-cache-v6.json',
             'native-shift-crops.npz', 'native-shift-crops.json', 'landmark-level-labels.npz',
             'landmark_level_model.py', 'test_landmark_level_geometry.py', 'landmark-geometry-contracts.json',
             'prepare_landmark_level_data.py', 'fit_landmark_levels.py', 'physical_level_augmentation.py',
             'repair_level_model.py', 'scale_color_mask.py', 'fit_dynamic_pair.py', 'repair_level_model_v2.py',
             'perception_midfill_data.py', 'perception_repair_data.py', 'pair-r13.json']
    cfg = dict(round='r14', seed=261110616, epochs=800, batch_size=64, lr=.0005,
               lr_steps=[301, 601], wall_seconds=2400, locator='tracker-t3.pt',
               protocol_sha256=sha(ROOT/'protocol.json'), calibration=cal,
               frozen_files={n: sha(ROOT/n) for n in files},
               point_weight=.2, level_weight=40., top_quartile_weight=.25,
               counts=read(ROOT/'level-cache-v6.json')['counts'],
               method='Separate supervised rim/surface heatmap CNNs for RGB and color-outline inputs. Gaussian heatmap cross entropy + 0.2 point smooth-L1 + 40*(level MSE + 0.25 worst-quartile MSE). Calibrated image rays reconstruct normalized liquid height. Native train-only crop alternatives; no validation gradients or reserved images.',
               scope='True scene centers and measured heights author training/validation labels only. Inference receives RGB, image-derived crop XY, declared cup geometry and fixed camera calibration. No true-position inputs, RL or active-controller update.')
    with (ROOT/'pair-r14-protocol.json').open('x') as f:
        json.dump(cfg, f, indent=2); f.write('\n')
    print(json.dumps(dict(prepared='r14', counts=cfg['counts'])), flush=True)


if __name__ == '__main__': main()
