"""Development-only sensitivity audit; truth crops never enter model inputs/cache."""
import json
from pathlib import Path
import numpy as np
import torch
from perception_repair_data import entries
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged
from validate_pour_recording import h5py, decode
from pour_guard_vision import cup_crop
from repair_level_model_v2 import predict_repair


@torch.inference_mode()
def main():
    assert not (ROOT/'component-selection.json').exists()
    assert not (ROOT/'midfill-crop-diagnosis.json').exists()
    assert torch.cuda.device_count() == 1 and '4070' in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    protected_unchanged()
    meta = read(ROOT/'level-cache-v5-meta.json')['validation']
    # NpzFile indexing decompresses on every access; materialize once.
    with np.load(ROOT/'level-cache-v5.npz') as archive:
        cache = {k: archive[k] for k in archive.files if k.startswith('validation_')}
    specs = {s['name']: (s, r) for s, r, split in entries() if split == 'validation'}
    kinds = ('estimated', 'truth_diagnostic', 'x_minus_1mm', 'x_plus_1mm', 'y_minus_1mm', 'y_plus_1mm')
    images = {k: [] for k in kinds}
    positions = {k: [] for k in kinds}
    for name in sorted(specs):
        s, r = specs[name]
        ids = [i for i, m in enumerate(meta['meta']) if m['name'] == name]
        path = Path(r['path']); assert sha(path) == r['sha256']
        with h5py.File(path) as f:
            d = f['data/demo_0']
            cal = dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),
                       T=d['obs/cup_detail_world_T_camera'][0].tolist(), table_z=1.04,
                       cup_geometry=s['declared_cup_geometry'])
            for i in ids:
                raw = decode(d['next_obs/cup_detail_image'][meta['meta'][i]['index']], 'jpeg95')
                estimated = cache['validation_xy'][i]
                points = (estimated, np.asarray(s['cup_world'][:2]),
                          estimated+[-.001, 0], estimated+[.001, 0],
                          estimated+[0, -.001], estimated+[0, .001])
                for k, xy in zip(kinds, points):
                    crop = cup_crop(raw, xy, cal)
                    # Float32 cache coordinates can move a bilinear sample by a tiny amount.
                    if k == 'estimated':
                        assert np.abs(crop.astype(float)-cache['validation_x'][i]).max() <= 1
                        crop = cache['validation_x'][i]
                    images[k].append((i, crop)); positions[k].append((i, xy))
    report = dict(cases=[], helper_sha256=sha(Path(__file__)),
                  cache_sha256=sha(ROOT/'level-cache-v5.npz'),
                  scope='All 36 development-validation episodes. Exact centers used only for sensitivity diagnosis; no truth crop is saved as training or inference data. No reserved images, fitting, or runtime changes.')
    predictions = {}
    for branch in ('rgb', 'color'):
        artifact = torch.load(ROOT/f'pair-r10-{branch}.pt', map_location='cpu', weights_only=False)
        predictions[branch] = {}
        for kind in kinds:
            x = torch.tensor(np.stack([v for i, v in sorted(images[kind])]), device='cuda')
            xy = np.stack([v for i, v in sorted(positions[kind])])
            values = []
            for j in range(0, len(x), 128):
                pred, valid = predict_repair(x[j:j+128], xy[j:j+128], meta['geometry'][j:j+128], artifact)
                assert valid.all(); values.extend(pred.cpu().tolist())
            predictions[branch][kind] = np.asarray(values)
    target = cache['validation_y']
    for name in sorted(specs):
        ids = [i for i, m in enumerate(meta['meta']) if m['name'] == name]
        s, r = specs[name]
        row = dict(name=name, samples=len(ids), target=float(target[ids].mean()),
                   center_error_mm=float(np.linalg.norm(cache['validation_xy'][ids[0]]-s['cup_world'][:2])*1000), branches={})
        for branch in predictions:
            row['branches'][branch] = {k: dict(mae=float(np.abs(v[ids]-target[ids]).mean()),
                                               bias=float((v[ids]-target[ids]).mean()))
                                       for k, v in predictions[branch].items()}
        report['cases'].append(row)
    protected_unchanged()
    with (ROOT/'midfill-crop-diagnosis.json').open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    worst = sorted(report['cases'], key=lambda r: -r['branches']['color']['estimated']['mae'])[:6]
    print(json.dumps(worst), flush=True)


if __name__ == '__main__':
    main()
