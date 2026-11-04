"""Explicit consumed episodes; future L30 cases can never enter this fit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from validate_pour_recording import decode, h5py
from pour_guard_vision import cup_crop
from pour_settled_study import ROOT, PARENT, PARENT_SHA

VALIDATION = {'T30_A_test_01', 'T30_A_test_08', 'T30_A_test_13', 'T30_A_test_21',
    'T30_A_test_04', 'T30_A_test_14', 'T30_A_test_26', 'H_A_test_00'}


def build_data(saved):
    cache = ROOT / 'settled-cache.pt'
    if cache.exists():
        return torch.load(cache, map_location='cpu', weights_only=False)
    assert hashlib.sha256((PARENT / 'best.pt').read_bytes()).hexdigest() == PARENT_SHA
    arrays = {s: {k: [] for k in ('image', 'target', 'cup_xy', 'episode')} for s in ('train', 'validation')}
    initial, manifest = [], []
    for p in sorted(Path('/workspace/outputs/fountain').glob('*/*/episode.hdf5')):
        with h5py.File(p, 'r') as f:
            spec = json.loads(f.attrs['spec']); name = spec.get('name', '')
            if not (name.startswith('T30_A_test_') or name.startswith('H_A_test_')):
                continue
            assert spec['spacing'] == .0018 and spec['dt'] == .0025
            assert f.attrs['status'] == 'complete' and f.attrs['camera_rig'] == 'A_task_wrist_v4'
            metrics = json.loads(f.attrs['metrics'])
            assert metrics['policy']['checkpoint_sha256'] == PARENT_SHA
            d = f['data/demo_0']; n = len(d['actions'])
            rows = json.loads((p.parent / 'samples.json').read_text())
            decisions = json.loads((p.parent / 'policy-decisions.json').read_text())
            assert len(rows) == n and len(decisions) == n
            initial.append(dict(name=name, image=decode(d['obs/task_image'][0], 'jpeg95').copy(),
                source_xyz=d['ground_truth/bottle_pose'][0, :3], cup_xyz=np.asarray(spec['cup_world'])))
            if not metrics['grasp_verified'] or not metrics['zero_spill']:
                continue
            xy = next(x['cup_xy'] for x in decisions if x.get('cup_xy') is not None)
            final = rows[-1]
            changed = [x['t'] for x in rows if abs(x['cup_ml'] - final['cup_ml']) > .5 * .0018**3 * 1e6]
            stable_after = max(changed, default=0.) + .35
            split = 'validation' if name in VALIDATION else 'train'
            used = []
            for i in range(0, n, 2):
                if rows[i]['t'] < stable_after:
                    continue
                assert abs(rows[i]['t'] - float(d['next_time'][i])) < 1e-5
                image = cup_crop(decode(d['next_obs/task_image'][i], 'jpeg95'), xy, saved['calibration'])
                for k, v in dict(image=image, target=final['cup_fill_fraction'], cup_xy=xy, episode=name).items():
                    arrays[split][k].append(v)
                used.append(i)
            manifest.append(dict(name=name, path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                split=split, frames=len(used), indices=used, stable_inventory_after_s=stable_after,
                target_final_height=final['cup_fill_fraction'], cup_crop_xy=xy,
                label='final settled particle surface; only frames after final cup inventory stabilizes',
                consumed_previous_test=True))
    assert len(initial) == 36 and len(manifest) == 35
    assert {x['name'] for x in manifest if x['split'] == 'validation'} == VALIDATION
    data = {s: {k: v if k == 'episode' else torch.from_numpy(np.asarray(v, dtype=np.uint8 if k == 'image' else np.float32))
        for k, v in a.items()} for s, a in arrays.items()}
    result = dict(data=data, initial=initial, manifest=manifest)
    torch.save(result, cache)
    (ROOT / 'data-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return result
