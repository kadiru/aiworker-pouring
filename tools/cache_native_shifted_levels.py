"""Training-only native camera re-crops; preserve original validation inputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged
from perception_midfill_data import entries
from validate_pour_recording import h5py, decode
from pour_guard_vision import cup_crop


def main():
    assert not (ROOT/'component-selection.json').exists()
    assert not read(ROOT/'pair-r12.json')['passed']
    assert not (ROOT/'native-shift-crops.json').exists()
    protected_unchanged()
    report = read(ROOT/'level-cache-v6.json')
    assert report['passed'] and report['cache_sha256'] == sha(ROOT/'level-cache-v6.npz')
    assert report['metadata_sha256'] == sha(ROOT/'level-cache-v6-meta.json')
    meta = read(ROOT/'level-cache-v6-meta.json')['train']
    with np.load(ROOT/'level-cache-v6.npz') as archive:
        original = archive['train_x']; xy = archive['train_xy']
    assert len(original) == len(meta['meta']) == 3195
    files = ['protocol.json', 'candidate.json', 'plan.json', 'tracker-t3.pt',
             'level-cache-v6.npz', 'level-cache-v6-meta.json', 'level-cache-v6.json',
             'perception_midfill_data.py', 'perception_repair_data.py',
             'cache_native_shifted_levels.py', 'pair-r12.json']
    plan = dict(seed=261110614, radius_m=.002, alternatives_per_frame=2,
                frozen_files={n: sha(ROOT/n) for n in files},
                scope='Two counterbalanced XY perturbations per training image, cropped directly from the original 1024px recording. Centers come from tracker-t3; true scene centers never set these crops. No validation/reserved augmentation.')
    with (ROOT/'native-shift-crop-protocol.json').open('x') as f:
        json.dump(plan, f, indent=2); f.write('\n')
    rng = np.random.default_rng(plan['seed'])
    angle = rng.uniform(0, 2*np.pi, len(original)); radius = np.sqrt(rng.uniform(0, 1, len(original)))*plan['radius_m']
    delta = np.stack((np.cos(angle)*radius, np.sin(angle)*radius), -1)
    offsets = np.stack((delta, -delta), 1)
    images = np.empty((len(original), 2, 96, 96, 3), dtype=np.uint8)
    written = np.zeros(len(original), dtype=bool); sources = []
    for s, r, split in entries():
        if split != 'train': continue
        ids = [i for i, m in enumerate(meta['meta']) if m['name'] == s['name']]
        assert ids and not written[ids].any()
        path = Path(r['path']); assert sha(path) == r['sha256']
        with h5py.File(path) as f:
            d = f['data/demo_0']
            cal = dict(K=f['calibration/cup_detail/intrinsics'][:].tolist(),
                       T=d['obs/cup_detail_world_T_camera'][0].tolist(), table_z=1.04,
                       cup_geometry=s['declared_cup_geometry'])
            for i in ids:
                raw = decode(d['next_obs/cup_detail_image'][meta['meta'][i]['index']], 'jpeg95')
                assert raw.shape == (1024, 1024, 3)
                baseline = cup_crop(raw, xy[i], cal)
                assert np.abs(baseline.astype(float)-original[i]).max() <= 1
                for j in range(2):
                    image = cup_crop(raw, xy[i]+offsets[i, j], cal)
                    a = image.astype(np.int64); red, green, blue = a.transpose(2, 0, 1)
                    visible = (red*1000 > green*1400+6375)&(red*1000 > blue*1200+5100)&(red > 30)
                    assert visible[14:83, 30:82].sum() >= 20, 'Required color removed: do not discard failed samples'
                    images[i, j] = image
                written[i] = True
        sources.append(dict(name=s['name'], sha256=r['sha256'], frames=len(ids), split=split))
        if len(sources)%30 == 0: print(json.dumps(dict(native_episodes=len(sources))), flush=True)
    assert written.all() and len(sources) == 243
    np.savez_compressed(ROOT/'native-shift-crops.npz', images=images, offsets_xy=offsets)
    # An honest diagnostic contact sheet of training inputs, not new renders.
    ids = [0, len(original)//2, len(original)-1]
    canvas = Image.new('RGB', (864, 936), 'white'); draw = ImageDraw.Draw(canvas)
    for row, i in enumerate(ids):
        for col, im in enumerate((original[i], images[i, 0], images[i, 1])):
            canvas.paste(Image.fromarray(im).resize((288, 288), Image.Resampling.NEAREST), (col*288, row*312+24))
            draw.text((col*288+2, row*312+2), meta['meta'][i]['name']+' / '+('original' if col == 0 else f'offset {col}'), fill='black')
    canvas.save(ROOT/'native-shift-crops-preview.png')
    assert all(sha(ROOT/n) == h for n, h in plan['frozen_files'].items())
    protected_unchanged()
    result = dict(passed=True, training_episodes=243, original_training_frames=3195,
                  additional_native_crops=6390, validation_crops_added=0, reserved_crops_added=0,
                  radius_m=plan['radius_m'], all_original_frames_retained=True,
                  sources=sources, crop_sha256=sha(ROOT/'native-shift-crops.npz'),
                  preview_sha256=sha(ROOT/'native-shift-crops-preview.png'),
                  protocol_sha256=sha(ROOT/'native-shift-crop-protocol.json'),
                  scope=plan['scope'])
    with (ROOT/'native-shift-crops.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sources'}), flush=True)


if __name__ == '__main__':
    main()
