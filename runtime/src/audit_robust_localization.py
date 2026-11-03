"""CPU-only replay of initial RGB localization in the frozen aborted case."""
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from validate_pour_recording import decode, h5py
from pour_guard_vision import locate, project
from pour_robust_study import ROOT, CHECKPOINT_SHA


def main():
    torch.set_num_threads(2)
    assert hashlib.sha256((ROOT / 'best.pt').read_bytes()).hexdigest() == CHECKPOINT_SHA
    saved = torch.load(ROOT / 'best.pt', map_location='cpu', weights_only=False)
    report = json.loads((ROOT / 'robust-results.json').read_text())
    cases = []
    for row in report['cases']:
        if row['stop_reason'] != 'localization_unconfirmed':
            continue
        path = Path(row['path']) / 'episode.hdf5'
        samples = json.loads((path.parent / 'samples.json').read_text())
        neck = project(np.asarray(samples[0]['bottle_pose_xyz']) + [0., 0., .135], saved['calibration'])
        bounds = saved['source_bank']['bounds']
        in_bounds = bounds[0] <= neck[0] <= bounds[2] and bounds[1] <= neck[1] <= bounds[3]
        frames = []
        with h5py.File(path, 'r') as f:
            d = f['data/demo_0']
            for i in sorted({0, 1, 5, len(d['actions']) - 1}):
                rgb = decode(d['obs/task_image'][i], f.attrs['image_encoding']).copy()
                matches = {}
                for kind in ('source', 'cup'):
                    bank = saved[kind + '_bank']
                    uv, score = locate(rgb, bank, device='cpu')
                    threshold = saved['thresholds'][kind + '_localization_score']
                    matches[kind] = dict(uv=uv.tolist(), score=score, threshold=threshold,
                        accepted=score >= threshold, search_bounds=bank['bounds'])
                frames.append(dict(observation_index=i, matches=matches))
                if i == 0:
                    Image.fromarray(rgb).save(ROOT / 'localization-initial-task.jpg', quality=95)
        cases.append(dict(name=row['name'], episode=str(path), frames=frames,
            offline_measured_neck_projection=neck.tolist(),
            measured_neck_inside_source_search_bounds=bool(in_bounds),
            projection_note='Offline use of the first measured bottle pose and fixed camera calibration to diagnose coverage; privileged geometry never entered runtime control.'))
    result = dict(checkpoint_sha256=CHECKPOINT_SHA, device='cpu', cases=cases,
        evidence='Offline CPU recomputation on saved observation images using the unchanged template bank. The original runtime logged the rejection reason, not its numeric localization scores; CPU and GPU arithmetic can differ slightly.',
        training_performed=False, controller_changed=False)
    (ROOT / 'localization-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
