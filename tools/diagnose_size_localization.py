"""Diagnose frozen cup-localization abstentions; never relax acceptance gates."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
sys.path[:0] = ['/workspace/src', '/workspace']
from pour_size_study import ROOT, OUTPUTS
from pour_guard_vision import locate
from pour_temporal_localization import consensus
from validate_pour_recording import h5py, decode


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    saved = torch.load(ROOT / 'best.pt', map_location='cpu', weights_only=False)
    config = saved['localization_consensus']
    threshold = saved['thresholds']['cup_localization_score']
    rows = []
    for path in sorted(OUTPUTS.glob('*/*SZ_A_cup_seed_*/episode.hdf5')):
        spec = json.loads((path.parent / 'parameters.json').read_text())
        history = []
        with h5py.File(path) as f:
            assert f.attrs['status'] == 'complete'
            d = f['data/demo_0']; times = d['time'][:]
            for i, t in enumerate(times):
                if t - times[0] > config['timeout_s']:
                    break
                uv, score = locate(decode(d['obs/task_image'][i], 'jpeg95'), saved['cup_bank'], device='cpu')
                history.append((uv.tolist(), score))
                if consensus(history, threshold, config['cup']) is not None:
                    break
        scores = [s for _, s in history]
        accepted = consensus(history, threshold, config['cup'])
        rows.append(dict(name=spec['name'], height_scale=spec['cup_height_scale'],
                         diameter_scale=spec['cup_diameter_scale'], frames=len(history),
                         max_score=max(scores), min_score=min(scores),
                         threshold_crossings=sum(s >= threshold for s in scores),
                         detections_span_px=np.ptp(np.asarray([u for u, _ in history]), axis=0).tolist(),
                         confirmation=accepted,
                         diagnosis='confirmed' if accepted else ('no_anchor_reaches_threshold' if max(scores) < threshold else 'temporal_consensus_rejected'),
                         source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(rows) == 9
    report = dict(scope='Frozen CPU detector on the same nine consumed low-fill pilots. No fitting, gate changes or new autonomy. Score rejection alone does not prove its underlying visual cause.',
                  threshold=threshold, temporal_settings=config['cup'], timeout_s=config['timeout_s'], cases=rows,
                  checkpoint_sha256=hashlib.sha256((ROOT/'best.pt').read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'localization-diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(threshold=threshold, diagnoses={r['name']:r['diagnosis'] for r in rows})))


if __name__ == '__main__':
    main()
