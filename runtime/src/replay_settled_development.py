"""CPU-only camera replay of retained failed development, never fresh tests."""
import hashlib
import json
from pathlib import Path
import torch
from validate_pour_recording import decode, h5py
from pour_guard_models import CupEstimator
from pour_guard_vision import cup_crop
from pour_settled_model import features, predict
from pour_settled_study import ROOT


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    saved = torch.load(ROOT / 'best.pt', map_location='cpu', weights_only=False)
    model = CupEstimator().eval(); model.load_state_dict(saved['cup_model'])
    result = []
    for p in Path('/workspace/outputs/fountain').glob('*/*L30_A_dev_03_r[01]*/episode.hdf5'):
        rows = json.loads((p.parent / 'samples.json').read_text())
        decisions = json.loads((p.parent / 'policy-decisions.json').read_text())
        with h5py.File(p, 'r') as f:
            name = json.loads(f.attrs['spec'])['name']
            d = f['data/demo_0']; values = []
            for i in range(0, len(rows), 2):
                a, decision = rows[i], decisions[i]
                if decision.get('fill_mode') != 'settle' or decision.get('stop_latched'):
                    continue
                xy = decision['cup_xy']
                image = torch.as_tensor(cup_crop(decode(d['obs/task_image'][i], 'jpeg95'), xy, saved['calibration']))[None]
                cnn = (model(image) + saved['cup_bias']).clamp(0, 1)
                prediction = float(predict(features(image, cnn, xy), saved['settled_regressor'])[0])
                values.append(dict(t=a['t'], prediction=prediction, prior_prediction=decision['settled_prediction'],
                    height=a['cup_fill_fraction'], cup_ml=a['cup_ml'], spill_ml=a['spilled_ml'], pause=decision['pause_count']))
        result.append(dict(name=name, frames=values))
    out = dict(checkpoint_sha256=hashlib.sha256((ROOT / 'best.pt').read_bytes()).hexdigest(),
        note='CPU replay on consumed failed development only; contemporaneous height is diagnostic, not the final-settled target.', cases=result)
    (ROOT / 'development-replay.json').write_text(json.dumps(out, indent=2) + '\n')
    for case in result:
        for pause in sorted({x['pause'] for x in case['frames']}):
            tail = [x for x in case['frames'] if x['pause'] == pause][-4:]
            print(case['name'], 'pause', pause, json.dumps(tail))


if __name__ == '__main__':
    main()
