"""CPU check that new spatial features preserve old checkpoint predictions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
    sys.path[:0]=[str(a.candidate_dir),'/workspace/src']
    import torch
    import numpy as np
    from pour_settled_model import features,predict
    from pour_guard_models import CupEstimator
    from pour_guard_vision import cup_crop
    from validate_pour_recording import h5py,decode
    torch.set_num_threads(2)
    output=a.candidate_dir/'level-prefix-compatibility.json';assert not output.exists()
    baseline=Path('/workspace/learning/A-wide-20261105-v1/fit-attempts/r2/src/pour_settled_model.py')
    spec=importlib.util.spec_from_file_location('old_level_features',baseline)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert max(saved['settled_regressor']['columns'])<21
    model=CupEstimator().eval();model.load_state_dict(saved['cup_model']);rows=[]
    with torch.inference_mode():
        for path in sorted(Path('/workspace/outputs/fountain').glob('*/*XW30_A_dev_*_r1*/episode.hdf5')):
            decisions=json.loads((path.parent/'policy-decisions.json').read_text())
            with h5py.File(path) as f:
                s=json.loads(f.attrs['spec']);assert s['study_split']=='development' and f.attrs['status']=='complete'
                d=f['data/demo_0'];ids=np.linspace(len(decisions)//2,len(decisions)-1,3,dtype=int)
                for i in ids:
                    xy=decisions[i]['cup_xy'];assert xy is not None
                    image=cup_crop(decode(d['next_obs/task_image'][i],'jpeg95'),xy,saved['calibration'])
                    image=torch.as_tensor(image)[None];cnn=(model(image)+saved['cup_bias']).clamp(0,1)
                    before=old.features(image,cnn,xy);after=features(image,cnn,xy)
                    assert before.shape==(1,21) and after.shape==(1,60)
                    assert torch.equal(before,after[:,:21]) and torch.isfinite(after).all()
                    delta=float(abs(predict(after,saved['settled_regressor'])-old.predict(before,saved['settled_regressor'])).max())
                    assert delta==0.
                    rows.append(dict(name=s['name'],index=int(i),prediction_difference=delta))
    assert rows and not torch.cuda.is_initialized()
    result=dict(device='cpu',cuda_initialized=False,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((a.candidate_dir/'pour_settled_model.py').read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),frames=len(rows),rows=rows,
        scope='Compatibility only on closed development images. No accuracy, fit or reserved-test claim.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(frames=len(rows),max_prediction_difference=0.,cuda_initialized=False)))


if __name__=='__main__':main()
