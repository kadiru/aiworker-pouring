"""Add known geometry to landmark feature extraction, retaining all prior evidence."""
import json
from fit_dynamic_rim import ROOT, read, sha, protected_unchanged


def main():
    protected_unchanged()
    assert not (ROOT/'component-selection.json').exists()
    assert not read(ROOT/'pair-r14.json')['passed']
    assert read(ROOT/'conditioned-landmark-contracts.json')['passed']
    previous=read(ROOT/'pair-r14-protocol.json')
    assert all(sha(ROOT/n)==h for n,h in previous['frozen_files'].items())
    files=list(previous['frozen_files'])+['pair-r14.json','pair-r14-protocol.json',
          'pair-r14-rgb.pt','pair-r14-color.pt','pair-r14-rgb.json','pair-r14-color.json',
          'landmark_level_model_v2.py','test_conditioned_landmarks.py','conditioned-landmark-contracts.json',
          'prepare_conditioned_landmarks.py','fit_conditioned_landmarks.py']
    cfg=dict(previous)
    cfg.update(round='r15',seed=261110617,lr=.0002,
               frozen_files={n:sha(ROOT/n) for n in files},
               initialization=dict(rgb='pair-r14-rgb.pt',color='pair-r14-color.pt'),
               method=previous['method']+' Known height/diameter/aspect channels additionally condition feature extraction; new-channel weights initialize to zero from the respective R14 model. Epoch zero is evaluated to retain a passing baseline. TF32 convolutions disabled locally for numerical parity.')
    with (ROOT/'pair-r15-protocol.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared='r15',counts=cfg['counts'])),flush=True)


if __name__=='__main__':main()
