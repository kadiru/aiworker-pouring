"""Pre-test controller calibration; learned tensors and motor paths stay fixed."""
import hashlib,json,shutil
from pathlib import Path
import torch
from pour_guard_study import ROOT


def main():
    assert not list(Path('/workspace/outputs/fountain').glob('*/*G_A_test*/parameters.json')),'Do not calibrate after tests start'
    p=ROOT/'best.pt';b=torch.load(p,map_location='cpu',weights_only=False);old_sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert b['thresholds']['full_settled_target']<.96,'Calibration already applied'
    shutil.copy2(p,ROOT/f'before-full-margin-{old_sha[:8]}.pt')
    before=dict(b['thresholds']);b['thresholds'].update(full_settled_target=.96,settle_seconds=1.2,trim_increment=.015)
    b.setdefault('training_sources',dict(b['sources']))
    b['sources']={n:hashlib.sha256(Path('/workspace/src',n).read_bytes()).hexdigest() for n in b['sources']}
    note=dict(parent_checkpoint_sha256=old_sha,before=before,after=b['thresholds'],
        basis='Consumed development grasp regression: no spill, final physical height.89203 vs CNN.94241. Six-point predicted-fill margin guards this observed five-point overestimate without changing the >=90% physical scoring goal. Relative trim increments prevent zero-addition pause loops.',
        learned_weights_and_motor_paths_unchanged=True,
        calibration_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    b['full_calibration']=note
    torch.save(b,ROOT/'best.partial.pt');(ROOT/'best.partial.pt').replace(p)
    after=torch.load(p,map_location='cpu',weights_only=False);old=torch.load(ROOT/f'before-full-margin-{old_sha[:8]}.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(old[k][n],after[k][n]) for k in ('cup_model','source_detector') for n in old[k])
    assert old['pour_primitive']==after['pour_primitive'] and old['primitive']==after['primitive']
    note['selected_checkpoint_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();note['cuda_initialized']=torch.cuda.is_initialized()
    (ROOT/'full-calibration.json').write_text(json.dumps(note,indent=2)+'\n');print(json.dumps(note,indent=2))


if __name__=='__main__':main()
