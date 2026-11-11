"""Freeze a separate repair candidate; retain prior failures and all holdouts."""
import json
from pathlib import Path
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged
from perception_repair_data import entries

def main():
    assert not (ROOT/'component-selection.json').exists()
    assert not list(Path('/workspace/outputs/fountain').glob('*/*SD_A_motion_holdout_*/parameters.json'))
    data=entries();protected_unchanged()
    files=['protocol.json','protocol-v3-before-level-repair.json','plan.json','candidate.json',
           'cup-neural-localization.pt','coarse-d2.pt','rim-d2.pt','tracker-t1.pt',
           'dynamic_cup_tracker.py','fit_expanded_rim.py','perception_repair_data.py',
           'scale_rim_refiner.py','scale_rim_augmentation.py','fit_scale_rim_refiner.py',
           'train-r2-audit.json','train-r2-manual-review.json','prepare_expanded_perception.py']
    config=dict(seed=261110601,epochs=160,lr=.00015,lr_steps=[81,121],wall_seconds=2700,
                coarse_checkpoint='coarse-d2.pt',fine_initialization='rim-d2.pt',balance_motion_groups=True,
                frozen_files={n:sha(ROOT/n) for n in files},
                entries=[dict(name=s['name'],split=split,path=r['path'],sha256=r['sha256']) for s,r,split in data],
                scope='Supervised initial-frame rim heatmap repair, 207 train / 36 validation episodes; 12 initial frames each. Full temporal replay remains mandatory. Original 3 mm accuracy and absence gates unchanged; all 18 reserved fixtures unused. No simulator or active controller change.')
    with (ROOT/'rim-d3-protocol.json').open('x') as f:json.dump(config,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared=True,counts={s:sum(t==s for a,b,t in data) for s in ('train','validation')})),flush=True)

if __name__=='__main__':main()
