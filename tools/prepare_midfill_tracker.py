"""Freeze expanded tracker before replaying every development case."""
import json
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged

def main():
    protected_unchanged()
    old=read(ROOT/'tracker-t1-protocol.json')
    report=read(ROOT/'rim-d4.json')
    assert report['checkpoint_sha256']==sha(ROOT/'rim-d4.pt')
    import torch
    parent=torch.load(ROOT/'tracker-t1.pt',map_location='cpu',weights_only=False)
    files=['protocol.json','plan.json','candidate.json','rim-d4.json','rim-d4.pt',
           'rim-d4-protocol.json','coarse-d2.pt','cup-neural-localization.pt',
           'tracker-t1.pt','dynamic_cup_tracker.py','perception_midfill_data.py',
           'evaluate_midfill_tracker.py','prepare_midfill_tracker.py','train-r3-audit.json',
           'train-r3-manual-review.json']
    cfg=dict(round='t3',settings=old['settings'],threshold=parent['threshold'],consensus=parent['consensus'],
             frozen_files={n:sha(ROOT/n) for n in files},gates=read(ROOT/'protocol.json')['gates'],
             scope='Updated learned rim initial localization; original coarse detector, confidence threshold, eight-frame confirmation and lower-outline temporal tracking unchanged. All 279 reviewed development episodes required.',
             accuracy_policy=old['accuracy_policy'])
    with (ROOT/'tracker-t3-protocol.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared='t3',rim_initial_checks_passed=report['passed'])),flush=True)

if __name__=='__main__':main()
