"""Freeze expanded tracker before replaying every development case."""
import json
from fit_dynamic_rim import ROOT,read,sha,protected_unchanged

def main():
    protected_unchanged()
    old=read(ROOT/'tracker-t1-protocol.json')
    report=read(ROOT/'rim-d3.json')
    assert report['checkpoint_sha256']==sha(ROOT/'rim-d3.pt')
    import torch
    parent=torch.load(ROOT/'tracker-t1.pt',map_location='cpu',weights_only=False)
    files=['protocol.json','plan.json','candidate.json','rim-d3.json','rim-d3.pt',
           'rim-d3-protocol.json','coarse-d2.pt','cup-neural-localization.pt',
           'tracker-t1.pt','dynamic_cup_tracker.py','perception_repair_data.py',
           'evaluate_expanded_tracker.py','prepare_expanded_tracker.py','train-r2-audit.json',
           'train-r2-manual-review.json']
    cfg=dict(round='t2',settings=old['settings'],threshold=parent['threshold'],consensus=parent['consensus'],
             frozen_files={n:sha(ROOT/n) for n in files},gates=read(ROOT/'protocol.json')['gates'],
             scope='Updated learned rim initial localization; original coarse detector, confidence threshold, eight-frame confirmation and lower-outline temporal tracking unchanged. All 243 reviewed development episodes required.',
             accuracy_policy=old['accuracy_policy'])
    with (ROOT/'tracker-t2-protocol.json').open('x') as f:json.dump(cfg,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared='t2',rim_initial_checks_passed=report['passed'])),flush=True)

if __name__=='__main__':main()
