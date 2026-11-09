"""Recheck the unchanged rim checkpoint against the expanded collection."""
import copy,json,sys
from pathlib import Path
import torch
sys.path[:0]=['/workspace/src','/workspace']
from fit_scale_rim_refiner import collect,evaluate,windows
from fit_scale_rim_calibration import corrected
from pour_cup_locator import CupLocator
from scale_rim_refiner import RimRefiner
from scale_data_admission import collection_paths
from fit_scale_level import ROOT,read,sha,write

@torch.inference_mode()
def main():
    assert not (ROOT/'locator-revalidation-r10.json').exists()
    m=read(ROOT/'candidate.json');ap,rp=collection_paths(ROOT,m);old=read(ROOT/'collection-r1/cup-neural-localization.json')
    assert sha(ROOT/'cup-neural-localization.pt')==old['checkpoint_sha256'] and sha(ROOT/'scale_rim_refiner.py')==old['module_sha256']
    assert m['collection_round']=='r2' and read(rp)['accepted'] and read(rp)['audit_sha256']==sha(ap)
    assert torch.cuda.device_count()==1 and '4070' in torch.cuda.get_device_name();torch.set_num_threads(4)
    artifact=torch.load(ROOT/'cup-neural-localization.pt',map_location='cpu',weights_only=False);bank=artifact['bank']
    coarse=CupLocator().cuda().eval();coarse.load_state_dict(bank['coarse_state']);fine=RimRefiner().cuda().eval();fine.load_state_dict(bank['state'])
    plan,data,cal=collect(coarse,'r10-revalidation');assert cal==artifact['calibration']
    results={s:corrected(evaluate(fine,a,cal,bank['max_correction_pixels']),a,cal,bank) for s,a in data.items()}
    win=[]
    for s,a in data.items():win+=windows(results[s],a['meta'],cal,artifact['threshold'],artifact['consensus'])
    gate=plan['component_gates'];margin=min(r['minimum_positive_score'] for r in results.values())-max(r['maximum_synthetic_absent_score'] for r in results.values())
    passed=all(r['max_error_mm']<gate['localization_max_error_mm'] for r in results.values()) and margin>gate['localization_negative_margin'] and all(w['confirmed'] and w['error_mm']<gate['localization_max_error_mm'] for w in win)
    report=dict(passed=passed,scope='all_nine_declared_cup_sizes',revalidation_only=True,learned_parameters=0,
        plan_sha256=m['plan_sha256'],audit_sha256=sha(ap),manual_review_sha256=sha(rp),checkpoint_sha256=sha(ROOT/'cup-neural-localization.pt'),
        original_fit_report_sha256=sha(ROOT/'collection-r1/cup-neural-localization.json'),module_sha256=sha(ROOT/'scale_rim_refiner.py'),
        training_check=results['train'],selected=results['validation'] if passed else None,best_attempt=results['validation'],temporal_windows=win,absence_margin=margin,
        counts={s:dict(images=len(a['meta']),episodes=len({r['name'] for r in a['meta']})) for s,a in data.items()},
        source_sha256=sha(Path(__file__)),method='Unchanged checkpoint, threshold and consensus evaluated on all expanded training/validation frames; no fitting or reserved images.')
    write('locator-revalidation-r10.json',report);write('cup-neural-localization.json',report)
    write('scope-selection.json',dict(passed=passed,scope=report['scope'],checkpoint_sha256=report['checkpoint_sha256'],module_sha256=report['module_sha256'],
        max_validation_error_mm=results['validation']['max_error_mm'],max_training_error_mm=results['train']['max_error_mm'],threshold=artifact['threshold'],revalidation_only=True))
    m.update(accepted_locator=passed,stage='locator_revalidated' if passed else 'locator_revalidation_failed');write('candidate.json',m)
    if passed:
        protocol=read(ROOT/'level-refit-r9-plan.json');protocol.update(round='r10',plan_sha256=m['plan_sha256'],audit_sha256=sha(ap),manual_review_sha256=sha(rp),
            fine_initialization='collection-r1/static-level-candidate.pt',fine_initialization_sha256=sha(ROOT/'collection-r1/static-level-candidate.pt'),
            training_episodes=126,validation_episodes=18,expected_updates=5280,
            unchanged=['Original 108 episodes and splits', 'all original thresholds', 'nine unused reserved scenes', 'active controller and runtime sources'],
            training='Continue supervised fine-tuning of all CupEstimator and context-correction weights from the archived rejected r9 checkpoint; AdamW, equal episode/size sampling, MSE. Image augmentation leaves the physical level label unchanged.',
            selection='Only candidates passing every original per-size train/validation gate; minimize worst per-size validation MAE + 0.3*near-full max error; retain the earlier epoch on an exact tie.',
            frozen_locator={n:sha(ROOT/n) for n in ('cup-neural-localization.pt','cup-neural-localization.json','scope-selection.json','scale_rim_refiner.py')},
            rationale='Same CNN method, initialized from rejected r9 best attempt, now with all four training layouts per size at an additional 0.35 seed cut. Original validation and reserved fixtures unchanged.')
        write('level-refit-r10-plan.json',protocol)
        for n in ('static-level-fit.json','static-level-candidate.pt','level-localization-check.json'):
            assert sha(ROOT/n)==sha(ROOT/'collection-r1'/n);(ROOT/n).unlink()
    print(json.dumps(dict(passed=passed,training_max_mm=results['train']['max_error_mm'],validation_max_mm=results['validation']['max_error_mm'],temporal_confirmed=sum(w['confirmed'] for w in win))),flush=True)
if __name__=='__main__':main()
