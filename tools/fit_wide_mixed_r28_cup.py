"""Calibrate pair 11's coarse cup heatmap on native, displaced and erased views."""
import sys
from fit_wide_mixed_r28_bottles import CUP, HISTORY, PARENT, ROOT, STUDY, save, sha
sys.path.insert(0,str(STUDY/"src"))

# Load the exact deployed tracker before the historical replay modules.
from pour_cup_outline import OutlineCupTracker
from diagnose_r22_startup import native_trials
from fit_wide_mixed_r20 import state_for
from fit_wide_mixed_r25_cup import fit
from replay_r16_startup import measure
import torch


def check(revisions,original,candidate):
    rows = []
    for revision in revisions:
        for case,frame in native_trials(ROOT/f"wide-mixed-r{revision}",{11}):
            rows.append(measure(case,frame,OutlineCupTracker,STUDY,("11_normal","11_swap"),41,
                                parent_config=original,candidate_config=candidate))
    positive = [r[k] for r in rows for k in ("candidate","translated_left","translated_right")]
    negative = [r[k] for r in rows for k in ("flat","background")]
    passed = all(v is not None and v["error_mm"]<=2.5 for v in positive) and not any(negative)
    return dict(passed=passed,positives=len(positive),negatives=len(negative),cases=rows)


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    checkpoint,state=state_for(CUP,PARENT)
    cfg=state["sized_cup_perception"]
    assert cfg["outline_cup_localization"]
    train=(26,"27b")
    fitted,training=fit(cfg,revisions=train,pair=11,seed=26112841)
    original=cfg["tracker"]["coarse_state"]
    attempts=[]
    for alpha in (.125,.25,.375,.5,.625,.75,.875,1.):
        weights={k:torch.lerp(v,fitted[k],alpha) if v.is_floating_point() else v for k,v in original.items()}
        candidate=dict(cfg,tracker=dict(cfg["tracker"],coarse_state=weights))
        fitting=check(train,cfg,candidate)
        attempts.append(dict(alpha=alpha,**fitting))
        if fitting["passed"]:
            break
    save(STUDY/"cup-calibration.json",dict(passed=fitting["passed"],attempts=attempts))
    assert fitting["passed"],"No blend passed fitting-view checks"
    target,=[r for r in fitting["cases"] if r["name"].endswith("r27b_test_11_swap")]
    assert target["parent"] is None
    validation=check([r for r in HISTORY if r not in train],cfg,candidate)
    path=STUDY/"cup-coarse-11.pt"
    torch.save(weights,path)
    save(STUDY/"cup-startup-replay.json",dict(passed=validation["passed"],checkpoint=checkpoint,training=training,
         alpha=alpha,fitting=fitting,validation=validation,artifact_sha256=sha(path),parent_plan_sha256=sha(PARENT/"plan.json")))
    print("Cup replay",alpha,validation["passed"],flush=True)
    assert validation["passed"]


if __name__=="__main__":
    main()
