"""Verify the deployed surface readout and seal four targeted and thirty unrun layouts."""
from collections import defaultdict
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT/"wide-mixed-r27"
STUDY = ROOT/"wide-mixed-r27b"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import settled_pair, settled_surface_bound, sha, surface_features
from fit_wide_mixed_r6 import read, save
from prepare_wide_mixed_r17 import identical


def verify_surface(model, fit, thresholds, tolerance):
    path = STUDY/"settled-crops.npz"
    metadata = STUDY/"settled-crops.json"
    assert sha(path)==fit["frames_sha256"] and sha(metadata)==fit["metadata_sha256"]
    rows = read(metadata)
    with np.load(path) as archive:
        crops = torch.from_numpy(archive["images"]).cuda()
    weights = model["weights"].cuda()
    # Match single-frame inference, including float32 arithmetic on the 4070.
    with torch.inference_mode():
        predictions = [(float(surface_features(c[None],model["cell"])[0] @ weights)+model["intercept"]) for c in crops]
    groups = defaultdict(list)
    for i,row in enumerate(rows):
        groups[row["name"],row["pause"]].append(i)
    checked=[]
    for window in fit["checks"]:
        indices = groups[window["name"],window["pause"]][-6:]
        estimates = [predictions[i] for i in indices]
        lower = settled_surface_bound(estimates)
        assert abs(float(np.quantile(estimates,.25))-window["predicted_low"]) < 1e-4
        low,stable,full = settled_pair([(rows[i]["rgb"],rows[i]["color"]) for i in indices],
                                       thresholds["full_settled_target"],tolerance,thresholds["settled_stability_std"])
        confirmed = full and lower is not None and lower>=model["minimum"]
        assert not confirmed or window["true_median"]>=.90
        checked.append(dict(name=window["name"],pause=window["pause"],split=window["split"],surface=lower,
                            rgb_color_low=low,rgb_color_stable=stable,confirmed=confirmed,truth=window["true_median"]))
    under, = [c for c in checked if "r26_test" in c["name"] and c["pause"]==5]
    full, = [c for c in checked if "r27_development" in c["name"] and c["pause"]==5]
    assert not under["confirmed"] and under["truth"]<.90 and full["confirmed"] and full["truth"]>=.90
    # Some earlier pauses never met the existing two-camera threshold. The
    # historical episode must still confirm once those readings have settled.
    terminal = {c["name"]:c for c in checked if c["split"]=="validation"}
    assert all(c["confirmed"] for c in terminal.values() if c["truth"]>=.908)
    assert all(c["surface"] is not None and c["surface"]>=model["minimum"]
               for c in checked if c["split"]=="validation" and c["truth"]>=.908)
    save(STUDY/"surface-replay.json",dict(passed=True,minimum=model["minimum"],windows=checked,
         training_windows=fit["train_windows"],validation_windows=fit["validation_windows"],artifact_sha256=fit["artifact_sha256"]))


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    plan = read(PARENT/"plan.json")
    dev = read(PARENT/"development-results.json")
    assert dev["plan_sha256"]==sha(PARENT/"plan.json") and dev["strict_passes"]==3
    failed, = [r for r in dev["cases"] if not r["strict_pass"]]
    assert failed["name"].endswith("09_swap") and failed["stop_reason"]=="fill_unconfirmed"
    assert not list(PARENT.glob("test-*-results.json"))
    path = next(s["checkpoint"] for s in plan["development"] if s["name"].endswith("09_swap"))
    assert all(sha(p)==digest for p,digest in plan["checkpoints"].items())
    before = torch.load(path,weights_only=False,map_location="cpu")
    original_plan=read(ROOT/"wide-mixed-r26"/"plan.json")
    original=next(s["checkpoint"] for s in original_plan["test"] if s["name"].endswith("09_swap"))
    assert sha(original)==original_plan["checkpoints"][original]
    previous=torch.load(original,weights_only=False,map_location="cpu")["thresholds"]
    limits=("full_settled_target","fitted_full_at_pulse_limit")
    thresholds=dict(before["thresholds"],**{k:previous[k] for k in limits})
    fit=read(STUDY/"surface-fit.json")
    artifact=STUDY/"surface-readout.pt"
    assert fit["passed"] and sha(artifact)==fit["artifact_sha256"]
    model=torch.load(artifact,weights_only=False,map_location="cpu")
    cfg=before["sized_cup_perception"]
    verify_surface(model,fit,thresholds,cfg["agreement_limit"])
    candidate=dict(before,thresholds=thresholds,sized_cup_perception=dict(cfg,settled_surface_readout=model))
    dest=STUDY/Path(path).name
    assert not dest.exists()
    torch.save(candidate,dest)
    actual=torch.load(dest,weights_only=False,map_location="cpu")
    assert identical(actual,candidate)
    del actual["sized_cup_perception"]["settled_surface_readout"]
    for k in limits:
        actual["thresholds"][k]=before["thresholds"][k]
    assert identical(actual,before)
    sources={p.name:sha(p) for p in (STUDY/"src").glob("*.py")}
    parent_sources=read(PARENT/"geometry.json")["sources"]
    assert sources.keys()==parent_sources.keys()
    assert {k for k in sources if sources[k]!=parent_sources[k]}=={"pour_size_autonomy.py","pour_guard_actor.py"}
    before_sha=plan["checkpoints"].pop(path)
    plan["checkpoints"][str(dest)]=sha(dest)
    frozen=STUDY/"frozen-controller.json"
    save(frozen,dict(passed=True,original=path,parent_sha256=before_sha,candidate_sha256=sha(dest),
                     runtime_sources=sources,unchanged_checkpoints=len(plan["checkpoints"])-1,
                     thresholds={k:thresholds[k] for k in limits},parent_plan_sha256=dev["plan_sha256"]))
    for part,seed in (("development",2611288100),("test",2611288200)):
        for i,spec in enumerate(plan[part]):
            spec.update(name=spec["name"].replace("_r27_","_r27b_"),seed=seed+i)
            if spec["checkpoint"]==path:
                spec["checkpoint"]=str(dest)
    plan["test_shards"]=[[n.replace("_r27_","_r27b_") for n in group] for group in plan["test_shards"]]
    assert set(plan["checkpoints"])=={s["checkpoint"] for s in plan["test"]}
    assert len(plan["development"])==4 and len(plan["test"])==30 and sum(s["swapped"] for s in plan["test"])==15
    plan["controller_scope"]="Pair 09 settled RGB surface readout with agreement, trim and near-full confirmation."
    for report in (PARENT/"plan.json",PARENT/"development-results.json",STUDY/"surface-fit.json",STUDY/"surface-replay.json",frozen):
        plan["prerequisite_reports"][str(report)]=sha(report)
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    save(STUDY/"plan.json",plan)


if __name__=="__main__":
    main()
