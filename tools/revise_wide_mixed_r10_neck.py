"""Seal the startup retrain using the same 30 untouched test fixtures."""
from pathlib import Path
import shutil
import sys

import torch

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/"wide-mixed-r10"
STUDY=ROOT/"wide-mixed-r10b"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import sha
from fit_wide_mixed_r6 import read,save


def main():
    torch.set_num_threads(1)
    before=PARENT/"plan.json"
    results=PARENT/"development-results.json"
    plan=read(before)
    old=read(results)
    assert old["plan_sha256"]==sha(before)
    failed=[x for x in old["cases"] if not x["strict_pass"]]
    assert len(failed)==1 and failed[0]["name"].endswith("12_normal")
    assert failed[0]["stop_reason"]=="localization_unconfirmed"
    report=STUDY/"neck-fusion.json"
    weights=STUDY/"neck-heatmap.pt"
    fit=read(report)
    assert fit["passed"] and sha(weights)==fit["checkpoint_sha256"]
    trained=torch.load(weights,map_location="cpu",weights_only=False)
    target=next(s for s in plan["development"] if s["name"]==failed[0]["name"])
    bottle=target["declared_bottle_geometry"]
    checkpoints=plan["checkpoints"]
    changed={}
    predecessors={}
    for part in ("test","development"):
        for row in plan[part]:
            source=row["checkpoint"]
            if row["declared_bottle_geometry"]==bottle:
                if source not in changed:
                    assert sha(source)==checkpoints.pop(source)
                    state=torch.load(source,map_location="cpu",weights_only=False)
                    state["source_heatmap"]=trained
                    dest=STUDY/Path(source).name
                    assert not dest.exists()
                    torch.save(state,dest)
                    changed[source]=str(dest)
                    checkpoints[str(dest)]=sha(dest)
                row["checkpoint"]=changed[source]
            if part=="development":
                previous=row["name"]
                row["name"]=previous.replace("_r10_development_","_r10b_development_")
                predecessors[row["name"]]=previous
    plan["clearance_predecessors"]=predecessors
    plan["development_empty_lag_cases"]=[s.replace("_r10_development_","_r10b_development_")
                                         for s in plan["development_empty_lag_cases"]]
    for filename in ("calibration.json","empty-replay.json","spill-trace.json"):
        shutil.copy2(PARENT/filename,STUDY/filename)
    plan["prerequisite_reports"].update({str(p):sha(p) for p in (before,results,report,STUDY/"spill-trace.json")})
    plan["controller_scope"] += " and normal-12 startup retrain for the short standard-width bottle"
    save(STUDY/"plan.json",plan)


if __name__=="__main__":
    main()
