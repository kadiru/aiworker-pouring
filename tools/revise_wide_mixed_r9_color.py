"""Calibrate the alternate color estimate for the repaired short-bottle pour."""
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT/"wide-mixed-r9"
STUDY = ROOT/"wide-mixed-r9b"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import (MAX_SETTLED_RGB_BIAS, calibrated_rgb_height, settled_rgb_weight,
                                sha, validate_rgb_calibration)
from fit_wide_mixed_r6 import folder, read, save

KNOTS = [.630,.667,.692,.740]
TARGET = "02_swap"


def frames(case):
    path = folder(case)
    assert sha(path/"episode.hdf5")==case["recording_sha256"]
    samples={round(item["t"],6):item["cup_fill_fraction"] for item in read(path/"samples.json")}
    rows=[]
    for f in read(path/"policy-decisions.json"):
        level=f.get("sized_cup_perception") or {}
        time=round(f["t"],6)
        if (time in samples and level.get("valid") and f["fill_mode"]=="coarse"
                and f["own_pour_phase_deg"]>=115.):
            v=level.get("guard_filtered",level["filtered"])
            rows.append((v["rgb"],v["color"],samples[time]))
        if f["stop_latched"]:
            break
    assert len(rows)>=18
    return rows


def train(geometry, target, parent):
    training=[]
    checks=[]
    for case in (target,parent):
        rows=frames(case)
        fit=[r for i,r in enumerate(rows[:-8]) if i%12<6]
        validation=[r for i,r in enumerate(rows) if i>=len(rows)-8 or i%12>=6]
        training.extend(fit)
        checks.append((case,validation))
    _,color,truth=torch.tensor(training,device="cuda",dtype=torch.float64).T
    basis=torch.tensor([settled_rgb_weight(v,KNOTS) for v in color.cpu().numpy()],device="cuda",dtype=torch.float64)
    bias=float(torch.clamp((basis*(color-truth)).sum()/basis.square().sum(),0.,MAX_SETTLED_RGB_BIAS))
    calibration=dict(geometry=geometry,knots=KNOTS,bias=bias)
    validate_rgb_calibration(calibration,geometry)
    reports=[]
    for case,rows in checks:
        rgb,color,truth=np.asarray(rows).T
        new=np.asarray([calibrated_rgb_height(v,calibration) for v in color])
        reports.append(dict(name=case["name"],recording_sha256=case["recording_sha256"],
                            validation_frames=len(rows),last_eight_gap_before=float(abs(rgb-color)[-8:].max()),
                            last_eight_gap_after=float(abs(rgb-new)[-8:].max()),
                            color_mae_before=float(abs(color-truth).mean()),color_mae_after=float(abs(new-truth).mean())))
    passed=all(x["last_eight_gap_after"]<.035 and x["color_mae_after"]<x["color_mae_before"] for x in reports)
    result=dict(passed=passed,training_frames=len(training),calibration=calibration,cases=reports,
                device=torch.cuda.get_device_name(0))
    save(STUDY/"draining-color-fit.json",result)
    assert passed,result
    return calibration


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name(0)
    torch.set_num_threads(1)
    parent=PARENT/"plan.json"
    plan=read(parent)
    current=PARENT/"development-results.json"
    recent=read(current)
    assert recent["plan_sha256"]==sha(parent)
    previous=ROOT/"wide-mixed-r8"/"completion.json"
    misses=[r for r in recent["cases"] if not r["strict_pass"]]
    assert len(misses)==1 and misses[0]["name"].endswith(TARGET)
    target=misses[0]
    assert target["measured_loss_particles"]==0 and target["failure_reasons"]==["sized_perception_stop_unconfirmed"]
    original=next(r for r in read(previous)["test_cases"] if r["name"].endswith(TARGET))
    spec=next(s for s in plan["development"] if s["name"]==target["name"])
    calibration=train(spec["declared_cup_geometry"],target,original)
    path=spec["checkpoint"]
    assert sha(path)==plan["checkpoints"].pop(path)
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    checkpoint["sized_cup_perception"]["draining_color_calibration"]=calibration
    dest=STUDY/Path(path).name
    assert not dest.exists()
    torch.save(checkpoint,dest)
    plan["checkpoints"][str(dest)]=sha(dest)
    for row in plan["development"]+plan["test"]:
        if row["checkpoint"]==path:
            row["checkpoint"]=str(dest)
    predecessors={}
    for row in plan["development"]:
        old=row["name"]
        row["name"]=old.replace("_r9_development_","_r9b_development_")
        predecessors[row["name"]]=old
    plan["clearance_predecessors"]=predecessors
    plan["development_empty_lag_cases"]=[s.replace("_r9_development_","_r9b_development_")
                                         for s in plan["development_empty_lag_cases"]]
    for filename in ("calibration.json","empty-replay.json","near-full-fit.json","spill-trace.json"):
        shutil.copy2(PARENT/filename,STUDY/filename)
    plan["prerequisite_reports"].update({str(p):sha(p) for p in
                                        (parent,current,STUDY/"near-full-fit.json",STUDY/"spill-trace.json",
                                         STUDY/"draining-color-fit.json")})
    save(STUDY/"plan.json",plan)


if __name__=="__main__":
    main()
