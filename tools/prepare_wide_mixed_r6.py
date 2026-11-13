"""Seal new swapped layouts and targeted wet development after both fits pass."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
STUDY=ROOT/"wide-mixed-r6"
sys.path.insert(0,str(STUDY/"src"))
from fit_wide_mixed_r6 import PARENT, SLOW, folder, read, save, sha
from pour_size_autonomy import MIDFILL_RGB_KNOTS, settled_rgb_weight


def midfill_calibration(case):
    trial=folder(case)
    labels={round(x["t"],6):x["cup_fill_fraction"] for x in read(trial/"samples.json")}
    rows=[]
    for frame in read(trial/"policy-decisions.json"):
        t=round(frame["t"],6)
        if t<case["first_zero_source_s"] or not frame["sized_cup_perception"]["valid"]:
            continue
        values=frame["sized_cup_perception"]["filtered"]
        rows.append((values["rgb"],values["color"],labels[t]))
        if frame["stop_latched"]:
            break
    fit,validate=np.asarray(rows[:32]),np.asarray(rows[32:])
    assert len(validate)>=40
    rgb,_,target=fit.T
    w=np.array([settled_rgb_weight(v,MIDFILL_RGB_KNOTS) for v in rgb])
    bias=float(np.clip(w@(rgb-target)/(w@w),0.,.035))
    rgb,color,target=validate.T
    corrected=rgb-bias*np.array([settled_rgb_weight(v,MIDFILL_RGB_KNOTS) for v in rgb])
    before=float(np.max(abs(rgb-color)))
    after=float(np.max(abs(corrected-color)))
    assert 0.<bias<=.035 and after<.035 and abs(corrected-target).mean()<abs(rgb-target).mean()
    return dict(bias=bias,train_frames=len(fit),validation_frames=len(validate),
                max_gap_before=before,max_gap_after=after,
                recording_sha256=case["recording_sha256"])


def main():
    torch.set_num_threads(2)
    parent=PARENT/"plan.json"
    parent_result=PARENT/"completion.json"
    plan=read(parent)
    data=read(parent_result)
    assert data["plan_sha256"]==sha(parent) and data["test"]["trials"]==30
    neck_path, wrist_path=STUDY/"neck-fusion.json",STUDY/"wrist-fit.json"
    neck,wrist=read(neck_path),read(wrist_path)
    assert neck["passed"] and wrist["passed"]
    neck_weight,wrist_weight=STUDY/"neck-heatmap.pt",STUDY/"exact-empty-wrist.pt"
    assert sha(neck_weight)==neck["checkpoint_sha256"] and sha(wrist_weight)==wrist["checkpoint_sha256"]
    heatmap=dict(torch.load(neck_weight,map_location="cpu",weights_only=False),**neck["operating_point"])
    depletion=torch.load(wrist_weight,map_location="cpu",weights_only=False)
    cases=data["test_cases"]
    midfill=midfill_calibration(next(r for r in cases if r["name"]==SLOW[0]))
    spill_combinations={(tuple(c["cup_size"]),tuple(c["bottle_size"])) for c in cases if c["measured_loss_particles"]}
    assert len(spill_combinations)==2
    specs={s["name"]:s for s in plan["test"]}
    plan["checkpoints"]={}
    checkpoint_map={}
    rng=np.random.default_rng(2611130600)
    deltas=rng.uniform(.0012,.0028,(15,2,2))*rng.choice((-1.,1.),(15,2,2))

    def updated_checkpoint(spec):
        source=spec["checkpoint"]
        if source not in checkpoint_map:
            previous=read(parent)
            assert sha(source)==previous["checkpoints"][source]
            saved=torch.load(source,map_location="cpu",weights_only=False)
            saved["source_heatmap"]=heatmap
            if spec["bottle_height_scale"]>1.:
                saved["trained_empty_confirmation"]=depletion
            cup=spec["cup_height_scale"],spec["cup_diameter_scale"]
            bottle=spec["bottle_height_scale"],spec["bottle_diameter_scale"]
            if cup==(.9,1.):
                saved["sized_cup_perception"]["midfill_rgb_calibration"]=dict(
                    geometry=spec["declared_cup_geometry"],bias=midfill["bias"])
            if (cup,bottle) in spill_combinations:
                saved["tail_setback_m"]=.008
            saved["model_version"]="visual_guard_wide_mixed_r6"
            dest=STUDY/Path(source).name
            assert not dest.exists()
            torch.save(saved,dest)
            checkpoint_map[source]=str(dest)
            plan["checkpoints"][str(dest)]=sha(dest)
        return checkpoint_map[source]

    test=[]
    for i,original in enumerate(plan["test"]):
        spec=copy.deepcopy(original)
        a,b=deltas[spec["pair_index"]][::(-1 if spec["swapped"] else 1)]
        spec["cup_world"][:2]=(np.asarray(spec["cup_world"][:2])+a).tolist()
        diff=b-a
        spec["source_xy"]=(np.asarray(spec["source_xy"])+[diff[1],-diff[0]]).tolist()
        spec.update(name=original["name"].replace("_r1_test_","_r6_test_"),seed=2611130700+i,
                    checkpoint=updated_checkpoint(original))
        test.append(spec)
    check_names=(*SLOW,"SA_A_wide_mixed_r1_test_12_normal","SA_A_wide_mixed_r1_test_06_swap")
    development=[]
    for i,name in enumerate(check_names):
        spec=copy.deepcopy(specs[name])
        spec.update(name=name.replace("_r1_test_","_r6_development_"),study_split="development",
                    seed=2611130800+i,checkpoint=updated_checkpoint(spec))
        development.append(spec)
    plan.update(development=development,test=test,test_shards=[[s["name"] for s in test[i:i+10]] for i in (0,10,20)],
                controller_scope="Retrained neck and exact-empty wrist on wider layouts and all fifteen exact swap pairs",
                development_empty_lag_limit_s=2.,development_empty_lag_cases=[s["name"] for s in development[:2]])
    plan["prerequisite_reports"].update({str(parent):sha(parent),str(parent_result):sha(parent_result),
                                         str(neck_path):sha(neck_path),str(wrist_path):sha(wrist_path)})
    report=STUDY/"empty-replay.json"
    save(report,dict(empty_fit_sha256=sha(wrist_path),neck_fit_sha256=sha(neck_path),midfill=midfill,
                     stop_replay=wrist["replay"],tail_setback_m=.008,
                     tail_setback_combinations=sorted(spill_combinations)))
    plan["empty_confirmation_replay_sha256"]=sha(report)
    shutil.copy2(PARENT/"calibration.json",STUDY/"calibration.json")
    save(STUDY/"plan.json",plan)
    print(json.dumps(dict(development=len(development),test=len(test),midfill=midfill)),flush=True)


if __name__=="__main__":
    main()
