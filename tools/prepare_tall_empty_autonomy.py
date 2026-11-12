"""Replay the tall-bottle gate and seal one verification plus six fresh fixtures."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
HOST = Path("/home/kadir/aiworker-pouring-4070")
STUDY = ROOT/"tall-empty-r1"
sys.path.insert(0, str(STUDY/"src"))
from bottle_pour import initial_points
from pour_size_autonomy import seeded_prefill, sha
from pour_source_watchdog import TallBottleEmptyGate


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    with path.open("x") as file:
        json.dump(data,file,indent=2,allow_nan=False)
        file.write("\n")


def native_trial(path):
    return Path("/workspace")/Path(path).relative_to(HOST)


def replay(report):
    rows = []
    for case in report["cases"]:
        if case["bottle_geometry"]["lip_height_m"] < .14:
            continue
        folder = native_trial(case["path"])
        decisions, samples = (folder/"policy-decisions.json",folder/"samples.json")
        labels = {round(x["t"],6):x["bottle_ml"] for x in read(samples)}
        scale = read(folder/"parameters.json")["declared_cup_geometry"]["legacy_brim_nominal_ml"]
        zero = next((t for t,ml in labels.items() if ml==0.),None)
        gate, valid, candidate, at_candidate = TallBottleEmptyGate(),0,None,None
        baseline = None
        for frame in read(decisions):
            if frame["stop_latched"]:
                baseline = frame["t"]
                break
            cup = frame.get("sized_cup_perception") or {}
            views = frame.get("empty_views") or {}
            valid = valid+1 if cup.get("valid") and views else 0
            fast = gate.update(held=bool(views.get("encoder_hold")),
                               level_ready=valid>=9 and cup.get("agreement",False),
                               present=frame["source_present_probability"],
                               wrist_empty=views.get("wrist_crop_empty",0.),
                               scene_empty=views.get("raw_empty",0.),
                               tilt=frame["estimated_tilt_from_joints"],
                               flow_ml_s=frame["visual_flow_fraction_s"]*scale)
            if fast and candidate is None:
                candidate = frame["t"]
                at_candidate = labels[round(candidate,6)]
        positive = case["stop_reason"]=="bottle_empty"
        assert baseline is not None
        if positive:
            assert candidate is not None and zero is not None and at_candidate==0.
            assert 0. <= candidate-zero <= 2. and candidate < baseline
        else:
            assert case["stop_reason"]=="cup_near_full" and candidate is None
        rows.append(dict(name=case["name"],expected_stop=case["stop_reason"],
                         decisions_sha256=sha(decisions),samples_sha256=sha(samples),
                         first_zero_s=zero,baseline_stop_s=baseline,candidate_stop_s=candidate,
                         candidate_inventory_ml=at_candidate,
                         baseline_lag_s=round(baseline-zero,6) if positive else None,
                         candidate_lag_s=round(candidate-zero,6) if positive else None))
    return rows


def checkpoint(plan, spec):
    path=spec["checkpoint"]
    assert sha(path)==plan["checkpoints"][path]
    return torch.load(path,map_location="cpu",weights_only=False)


def main():
    cup_root = ROOT/"sized-autonomy-r1"
    bottle_root = ROOT/"bottle-autonomy"
    empty_root = ROOT/"bottle-empty-autonomy"
    mixed_root = ROOT/"mixed-vessel-autonomy"
    cup_plan,bottle_plan,empty_plan = (read(p/"plan.json") for p in (cup_root,bottle_root,empty_root))
    parents = (empty_root/"development-results.json",mixed_root/"development-results.json",
               bottle_root/"test-results.json",bottle_root/"development-results.json")
    rows = []
    for path in parents:
        report=read(path)
        assert report["plan_sha256"]==sha(path.with_name("plan.json"))
        assert report["strict_passes"]==report["completed"]==report["planned"]
        rows.extend(replay(report))
    assert sum(x["expected_stop"]=="bottle_empty" for x in rows)==3 and len(rows)==6
    receipt=STUDY/"empty-replay.json"
    save(receipt,dict(scope="Consumed tall-bottle decision streams, no fresh trial outcomes",
                      cases=rows,positive_cases=3,near_full_negative_cases=3,
                      conditions=dict(tilt_deg=120.,wrist=.92,scene=.98,presence=.7,
                                      flow_ml_s_below=1.,consecutive_frames=10)))
    calibration=STUDY/"calibration.json"
    assert not calibration.exists()
    shutil.copy2(cup_root/"calibration.json",calibration)
    plan={k:v for k,v in cup_plan.items() if k not in ("development","test","checkpoints")}
    plan.update(development=[],test=[],checkpoints={},development_requires_strict_zero=True,
                development_empty_lag_limit_s=2.,empty_confirmation_replay_sha256=sha(receipt),
                prerequisite_reports={str(path):sha(path) for path in parents},
                controller_scope="One tall-bottle latency verification, then six crossed-size and placement tests")
    development=copy.deepcopy(next(s for s in empty_plan["development"]
                                   if s["name"].endswith("tall_narrow")))
    dev_saved=checkpoint(empty_plan,development)
    dev_saved["tall_empty_confirmation"]=True
    dev_saved["model_version"]="visual_guard_tall_empty_r1"
    dev_path=STUDY/"verify.pt"
    assert not dev_path.exists()
    torch.save(dev_saved,dev_path)
    plan["checkpoints"][str(dev_path)]=sha(dev_path)
    development.update(name="SA_A_tall_empty_verify_r1",seed=2611122000,checkpoint=str(dev_path))
    plan["development"].append(development)
    fixtures=(
        ("empty",(.9,1.),(1.075,1.),(.480,-.147),(-.121,-.011)),
        ("full",(1.,.9),(1.075,.95),(.502,-.192),(-.105,.005)),
        ("empty",(1.1,.9),(1.075,.95),(.486,-.187),(-.115,.009)),
        ("full",(1.1,1.1),(.925,1.),(.505,-.160),(-.126,-.014)),
        ("empty",(1.,1.1),(.925,.95),(.476,-.182),(-.108,-.008)),
        ("full",(.9,1.1),(1.075,1.),(.497,-.141),(-.120,.006)),
    )
    for i,(goal,cup_size,bottle_size,cup_xy,source_xy) in enumerate(fixtures):
        cup=next(s for s in cup_plan["test"] if (s["cup_height_scale"],s["cup_diameter_scale"])==cup_size)
        bottle=next(s for s in bottle_plan["test"] if (s["bottle_height_scale"],s["bottle_diameter_scale"])==bottle_size)
        saved,base=checkpoint(cup_plan,cup),checkpoint(bottle_plan,bottle)
        assert all(np.array_equal(saved["calibration"][k],base["calibration"][k]) for k in ("K","T","table_z"))
        for k in ("source_bank","source_tracking_bank","bottle_geometry","inputs"):
            saved[k]=base[k]
        saved["tall_empty_confirmation"]=saved["bottle_geometry"]["lip_height_m"]>=.14
        saved["model_version"]="visual_guard_tall_empty_r1"
        weight=STUDY/f"test-{i:02}.pt"
        assert not weight.exists()
        torch.save(saved,weight)
        plan["checkpoints"][str(weight)]=sha(weight)
        scene=copy.deepcopy(cup)
        for k in ("bottle_size_autonomy","bottle_height_scale","bottle_diameter_scale","declared_bottle_geometry"):
            scene[k]=bottle[k]
        scene.update(name=f"SA_A_tall_empty_test_r1_{i:02}_{goal}",study_split="test",
                     expected_stop="cup_near_full" if goal=="full" else "bottle_empty",
                     initial_ml=80. if goal=="full" else 28.,
                     calibration_initial_height=.55 if goal=="full" else .26,
                     cup_world=[*cup_xy,cup["cup_world"][2]],source_xy=list(source_xy),
                     seed=2611122100+i,checkpoint=str(weight))
        quantum=scene["spacing"]**3*1e6
        scene["cup_initial_ml"]=len(seeded_prefill(scene))*quantum
        initial_points(scene)
        plan["test"].append(scene)
    save(STUDY/"plan.json",plan)
    print(json.dumps(dict(replay=rows,development=1,test=len(fixtures))),flush=True)


if __name__=="__main__":
    main()
