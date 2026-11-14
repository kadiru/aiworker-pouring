"""Provenance for the four spill repairs and a new 30-trial wide benchmark."""
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT/"wide-mixed-r7b"
STUDY = ROOT/"wide-mixed-r8"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import sha
from fit_wide_mixed_r6 import folder, read, save

LOCALIZATION = "02_swap"
# Geometry and tilt phase determine the live pose. Layout and outcome IDs
# identify the archived evidence used to choose these checkpoint settings.
MOUTH_REPAIR = {
    "03_normal": dict(tail_setback_m=.009,tail_window_deg=(115.,119.),tail_drop_m=.004),
    "10_swap": dict(tail_setback_m=.010,tail_window_deg=(115.,120.),tail_drop_m=.005),
    "11_swap": dict(low_tilt_drop_m=.004),
    "14_swap": dict(tail_setback_m=.010,tail_window_deg=(117.,124.),tail_drop_m=.006),
}


def suffix(case):
    return case["name"].split("_test_")[-1]


def measure_spill(case):
    path = folder(case)
    samples = read(path/"samples.json")
    escaped = {p["index"] for p in samples[-1]["diagnostic_outside_particles"]}
    assert len(escaped) == case["measured_loss_particles"] == 2
    stop = next(f for f in read(path/"policy-decisions.json") if f["stop_latched"])
    points = []
    for row in samples:
        particles = [p for p in row["diagnostic_outside_particles"] if p["index"] in escaped]
        if particles:
            points.append(dict(time=row["t"],tilt_deg=row["bottle_tilt_deg"],phase=row["phase"],
                               source_ml=row["bottle_ml"],height_fraction=row["cup_fill_fraction"],
                               particles=particles))
        if len(points)==6:
            break
    assert points and all(p["origin"] == "bottle" for p in points[0]["particles"])
    return dict(name=case["name"],recording_sha256=case["recording_sha256"],
                cup_size=case["cup_size"],bottle_size=case["bottle_size"],
                stop_time=stop["t"],observations=points,mouth_options=MOUTH_REPAIR[suffix(case)])


def main():
    torch.set_num_threads(2)
    parent_plan = PARENT/"plan.json"
    parent_report = PARENT/"completion.json"
    plan = read(parent_plan)
    prior = read(parent_report)
    assert prior["plan_sha256"] == sha(parent_plan) and prior["test"]["trials"] == 30
    results = prior["test_cases"]
    failures = {suffix(r) for r in results if not r["strict_pass"]}
    assert failures == MOUTH_REPAIR.keys() | {LOCALIZATION}
    spill_path = STUDY/"spill-trace.json"
    save(spill_path,dict(cases=[measure_spill(r) for r in results if suffix(r) in MOUTH_REPAIR]))
    neck_report = STUDY/"neck-fusion.json"
    neck_path = STUDY/"neck-heatmap.pt"
    fit = read(neck_report)
    assert fit["passed"] and sha(neck_path) == fit["checkpoint_sha256"]
    neck = torch.load(neck_path,map_location="cpu",weights_only=False)
    updated = {}
    original_checkpoints = plan["checkpoints"]
    plan["checkpoints"] = {}
    old = {suffix(s):s for s in plan["test"]}
    repair_by_checkpoint = {old[key]["checkpoint"]:value for key,value in MOUTH_REPAIR.items()}
    assert len(repair_by_checkpoint)==len(MOUTH_REPAIR)
    assert all(all(p["declared_cup_geometry"] == old[key]["declared_cup_geometry"] and
                   p["declared_bottle_geometry"] == old[key]["declared_bottle_geometry"]
                   for p in plan["test"] if p["checkpoint"]==old[key]["checkpoint"])
               for key in MOUTH_REPAIR)

    def checkpoint(source):
        if source not in updated:
            assert sha(source)==original_checkpoints[source]
            state = torch.load(source,map_location="cpu",weights_only=False)
            state["source_heatmap"] = neck
            state.update(repair_by_checkpoint.get(source,{}))
            state["model_version"] = "visual_guard_wide_mixed_r8"
            dest = STUDY/Path(source).name
            assert not dest.exists()
            torch.save(state,dest)
            updated[source] = str(dest)
            plan["checkpoints"][str(dest)] = sha(dest)
        return updated[source]

    rng = np.random.default_rng(2611130800)
    deltas = rng.uniform(.0011,.0025,(15,2,2))*rng.choice((-1.,1.),(15,2,2))
    tests = []
    for i,(key,original) in enumerate(old.items()):
        spec = copy.deepcopy(original)
        cup,bottle = deltas[spec["pair_index"]][::(-1 if spec["swapped"] else 1)]
        spec["cup_world"][:2] = (np.asarray(spec["cup_world"][:2])+cup).tolist()
        delta = bottle-cup
        spec["source_xy"] = (np.asarray(spec["source_xy"])+[delta[1],-delta[0]]).tolist()
        spec.update(name="SA_A_wide_mixed_r8_test_"+key,seed=2611132700+i,
                    checkpoint=checkpoint(spec["checkpoint"]))
        tests.append(spec)

    development = []
    for i,key in enumerate((LOCALIZATION,*MOUTH_REPAIR)):
        spec = copy.deepcopy(old[key])
        spec.update(name="SA_A_wide_mixed_r8_development_"+key,seed=2611132800+i,
                    study_split="development",checkpoint=checkpoint(spec["checkpoint"]))
        development.append(spec)

    plan.pop("clearance_predecessors",None)
    largest = max(r["recording_bytes"] for r in results+prior["development_cases"])
    plan.update(development=development,test=tests,
                test_shards=[[s["name"] for s in tests[i:i+10]] for i in (0,10,20)],
                storage_recording_upper_bytes=max(512*1024**2,3*largest),
                development_empty_lag_cases=[s["name"] for s in development
                                              if s["bottle_height_scale"]>1.
                                              and s["expected_stop"]=="bottle_empty"],
                development_requires_strict_zero=True,
                controller_scope="Retrained neck proposal plus phase-specific mouth motion for four prior spills")
    plan["prerequisite_reports"].update({str(p):sha(p) for p in (parent_plan,parent_report,neck_report,spill_path)})
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    plan["empty_confirmation_replay_sha256"] = sha(STUDY/"empty-replay.json")
    save(STUDY/"plan.json",plan)
    print(dict(development=len(development),fresh=len(tests),
               max_prior_recording_bytes=largest,storage_recording_upper_bytes=plan["storage_recording_upper_bytes"]),
          flush=True)


if __name__=="__main__":
    main()
