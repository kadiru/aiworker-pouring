"""Seal the current RTX 4070 pour repair and fresh exact-swap fixtures."""
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT/"wide-mixed-r8"
STUDY = ROOT/"wide-mixed-r9"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import sha
from fit_wide_mixed_r6 import read, save
from prepare_wide_mixed_r8 import measure_spill, suffix

NEAR_FULL = "09_swap"
# These options are installed by declared bottle/cup combination. The
# archived aliases are only used to link the evidence and checkpoints.
MOUTH_REPAIR = {
    "02_swap": dict(tail_setback_m=.010,tail_window_deg=(115.,120.),tail_drop_m=.005),
    "09_normal": dict(tail_setback_m=.007,tail_window_deg=(113.,117.),tail_drop_m=.006),
    "12_swap": dict(tail_setback_m=.009,tail_window_deg=(116.,121.),tail_drop_m=.006),
}


def main():
    torch.set_num_threads(2)
    parent_plan = PARENT/"plan.json"
    parent_report = PARENT/"completion.json"
    plan = read(parent_plan)
    prior = read(parent_report)
    assert prior["plan_sha256"]==sha(parent_plan) and prior["test"]["trials"]==30
    cases = prior["test_cases"]
    assert {suffix(c) for c in cases if not c["strict_pass"]} == MOUTH_REPAIR.keys() | {NEAR_FULL}
    fit_path = STUDY/"near-full-fit.json"
    fit = read(fit_path)
    assert fit["passed"] and .920<=fit["pulse_limit_threshold"]<.925
    spill_path = STUDY/"spill-trace.json"
    save(spill_path,dict(cases=[measure_spill(c,MOUTH_REPAIR[suffix(c)])
                               for c in cases if suffix(c) in MOUTH_REPAIR]))
    old = {suffix(s):s for s in plan["test"]}
    parent_checkpoints = plan["checkpoints"]
    plan["checkpoints"] = {}
    new_paths = {}
    changes = {old[key]["checkpoint"]:options for key,options in MOUTH_REPAIR.items()}
    assert len(changes)==len(MOUTH_REPAIR)

    def checkpoint(spec):
        path = spec["checkpoint"]
        assert sha(path)==parent_checkpoints[path]
        cup = [spec["cup_height_scale"], spec["cup_diameter_scale"]]
        if path not in changes and cup!=fit["cup_size"]:
            plan["checkpoints"][path] = parent_checkpoints[path]
            return path
        if path not in new_paths:
            state = torch.load(path,map_location="cpu",weights_only=False)
            state.update(changes.get(path,{}))
            if cup==fit["cup_size"]:
                assert state["thresholds"]["full_settled_target"]==fit["prior_threshold"]
                state["thresholds"]["fitted_full_at_pulse_limit"]=fit["pulse_limit_threshold"]
            state["model_version"]="visual_guard_wide_mixed_r9"
            dest = STUDY/Path(path).name
            assert not dest.exists()
            torch.save(state,dest)
            new_paths[path] = str(dest)
            plan["checkpoints"][str(dest)] = sha(dest)
        return new_paths[path]

    rng = np.random.default_rng(2611140900)
    deltas = rng.uniform(.0011,.0025,(15,2,2))*rng.choice((-1.,1.),(15,2,2))
    fresh = []
    for i,(key,previous) in enumerate(old.items()):
        spec=copy.deepcopy(previous)
        cup,bottle=deltas[spec["pair_index"]][::(-1 if spec["swapped"] else 1)]
        spec["cup_world"][:2]=(np.asarray(spec["cup_world"][:2])+cup).tolist()
        diff=bottle-cup
        spec["source_xy"]=(np.asarray(spec["source_xy"])+[diff[1],-diff[0]]).tolist()
        spec.update(name="SA_A_wide_mixed_r9_test_"+key,seed=2611141700+i,
                    checkpoint=checkpoint(previous))
        fresh.append(spec)
    development=[]
    for i,key in enumerate((*MOUTH_REPAIR,NEAR_FULL)):
        spec=copy.deepcopy(old[key])
        spec.update(name="SA_A_wide_mixed_r9_development_"+key,seed=2611141800+i,
                    study_split="development",checkpoint=checkpoint(old[key]))
        development.append(spec)
    historic = max(r["recording_bytes"] for r in cases+prior["development_cases"])
    plan.update(development=development,test=fresh,
                test_shards=[[s["name"] for s in fresh[i:i+10]] for i in (0,10,20)],
                storage_recording_upper_bytes=max(plan["storage_recording_upper_bytes"],3*historic),
                development_empty_lag_cases=[s["name"] for s in development if s["bottle_height_scale"]>1.
                                              and s["expected_stop"]=="bottle_empty"],
                development_requires_strict_zero=True,
                controller_scope="Fitted small-cup confirmation at the pulse limit and three revised late-mouth paths")
    plan["prerequisite_reports"].update({str(p):sha(p) for p in (parent_plan,parent_report,fit_path,spill_path)})
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    plan["empty_confirmation_replay_sha256"] = sha(STUDY/"empty-replay.json")
    save(STUDY/"plan.json",plan)
    print(dict(development=len(development),fresh=len(fresh)),flush=True)


if __name__=="__main__":
    main()
