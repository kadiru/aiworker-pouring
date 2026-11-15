"""Seal the retrained neck and two updated late-pour paths for Isaac Sim."""
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT/"wide-mixed-r9b"
STUDY = ROOT/"wide-mixed-r10"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import sha
from fit_wide_mixed_r6 import read, save
from prepare_wide_mixed_r8 import measure_spill, suffix

LOCALIZATION = "12_normal"
MOUTH_REPAIR = {
    "01_normal": dict(tail_setback_m=.009,tail_window_deg=(114.,118.),tail_drop_m=.004),
    "02_swap": dict(tail_setback_m=.007,tail_window_deg=(114.,119.),tail_drop_m=.006),
}
# Additional development perturbations are independent of the frozen test draw.
DEVELOPMENT_OFFSETS = {
    "01_normal": ([.0032,-.0029],[-.0035,.0031]),
    "02_swap": ([-.0033,.0031],[.0034,-.0032]),
}


def shift(spec, cup, bottle):
    spec["cup_world"][:2] = (np.asarray(spec["cup_world"][:2])+cup).tolist()
    diff = np.asarray(bottle)-cup
    spec["source_xy"] = (np.asarray(spec["source_xy"])+[diff[1],-diff[0]]).tolist()


def main():
    torch.set_num_threads(2)
    plan_path = PARENT/"plan.json"
    completion = PARENT/"completion.json"
    plan = read(plan_path)
    prior = read(completion)
    assert prior["plan_sha256"]==sha(plan_path) and prior["test"]["trials"]==30
    cases = prior["test_cases"]
    assert {suffix(c) for c in cases if not c["strict_pass"]} == MOUTH_REPAIR.keys() | {LOCALIZATION}
    spill_path = STUDY/"spill-trace.json"
    save(spill_path,dict(cases=[measure_spill(c,MOUTH_REPAIR[suffix(c)])
                               for c in cases if suffix(c) in MOUTH_REPAIR]))
    neck_path = STUDY/"neck-heatmap.pt"
    neck_report = STUDY/"neck-fusion.json"
    trained = read(neck_report)
    assert trained["passed"] and trained["checkpoint_sha256"]==sha(neck_path)
    neck = torch.load(neck_path,map_location="cpu",weights_only=False)
    old = {suffix(s):s for s in plan["test"]}
    checkpoints = plan["checkpoints"]
    plan["checkpoints"] = {}
    paths = {}
    corrected_paths = {old[key]["checkpoint"]:value for key,value in MOUTH_REPAIR.items()}
    assert len(corrected_paths)==len(MOUTH_REPAIR)
    assert all(all(s["declared_cup_geometry"] == old[key]["declared_cup_geometry"] and
                   s["declared_bottle_geometry"] == old[key]["declared_bottle_geometry"]
                   for s in old.values() if s["checkpoint"]==old[key]["checkpoint"])
               for key in MOUTH_REPAIR)

    def update(source):
        if source not in paths:
            assert sha(source)==checkpoints[source]
            state = torch.load(source,map_location="cpu",weights_only=False)
            state["source_heatmap"] = neck
            state.update(corrected_paths.get(source,{}))
            state["model_version"] = "visual_guard_wide_mixed_r10"
            out = STUDY/Path(source).name
            assert not out.exists()
            torch.save(state,out)
            paths[source]=str(out)
            plan["checkpoints"][str(out)] = sha(out)
        return paths[source]

    rng = np.random.default_rng(2611151000)
    delta = rng.uniform(.0011,.0025,(15,2,2))*rng.choice((-1.,1.),(15,2,2))
    fresh = []
    for i,(key,original) in enumerate(old.items()):
        spec=copy.deepcopy(original)
        shift(spec,*delta[spec["pair_index"]][::(-1 if spec["swapped"] else 1)])
        spec.update(name="SA_A_wide_mixed_r10_test_"+key,seed=2611152700+i,
                    checkpoint=update(original["checkpoint"]))
        fresh.append(spec)

    development = []
    for key, offset in [(key, None) for key in (*MOUTH_REPAIR,LOCALIZATION)]+list(DEVELOPMENT_OFFSETS.items()):
        spec=copy.deepcopy(old[key])
        label=key
        if offset is not None:
            shift(spec,*offset)
            label += "_offset"
        spec.update(name="SA_A_wide_mixed_r10_development_"+label,seed=2611152800+len(development),
                    study_split="development",checkpoint=update(spec["checkpoint"]))
        development.append(spec)

    plan.pop("clearance_predecessors",None)
    largest = max(r["recording_bytes"] for r in cases+prior["development_cases"])
    plan.update(development=development,test=fresh,
                test_shards=[[s["name"] for s in fresh[i:i+10]] for i in (0,10,20)],
                storage_recording_upper_bytes=max(plan["storage_recording_upper_bytes"],3*largest),
                development_empty_lag_cases=[s["name"] for s in development if s["bottle_height_scale"]>1.
                                             and s["expected_stop"]=="bottle_empty"],
                development_requires_strict_zero=True,
                controller_scope="Retrained neck heatmap and revised late mouth paths, with shifted spill regressions")
    plan["prerequisite_reports"].update({str(p):sha(p) for p in (plan_path,completion,neck_report,spill_path)})
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    plan["empty_confirmation_replay_sha256"] = sha(STUDY/"empty-replay.json")
    save(STUDY/"plan.json",plan)
    print(dict(development=len(development),fresh=len(fresh)),flush=True)


if __name__=="__main__":
    main()
