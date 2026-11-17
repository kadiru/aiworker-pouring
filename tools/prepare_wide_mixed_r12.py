"""Seal the fitted normal-01 motion and 30 fresh normal/swapped trials."""
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/"wide-mixed-r11"
STUDY=ROOT/"wide-mixed-r12"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import sha
from fit_wide_mixed_r6 import read,save
from prepare_wide_mixed_r8 import measure_spill,suffix
from prepare_wide_mixed_r10 import shift

SOURCE="01_normal"
DEVELOPMENT=(
    (SOURCE,"",None),
    (SOURCE,"_offset_a",([.0031,-.0028],[-.0033,.0029])),
    (SOURCE,"_offset_b",([-.0032,.0027],[.003,-.0028])),
    ("01_swap","",None),
)


def main():
    torch.set_num_threads(2)
    plan_path=PARENT/"plan.json"
    previous=PARENT/"completion.json"
    plan=read(plan_path)
    prior=read(previous)
    cases=prior["test_cases"]
    assert prior["plan_sha256"]==sha(plan_path) and len(cases)==30
    failed=[c for c in cases if not c["strict_pass"]]
    assert len(failed)==1 and suffix(failed[0])==SOURCE and failed[0]["measured_loss_particles"]==1
    fit_path=STUDY/"stream-fit.json"
    fit=read(fit_path)
    assert fit["passed"] and fit["device"]=="NVIDIA GeForce RTX 4070"
    spill_path=STUDY/"spill-trace.json"
    save(spill_path,dict(cases=[measure_spill(failed[0],fit["mouth_options"])]))
    old={suffix(spec):spec for spec in plan["test"]}
    original=old[SOURCE]["checkpoint"]
    assert old["01_swap"]["checkpoint"]==original
    assert sha(original)==plan["checkpoints"].pop(original)
    state=torch.load(original,map_location="cpu",weights_only=False)
    state.update(fit["mouth_options"])
    state["model_version"]="visual_guard_wide_mixed_r12"
    dest=STUDY/Path(original).name
    assert not dest.exists()
    torch.save(state,dest)
    plan["checkpoints"][str(dest)]=sha(dest)

    def candidate(parent):
        spec=copy.deepcopy(parent)
        if spec["checkpoint"]==original:
            spec["checkpoint"]=str(dest)
        return spec

    rng=np.random.default_rng(2611161200)
    delta=rng.uniform(.0011,.0025,(15,2,2))*rng.choice((-1.,1.),(15,2,2))
    fresh=[]
    for i,(key,parent) in enumerate(old.items()):
        spec=candidate(parent)
        shift(spec,*delta[spec["pair_index"]][::(-1 if spec["swapped"] else 1)])
        spec.update(name="SA_A_wide_mixed_r12_test_"+key,seed=2611163700+i)
        fresh.append(spec)
    development=[]
    for i,(key,label,offset) in enumerate(DEVELOPMENT):
        spec=candidate(old[key])
        if offset is not None:
            shift(spec,*offset)
        spec.update(name="SA_A_wide_mixed_r12_development_"+key+label,seed=2611163800+i,
                    study_split="development")
        development.append(spec)

    largest=max(r["recording_bytes"] for r in cases+prior["development_cases"])
    plan.update(development=development,test=fresh,
                test_shards=[[s["name"] for s in fresh[i:i+10]] for i in (0,10,20)],
                storage_recording_upper_bytes=max(512*1024**2,2*largest),
                development_empty_lag_cases=[r["name"] for r in development if r["expected_stop"]=="bottle_empty"],
                development_requires_strict_zero=True,
                controller_scope="Fitted normal-01 mouth path with the existing localization and stop controller")
    plan["prerequisite_reports"].update({str(p):sha(p) for p in (plan_path,previous,fit_path,spill_path)})
    for filename in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/filename,STUDY/filename)
    plan["empty_confirmation_replay_sha256"]=sha(STUDY/"empty-replay.json")
    save(STUDY/"plan.json",plan)
    print(dict(development=len(development),fresh=len(fresh),
               largest_prior_bytes=largest,per_recording_budget_bytes=plan["storage_recording_upper_bytes"]),
          flush=True)


if __name__=="__main__":
    main()
