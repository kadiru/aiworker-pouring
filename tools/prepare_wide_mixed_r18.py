"""Freeze one high-angle mouth pose and predeclare both targeted pours and 30 fresh tests."""
import copy
from pathlib import Path
import shutil

import numpy as np
import torch

from fit_wide_mixed_r18 import PARENT, STUDY, TARGET
from fit_wide_mixed_r6 import read, save
from pour_size_autonomy import sha
from prepare_wide_mixed_r8 import suffix
from prepare_wide_mixed_r10 import shift
from prepare_wide_mixed_r17 import identical


def main():
    torch.set_num_threads(2)
    plan = read(PARENT/"plan.json")
    previous = read(PARENT/"completion.json")
    assert previous["plan_sha256"] == sha(PARENT/"plan.json")
    failure, = [r for r in previous["test_cases"] if not r["strict_pass"]]
    assert suffix(failure)==TARGET and failure["measured_loss_particles"]==2
    fit = read(STUDY/"stream-fit.json")
    assert fit["passed"] and list(fit["cases"])==[TARGET] and "4070" in fit["device"]
    result = fit["cases"][TARGET]
    assert result["passed"] and result["parent_completion_sha256"]==sha(PARENT/"completion.json")
    old = result["checkpoint"]
    assert all(sha(p)==h for p,h in plan["checkpoints"].items())
    sources = {p.name:sha(p) for p in (STUDY/"src").glob("*.py")}
    assert sources==read(PARENT/"geometry.json")["sources"]
    settings = result["mouth_options"]
    assert set(settings)=={"finish_mouth_pose"}
    original = torch.load(old,map_location="cpu",weights_only=False)
    dest = STUDY/Path(old).name
    candidate = dict(original,**settings)
    torch.save(candidate,dest)
    reopened = torch.load(dest,map_location="cpu",weights_only=False)
    assert identical(reopened,candidate)
    assert all(identical(value,reopened[key]) for key,value in original.items() if key!="finish_mouth_pose")
    digest = plan["checkpoints"].pop(old)
    plan["checkpoints"][str(dest)] = sha(dest)
    save(STUDY/"frozen-localization.json",dict(passed=True,runtime_sources=sources,
         modified_checkpoint=dict(original=old,parent_sha256=digest,candidate_sha256=sha(dest),motion_fields=settings),
         unchanged_checkpoints=len(plan["checkpoints"])-1,parent_plan_sha256=sha(PARENT/"plan.json")))
    parents = {suffix(s):s for s in plan["test"]}
    assert parents["02_normal"]["checkpoint"]==parents[TARGET]["checkpoint"]==old

    def trial(key):
        spec=copy.deepcopy(parents[key])
        if spec["checkpoint"]==old:
            spec["checkpoint"]=str(dest)
        return spec

    rng = np.random.default_rng(2611201800)
    delta = rng.uniform(.0011,.0025,(15,2,2))*rng.choice((-1.,1.),(15,2,2))
    fresh=[]
    for index,key in enumerate(parents):
        spec=trial(key)
        offsets=delta[spec["pair_index"]]
        shift(spec,*(offsets[::-1] if spec["swapped"] else offsets))
        spec.update(name="SA_A_wide_mixed_r18_test_"+key,seed=2611208700+index)
        fresh.append(spec)
    dev=[]
    for index,key in enumerate((TARGET,"02_normal")):
        spec=trial(key)
        spec.update(name="SA_A_wide_mixed_r18_development_"+key,seed=2611208800+index,study_split="development")
        dev.append(spec)
    assert len(fresh)==30 and sum(s["swapped"] for s in fresh)==15
    assert set(plan["checkpoints"])=={s["checkpoint"] for s in fresh}
    assert plan["development_requires_strict_zero"]
    plan.update(development=dev,test=fresh,test_shards=[[s["name"]] for s in fresh],
                shared_recording_shards=list(range(1,31)),development_empty_lag_cases=[],
                development_recording_upper_bytes=plan["storage_recording_upper_bytes"],
                controller_scope="One geometry-specific high-angle mouth pose. All other checkpoint fields and runtime sources frozen.")
    # Both targeted recordings share the recording volume. Budget from the
    # prior maximum of each fixture, plus 25 percent and 16 MiB for sidecars.
    historical = [r for p in sorted(PARENT.parent.glob("wide-mixed-r*/completion.json"))
                  for r in read(p)["test_cases"] if suffix(r) in (TARGET,"02_normal")]
    plan["development_recording_upper_bytes"]=int(1.25*max(r["recording_bytes"] for r in historical))+16*1024**2
    for path in (PARENT/"plan.json",PARENT/"completion.json",STUDY/"stream-fit.json",STUDY/"spill-trace.json",STUDY/"frozen-localization.json"):
        plan["prerequisite_reports"][str(path)]=sha(path)
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    save(STUDY/"plan.json",plan)
    print("Sealed 2 development layouts and 30 fresh trials",flush=True)


if __name__=="__main__":
    main()
