"""Apply only the fitted post-lift and outbound settings and seal the 30 fresh trials."""
import copy
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from fit_wide_mixed_r17 import LAYOUTS, PARENT, STUDY
from fit_wide_mixed_r6 import read, save
from pour_size_autonomy import sha
from prepare_wide_mixed_r8 import suffix
from prepare_wide_mixed_r10 import shift


def identical(a,b):
    if type(a) is not type(b):
        return False
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(identical(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    if torch.is_tensor(a):
        return a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return a.dtype==b.dtype and np.array_equal(a,b)
    return a==b


def main():
    torch.set_num_threads(2)
    plan = read(PARENT/"plan.json")
    parent = read(PARENT/"completion.json")
    assert parent["plan_sha256"] == sha(PARENT/"plan.json")
    assert {suffix(c): c["measured_loss_particles"] for c in parent["test_cases"] if not c["strict_pass"]} == {
        "05_swap":0,"05_normal":1,"10_swap":2,"14_swap":2}
    specs = {suffix(s):s for s in plan["test"]}
    fit = read(STUDY/"stream-fit.json")
    assert fit["passed"] and set(fit["cases"])==set(LAYOUTS) and "4070" in fit["device"]
    changes = {}
    for key,result in fit["cases"].items():
        original = result["checkpoint"]
        assert original == specs[key]["checkpoint"] and result["passed"]
        fields, thresholds = changes.setdefault(original,({},{}))
        new, limits = result["mouth_options"], result.get("threshold_options",{})
        assert set(new)<={"tail_window_deg","tail_setback_m","tail_drop_m","late_mouth_shift_m","finish_mouth_pose"}
        assert set(limits)<={"grasp_follow_radius_px"}
        assert not (fields.keys() & new.keys() or thresholds.keys() & limits.keys())
        fields.update(new)
        thresholds.update(limits)

    sources = {p.name:sha(p) for p in (STUDY/"src").glob("*.py")}
    assert sources==read(PARENT/"geometry.json")["sources"]
    replacements, proofs = {}, []
    for old,digest in list(plan["checkpoints"].items()):
        assert sha(old)==digest
        if old not in changes:
            continue
        fields,thresholds = changes[old]
        state = torch.load(old,map_location="cpu",weights_only=False)
        candidate = dict(state, **fields)
        candidate["thresholds"]=dict(state["thresholds"],**thresholds)
        dest = STUDY/Path(old).name
        torch.save(candidate,dest)
        verified = torch.load(dest,map_location="cpu",weights_only=False)
        assert identical(verified,candidate)
        assert all(identical(v,verified[k]) for k,v in state.items() if k not in fields and k!="thresholds")
        assert all(identical(v,verified["thresholds"][k]) for k,v in state["thresholds"].items() if k not in thresholds)
        replacements[old]=str(dest)
        del plan["checkpoints"][old]
        plan["checkpoints"][str(dest)]=sha(dest)
        proofs.append(dict(original=old,parent_sha256=digest,candidate_sha256=sha(dest),
                           motion_fields=fields,post_lift_thresholds=thresholds))
    save(STUDY/"frozen-localization.json",dict(passed=True,runtime_sources=sources,modified_checkpoints=proofs,
         unchanged_checkpoints=len(plan["checkpoints"])-len(proofs),parent_plan_sha256=sha(PARENT/"plan.json")))

    def candidate(key):
        spec=copy.deepcopy(specs[key])
        spec["checkpoint"]=replacements.get(spec["checkpoint"],spec["checkpoint"])
        return spec

    rng=np.random.default_rng(2611191700)
    delta=rng.uniform(.0011,.0025,(15,2,2))*rng.choice((-1.,1.),(15,2,2))
    fresh=[]
    for index,key in enumerate(specs):
        spec=candidate(key)
        offsets=delta[spec["pair_index"]]
        shift(spec,*(offsets[::-1] if spec["swapped"] else offsets))
        spec.update(name="SA_A_wide_mixed_r17_test_"+key,seed=2611197700+index)
        fresh.append(spec)
    development=[]
    for index,key in enumerate(LAYOUTS):
        spec=candidate(key)
        spec.update(name="SA_A_wide_mixed_r17_development_"+key,seed=2611197800+index,study_split="development")
        development.append(spec)
    assert set(plan["checkpoints"])=={s["checkpoint"] for s in fresh}
    # Reserve 25% above the largest observed native file among these four fixtures,
    # plus 16 MiB per episode for sidecars. The live volume reserve stays at 3 GiB.
    sizes=[c["recording_bytes"] for path in sorted(PARENT.parent.glob("wide-mixed-r*/completion.json"))
           for c in read(path)["test_cases"] if suffix(c) in LAYOUTS]
    plan.update(development=development,test=fresh,test_shards=[[s["name"] for s in fresh[i:i+2]] for i in range(0,30,2)],
                development_recording_upper_bytes=math.ceil(max(sizes)*1.25)+16*1024**2,
                development_empty_lag_cases=[s["name"] for s in development if s["expected_stop"]=="bottle_empty" and s["bottle_height_scale"]>1.],
                controller_scope="Geometry-specific bottle-mouth trajectory and post-lift confirmation radius")
    for path in (PARENT/"plan.json",PARENT/"completion.json",STUDY/"stream-fit.json",STUDY/"spill-trace.json",STUDY/"frozen-localization.json"):
        plan["prerequisite_reports"][str(path)]=sha(path)
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    save(STUDY/"plan.json",plan)
    print("Sealed",len(fresh),"fresh trials and",len(development),"development fixtures",flush=True)


if __name__=="__main__":
    main()
