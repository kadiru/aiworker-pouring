"""Seal the three development repairs and 30 new matched layouts before simulation."""
import copy
from pathlib import Path
import shutil

import numpy as np
import torch

from fit_wide_mixed_r19 import PARENT, STUDY
from fit_wide_mixed_r6 import read,save
from pour_size_autonomy import sha
from prepare_wide_mixed_r8 import suffix
from prepare_wide_mixed_r10 import shift
from prepare_wide_mixed_r17 import identical

TARGETS=("01_normal","06_swap","12_swap")


def main():
    torch.set_num_threads(2)
    plan=read(PARENT/"plan.json")
    prior=read(PARENT/"completion.json")
    assert prior["plan_sha256"]==sha(PARENT/"plan.json")
    assert {suffix(c):c["measured_loss_particles"] for c in prior["test_cases"] if not c["strict_pass"]}==dict(zip(TARGETS,(0,1,2)))
    parents={suffix(s):s for s in plan["test"]}
    motion=read(STUDY/"stream-fit.json")
    neck=read(STUDY/"neck-replay.json")
    assert motion["passed"] and neck["passed"] and set(motion["cases"])==set(TARGETS[1:])
    assert neck["artifact_sha256"]==sha(STUDY/"neck-fallback.pt")
    sources={p.name:sha(p) for p in (STUDY/"src").glob("*.py")}
    assert sources==read(PARENT/"geometry.json")["sources"]
    replacements,proofs={},[]
    for key in TARGETS:
        old=parents[key]["checkpoint"]
        before=plan["checkpoints"].pop(old)
        assert sha(old)==before
        state=torch.load(old,map_location="cpu",weights_only=False)
        if key=="01_normal":
            assert neck["checkpoint"]==old and not state.get("source_heatmap_fallback")
            changes=dict(source_heatmap_fallback=torch.load(STUDY/"neck-fallback.pt",map_location="cpu",weights_only=False))
        else:
            fit=motion["cases"][key]
            assert fit["passed"] and old==fit["checkpoint"]
            changes=fit["mouth_options"]
            assert changes.keys()<={"tail_window_deg","tail_setback_m","tail_drop_m","finish_mouth_pose"}
        dest=STUDY/Path(old).name
        torch.save(dict(state,**changes),dest)
        actual=torch.load(dest,weights_only=False,map_location="cpu")
        assert all(identical(v,actual[k]) for k,v in state.items() if k not in changes)
        assert all(identical(v,actual[k]) for k,v in changes.items())
        replacements[old]=str(dest)
        plan["checkpoints"][str(dest)]=sha(dest)
        proofs.append(dict(original=old,parent_sha256=before,candidate_sha256=sha(dest),fields=sorted(changes)))
    assert all(sha(p)==h for p,h in plan["checkpoints"].items())
    assert parents["02_swap"]["checkpoint"] not in replacements
    save(STUDY/"frozen-controller.json",dict(passed=True,runtime_sources=sources,changed=proofs,
        unchanged_checkpoints=len(plan["checkpoints"])-len(proofs),parent_plan_sha256=sha(PARENT/"plan.json")))

    def candidate(key):
        spec=copy.deepcopy(parents[key])
        spec["checkpoint"]=replacements.get(spec["checkpoint"],spec["checkpoint"])
        return spec

    rng=np.random.default_rng(2611201900)
    displacements=rng.uniform(.0011,.0025,(15,2,2))*rng.choice((-1.,1.),(15,2,2))
    fresh=[]
    for i,key in enumerate(parents):
        spec=candidate(key)
        offsets=displacements[spec["pair_index"]]
        shift(spec,*(offsets[::-1] if spec["swapped"] else offsets))
        spec.update(name="SA_A_wide_mixed_r19_test_"+key,seed=2611209700+i)
        fresh.append(spec)
    dev=[]
    for i,key in enumerate(TARGETS):
        spec=candidate(key)
        spec.update(name="SA_A_wide_mixed_r19_development_"+key,seed=2611209800+i,study_split="development")
        dev.append(spec)
    assert len(fresh)==30 and sum(s["swapped"] for s in fresh)==15
    assert set(plan["checkpoints"])=={s["checkpoint"] for s in fresh}
    plan.update(development=dev,test=fresh,test_shards=[[s["name"]] for s in fresh],
                shared_recording_shards=[],development_empty_lag_cases=[],
                development_requires_strict_zero=True,development_recording_upper_bytes=plan["storage_recording_upper_bytes"],
                controller_scope="Geometry-specific startup neck fallback and two phase-local pouring trajectories")
    for p in (PARENT/"plan.json",PARENT/"completion.json",STUDY/"neck-replay.json",STUDY/"stream-fit.json",
              STUDY/"spill-trace.json",STUDY/"frozen-controller.json"):
        plan["prerequisite_reports"][str(p)]=sha(p)
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    save(STUDY/"plan.json",plan)
    print("Sealed",len(dev),"development and",len(fresh),"fresh cases",flush=True)


if __name__=="__main__":
    main()
