"""Verify the neck fallback and seal three spill repairs with fresh swap pairs."""
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/"wide-mixed-r10b"
STUDY=ROOT/"wide-mixed-r11"
sys.path.insert(0,str(STUDY/"src"))
from pour_guard_vision import bottle_base_from_neck, locate_initial_source
from pour_size_autonomy import sha
from pour_temporal_localization import consensus
from fit_wide_mixed_r7 import erased, r11_archives, rows, shifted
from fit_wide_mixed_r6 import read,save
from prepare_wide_mixed_r8 import measure_spill,suffix
from prepare_wide_mixed_r10 import shift

LOCALIZATION={"07_normal","14_normal"}
MOUTH_REPAIR={
    "00_swap":dict(tail_setback_m=.006,tail_window_deg=(110.,115.),tail_drop_m=.004),
    "04_swap":dict(tail_setback_m=.009,tail_window_deg=(115.,121.),tail_drop_m=.005),
    "06_swap":dict(tail_setback_m=.010,tail_window_deg=(114.,121.),tail_drop_m=.006),
}


def bottle_size(spec):
    return spec["bottle_height_scale"],spec["bottle_diameter_scale"]


def verify_neck(neck, current):
    results=[]
    raw=rows(PARENT,LOCALIZATION,r11_archives(),full_startup=True)
    for row in raw:
        spec=current.get(suffix(row)) if "_test_" in row["name"] else None
        if spec is None:
            continue
        saved=torch.load(spec["checkpoint"],map_location="cpu",weights_only=False)
        saved["source_heatmap_fallback"]=neck
        limit=saved["thresholds"]["source_localization_score"]
        settings=saved["localization_consensus"]["source"]
        frames=row["frames"]
        uv=row["uv"]
        variants=[("native",frames,uv)]
        if row["target"]:
            for dx,dy in ((3,-4),(-4,3)):
                variants.append(("translated",[shifted(im,dx,dy) for im in frames],uv+[dx,dy]))
        missing=[erased(im,uv,i%2) for i,im in enumerate(frames)]
        methods=[locate_initial_source(im,saved)[2] for im in missing]
        erased_learned=sum(method.startswith("neck_heatmap") for method in methods)
        for mode,images,truth in variants:
            detected=[locate_initial_source(im,saved) for im in images]
            latest=[(p.tolist(),v) for p,v,_ in detected]
            vote=next((hit for i in range(settings["votes"],len(latest)+1)
                       if (hit:=consensus(latest[:i],limit,settings))),None)
            error=None
            if vote:
                estimate=bottle_base_from_neck(vote["uv"],row["cal"],row["lip"])
                expected=bottle_base_from_neck(truth,row["cal"],row["lip"])
                error=float(np.linalg.norm(estimate[:2]-expected[:2])*1000.)
            results.append(dict(name=row["name"],recording_sha256=row["sha256"],
                                target=row["target"],mode=mode,error_mm=error,
                                frames=len(images),fallback_frames=sum(m=="neck_heatmap_fallback" for _,_,m in detected),
                                erased_learned_frames=erased_learned if mode=="native" else None))
    targets=[r for r in results if r["target"]]
    passed=(len(targets)==6 and all(r["fallback_frames"]>=8 for r in targets if r["mode"]=="native")
            and all(r["error_mm"] is not None and r["error_mm"]<=2.5 for r in results)
            and not any(r["erased_learned_frames"] for r in results))
    path=STUDY/"neck-cascade-replay.json"
    save(path,dict(passed=passed,cases=results))
    print(dict(passed=passed,failed=[r for r in results if r["error_mm"] is None
                                   or r["error_mm"]>2.5 or r["erased_learned_frames"]]),flush=True)
    assert passed,targets
    return path


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    parent_plan=PARENT/"plan.json"
    parent_report=PARENT/"completion.json"
    plan=read(parent_plan)
    prior=read(parent_report)
    assert prior["plan_sha256"]==sha(parent_plan) and prior["test"]["trials"]==30
    cases=prior["test_cases"]
    assert {suffix(c) for c in cases if not c["strict_pass"]}==MOUTH_REPAIR.keys() | LOCALIZATION
    spill_path=STUDY/"spill-trace.json"
    save(spill_path,dict(cases=[measure_spill(c,MOUTH_REPAIR[suffix(c)])
                               for c in cases if suffix(c) in MOUTH_REPAIR]))
    neck_path=STUDY/"neck-heatmap.pt"
    neck_report=STUDY/"neck-fusion.json"
    fit=read(neck_report)
    assert fit["passed"] and fit["checkpoint_sha256"]==sha(neck_path)
    neck=torch.load(neck_path,map_location="cpu",weights_only=False)
    old={suffix(s):s for s in plan["test"]}
    affected={bottle_size(old[key]) for key in LOCALIZATION}
    cascade=verify_neck(neck,{key:s for key,s in old.items() if bottle_size(s) in affected})
    source_checksums=plan["checkpoints"]
    plan["checkpoints"]={}
    paths={}
    motion={old[key]["checkpoint"]:value for key,value in MOUTH_REPAIR.items()}
    assert len(motion)==len(MOUTH_REPAIR)
    assert all(all(s["declared_cup_geometry"]==old[key]["declared_cup_geometry"] and
                       s["declared_bottle_geometry"]==old[key]["declared_bottle_geometry"]
                   for s in old.values() if s["checkpoint"]==old[key]["checkpoint"])
               for key in MOUTH_REPAIR)

    def update(spec):
        source=spec["checkpoint"]
        if source not in paths:
            assert sha(source)==source_checksums[source]
            paths[source]=source
            if source in motion or bottle_size(spec) in affected:
                state=torch.load(source,map_location="cpu",weights_only=False)
                if bottle_size(spec) in affected:
                    state["source_heatmap_fallback"]=neck
                state.update(motion.get(source,{}))
                state["model_version"]="visual_guard_wide_mixed_r11"
                output=STUDY/Path(source).name
                assert not output.exists()
                torch.save(state,output)
                paths[source]=str(output)
            path=paths[source]
            plan["checkpoints"][path]=sha(path)
        return paths[source]

    rng=np.random.default_rng(2611151100)
    delta=rng.uniform(.0011,.0025,(15,2,2))*rng.choice((-1.,1.),(15,2,2))
    fresh=[]
    for i,(key,parent) in enumerate(old.items()):
        spec=copy.deepcopy(parent)
        shift(spec,*delta[spec["pair_index"]][::(-1 if spec["swapped"] else 1)])
        spec.update(name="SA_A_wide_mixed_r11_test_"+key,seed=2611153700+i,
                    checkpoint=update(parent))
        fresh.append(spec)
    development=[]
    for i,key in enumerate((*MOUTH_REPAIR,*sorted(LOCALIZATION))):
        parent=old[key]
        spec=copy.deepcopy(parent)
        spec.update(name="SA_A_wide_mixed_r11_development_"+key,seed=2611153800+i,
                    study_split="development",checkpoint=update(parent))
        development.append(spec)

    plan.pop("clearance_predecessors",None)
    largest=max(r["recording_bytes"] for r in cases+prior["development_cases"])
    plan.update(development=development,test=fresh,
                test_shards=[[s["name"] for s in fresh[i:i+10]] for i in (0,10,20)],
                storage_recording_upper_bytes=max(plan["storage_recording_upper_bytes"],3*largest),
                development_empty_lag_cases=[s["name"] for s in development if s["bottle_height_scale"]>1.
                                             and s["expected_stop"]=="bottle_empty"],
                development_requires_strict_zero=True,
                controller_scope="Fallback bottle-neck heatmap for two shapes and three spill path repairs")
    plan["prerequisite_reports"].update({str(p):sha(p) for p in
                                        (parent_plan,parent_report,neck_report,spill_path,cascade)})
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    plan["empty_confirmation_replay_sha256"]=sha(STUDY/"empty-replay.json")
    save(STUDY/"plan.json",plan)
    print(dict(development=len(development),fresh=len(fresh)),flush=True)


if __name__=="__main__":
    main()
