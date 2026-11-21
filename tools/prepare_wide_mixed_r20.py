"""Freeze four repairs and commit the fresh positions and episode seeds before simulation."""
from pathlib import Path
import shutil
import torch

from fit_wide_mixed_r20 import NECK, MOTION, PARENT, STUDY, add_refinement
from fit_wide_mixed_r6 import read, save
from pour_size_autonomy import sha
from prepare_wide_mixed_r8 import suffix
from prepare_wide_mixed_r17 import identical
from prepare_wide_mixed_r19 import seal_layouts


def main():
    torch.set_num_threads(2)
    plan = read(PARENT/"plan.json")
    previous = read(PARENT/"completion.json")
    assert previous["plan_sha256"]==sha(PARENT/"plan.json")
    assert {suffix(c):c["measured_loss_particles"] for c in previous["test_cases"] if not c["strict_pass"]}=={
        "12_swap":0,"14_swap":0,"11_normal":2,"13_normal":2}
    parents = {suffix(s):s for s in plan["test"]}
    motion, neck = (read(STUDY/n) for n in ("stream-fit.json","neck-replay.json"))
    assert motion["passed"] and neck["passed"]
    assert set(motion["cases"])==set(MOTION) and set(neck["cases"])==set(NECK)
    sources = {p.name:sha(p) for p in (STUDY/"src").glob("*.py")}
    assert sources==read(PARENT/"geometry.json")["sources"]
    replacements, proofs = {}, []
    for key in NECK+MOTION:
        fit = (neck if key in NECK else motion)["cases"][key]
        old = fit["checkpoint"]
        assert fit["passed"] and old==parents[key]["checkpoint"] and old not in replacements
        before = plan["checkpoints"].pop(old)
        assert sha(old)==before
        state = torch.load(old,map_location="cpu",weights_only=False)
        if key in NECK:
            artifact = STUDY/("neck-"+key+".pt")
            assert sha(artifact)==fit["bank_sha256"]
            candidate = add_refinement(state,torch.load(artifact,map_location="cpu",weights_only=False))
            assert all(identical(v,candidate["source_heatmap"][k]) for k,v in state["source_heatmap"].items() if k!="refinement")
            fields = ["source_heatmap"]
        else:
            changes = fit["mouth_options"]
            assert changes.keys()<={"tail_window_deg","tail_setback_m","tail_drop_m","finish_mouth_pose","late_mouth_shift_m"}
            candidate = dict(state,**changes)
            fields = sorted(changes)
        dest = STUDY/Path(old).name
        torch.save(candidate,dest)
        actual = torch.load(dest,map_location="cpu",weights_only=False)
        assert identical(actual,candidate)
        assert all(identical(value,actual[k]) for k,value in state.items() if k not in fields)
        replacements[old] = str(dest)
        plan["checkpoints"][str(dest)] = sha(dest)
        proofs.append(dict(original=old,parent_sha256=before,candidate_sha256=sha(dest),fields=fields))
    assert all(sha(p)==h for p,h in plan["checkpoints"].items())
    save(STUDY/"frozen-controller.json",dict(passed=True,runtime_sources=sources,changed=proofs,
         unchanged_checkpoints=len(plan["checkpoints"])-len(proofs),parent_plan_sha256=sha(PARENT/"plan.json")))
    seal_layouts(plan,parents,replacements,NECK+MOTION,"SA_A_wide_mixed_r20",2611212000,2611212700,2,
                 "Bottle-neck template corroboration for two checkpoints and two mid-pour trajectories")
    for path in (PARENT/"plan.json",PARENT/"completion.json",STUDY/"neck-replay.json",STUDY/"stream-fit.json",
                 STUDY/"spill-trace.json",STUDY/"frozen-controller.json"):
        plan["prerequisite_reports"][str(path)] = sha(path)
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    save(STUDY/"plan.json",plan)
    print("Sealed four targeted pours and 15 fresh matched pairs",flush=True)


if __name__=="__main__":
    main()
