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


def prepare(parent, study, neck_targets, motion_targets, losses, prefix, layout_seed, trial_seed, scope):
    torch.set_num_threads(2)
    plan = read(parent/"plan.json")
    previous = read(parent/"completion.json")
    assert previous["plan_sha256"]==sha(parent/"plan.json")
    assert {suffix(c):c["measured_loss_particles"] for c in previous["test_cases"] if not c["strict_pass"]}==losses
    parents = {suffix(s):s for s in plan["test"]}
    motion = read(study/"stream-fit.json")
    assert motion["passed"] and set(motion["cases"]) == set(motion_targets)
    if neck_targets:
        neck = read(study/"neck-replay.json")
        assert neck["passed"] and set(neck["cases"]) == set(neck_targets)
    sources = {p.name:sha(p) for p in (study/"src").glob("*.py")}
    assert sources==read(parent/"geometry.json")["sources"]
    replacements, proofs = {}, []
    for key in neck_targets+motion_targets:
        fit = (neck if key in neck_targets else motion)["cases"][key]
        old = fit["checkpoint"]
        assert fit["passed"] and old==parents[key]["checkpoint"] and old not in replacements
        before = plan["checkpoints"].pop(old)
        assert sha(old)==before
        state = torch.load(old,map_location="cpu",weights_only=False)
        if key in neck_targets:
            artifact = study/("neck-"+key+".pt")
            assert sha(artifact)==fit["bank_sha256"]
            bank = torch.load(artifact,map_location="cpu",weights_only=False)
            field = fit.get("field", "source_heatmap")
            if field == "source_heatmap":
                candidate = add_refinement(state, bank)
                assert all(identical(v,candidate[field][k]) for k,v in state[field].items() if k!="refinement")
            else:
                assert field == "source_heatmap_fallback" and state.get(field) is None
                candidate = dict(state, source_heatmap_fallback=bank)
            fields = [field]
        else:
            changes = fit["mouth_options"]
            assert changes.keys()<={"tail_window_deg","tail_setback_m","tail_drop_m","finish_mouth_pose","late_mouth_shift_m"}
            candidate = dict(state,**changes)
            fields = sorted(changes)
        dest = study/Path(old).name
        torch.save(candidate,dest)
        actual = torch.load(dest,map_location="cpu",weights_only=False)
        assert identical(actual,candidate)
        assert all(identical(value,actual[k]) for k,value in state.items() if k not in fields)
        replacements[old] = str(dest)
        plan["checkpoints"][str(dest)] = sha(dest)
        proofs.append(dict(original=old,parent_sha256=before,candidate_sha256=sha(dest),fields=fields))
    assert all(sha(p)==h for p,h in plan["checkpoints"].items())
    save(study/"frozen-controller.json",dict(passed=True,runtime_sources=sources,changed=proofs,
         unchanged_checkpoints=len(plan["checkpoints"])-len(proofs),parent_plan_sha256=sha(parent/"plan.json")))
    seal_layouts(plan,parents,replacements,neck_targets+motion_targets,prefix,layout_seed,trial_seed,2,scope)
    if neck_targets:
        path = study/"neck-replay.json"
        plan["prerequisite_reports"][str(path)] = sha(path)
    for path in (parent/"plan.json",parent/"completion.json",study/"stream-fit.json",
                 study/"spill-trace.json",study/"frozen-controller.json"):
        plan["prerequisite_reports"][str(path)] = sha(path)
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(parent/name,study/name)
    save(study/"plan.json",plan)
    print(f"Sealed {len(losses)} targeted pours and 15 fresh matched pairs",flush=True)


if __name__=="__main__":
    prepare(PARENT, STUDY, NECK, MOTION, {"12_swap":0,"14_swap":0,"11_normal":2,"13_normal":2},
            "SA_A_wide_mixed_r20", 2611212000, 2611212700,
            "Bottle-neck template corroboration for two checkpoints and two mid-pour trajectories")
