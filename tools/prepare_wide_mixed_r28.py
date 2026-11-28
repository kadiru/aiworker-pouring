"""Seal three bottle detectors and one cup detector, preserving all other checkpoint fields."""
from pathlib import Path
import shutil
import torch

from fit_wide_mixed_r28_bottles import BOTTLES, CUP, PARENT, STUDY
from fit_wide_mixed_r20 import add_refinement
from fit_wide_mixed_r6 import read, save
from pour_size_autonomy import sha
from prepare_wide_mixed_r8 import suffix
from prepare_wide_mixed_r17 import identical
from prepare_wide_mixed_r19 import seal_layouts


def main():
    torch.set_num_threads(2)
    plan = read(PARENT/"plan.json")
    completion = read(PARENT/"completion.json")
    assert sha(PARENT/"plan.json")==completion["plan_sha256"]
    targets = (*BOTTLES,CUP)
    assert {suffix(c) for c in completion["test_cases"] if not c["strict_pass"]}==set(targets)
    parents = {suffix(s):s for s in plan["test"]}
    neck, cup = (read(STUDY/n) for n in ("neck-replay.json","cup-startup-replay.json"))
    assert set(neck["cases"])==set(BOTTLES) and neck["passed"] and cup["passed"]
    assert neck["parent_plan_sha256"]==cup["parent_plan_sha256"]==completion["plan_sha256"]
    sources={p.name:sha(p) for p in (STUDY/"src").glob("*.py")}
    assert sources==read(PARENT/"geometry.json")["sources"]
    replacements, changes = {}, []
    for key in targets:
        old = parents[key]["checkpoint"]
        parent_hash = plan["checkpoints"].pop(old)
        assert sha(old)==parent_hash and old not in replacements
        before = torch.load(old,map_location="cpu",weights_only=False)
        if key in BOTTLES:
            fit = neck["cases"][key]
            artifact = STUDY/f"neck-{key}.pt"
            assert fit["checkpoint"]==old and fit["passed"] and sha(artifact)==fit["bank_sha256"]
            assert fit["field"]=="source_heatmap_proposal_and_refinement"
            bank = torch.load(artifact,map_location="cpu",weights_only=False)
            candidate = add_refinement(before,bank["refinement"])
            candidate["source_heatmap"]["proposal"] = bank["proposal"]
            fields = [("source_heatmap","proposal"),("source_heatmap","refinement")]
        else:
            artifact=STUDY/"cup-coarse-11.pt"
            assert cup["checkpoint"]==old and sha(artifact)==cup["artifact_sha256"]
            weights=torch.load(artifact,map_location="cpu",weights_only=False)
            cfg=before["sized_cup_perception"]
            candidate=dict(before,sized_cup_perception=dict(cfg, tracker=dict(cfg["tracker"],coarse_state=weights)))
            fields=[("sized_cup_perception","tracker","coarse_state")]
        dest = STUDY/Path(old).name
        assert not dest.exists()
        torch.save(candidate,dest)
        actual = torch.load(dest,map_location="cpu",weights_only=False)
        assert identical(actual,candidate)
        for field_path in fields:
            saved,previous=actual,before
            for component in field_path[:-1]:
                saved,previous=saved[component],previous[component]
            saved[field_path[-1]]=previous[field_path[-1]]
        assert identical(actual,before)
        replacements[old] = str(dest)
        plan["checkpoints"][str(dest)] = sha(dest)
        changes.append(dict(original=old,parent_sha256=parent_hash,candidate_sha256=sha(dest),fields=fields))
    assert all(sha(p)==h for p,h in plan["checkpoints"].items())
    frozen = STUDY/"frozen-controller.json"
    save(frozen,dict(passed=True,changed=changes,runtime_sources=sources,
         unchanged_checkpoints=len(plan["checkpoints"])-len(changes),parent_plan_sha256=completion["plan_sha256"]))
    seal_layouts(plan,parents,replacements,targets,"SA_A_wide_mixed_r28",2611289400,2611289700,2,
                 "Three bottle startup detectors and pair 11 cup startup heatmap. All stop, grasp and motion fields frozen.")
    for report in (PARENT/"plan.json",PARENT/"completion.json",STUDY/"neck-replay.json",STUDY/"cup-startup-replay.json",
                   STUDY/"cup-calibration.json",frozen):
        plan["prerequisite_reports"][str(report)] = sha(report)
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    save(STUDY/"plan.json",plan)


if __name__=="__main__":
    main()
