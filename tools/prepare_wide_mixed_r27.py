"""Seal source localization and pair-09 stop calibration before autonomous evaluation."""
from pathlib import Path
import shutil
import torch

from fit_wide_mixed_r27_neck import PARENT, STUDY, TARGETS
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
    targets = (*TARGETS, "09_normal", "09_swap")
    assert {suffix(c) for c in completion["test_cases"] if not c["strict_pass"]}==set(targets)
    parents = {suffix(s):s for s in plan["test"]}
    assert parents["09_normal"]["checkpoint"]==parents["09_swap"]["checkpoint"]
    neck, stop = (read(STUDY/n) for n in ("neck-replay.json","stop-fit.json"))
    assert set(neck["cases"])==set(TARGETS) and neck["passed"] and stop["passed"]
    sources = {p.name:sha(p) for p in (STUDY/"src").glob("*.py")}
    original = read(PARENT/"geometry.json")["sources"]
    assert original.keys()==sources.keys()
    assert [k for k in sources if sources[k]!=original[k]]==["pour_size_autonomy.py"]
    replacements, changes = {}, []
    for key in (*TARGETS,"09_normal"):
        old = parents[key]["checkpoint"]
        parent_hash = plan["checkpoints"].pop(old)
        assert sha(old)==parent_hash and old not in replacements
        before = torch.load(old,map_location="cpu",weights_only=False)
        if key in TARGETS:
            fit = neck["cases"][key]
            artifact = STUDY/f"neck-{key}.pt"
            assert fit["checkpoint"]==old and fit["passed"] and sha(artifact)==fit["bank_sha256"]
            assert fit["field"]=="source_heatmap_proposal_and_refinement"
            bank = torch.load(artifact,map_location="cpu",weights_only=False)
            candidate = add_refinement(before,bank["refinement"])
            candidate["source_heatmap"]["proposal"] = bank["proposal"]
            fields = [("source_heatmap","proposal"),("source_heatmap","refinement")]
        else:
            assert stop["checkpoint"]==old and stop["parent_sha256"]==parent_hash
            cfg = before["sized_cup_perception"]
            assert "draining_rgb_residual" not in cfg and cfg["agreement_limit"]==stop["agreement_limit"]
            limits = ("full_settled_target","fitted_full_at_pulse_limit")
            candidate = dict(before,sized_cup_perception=dict(cfg,draining_rgb_residual=stop["draining_rgb_residual"]),
                             thresholds=dict(before["thresholds"],**{k:stop[k] for k in limits}))
            fields = [("sized_cup_perception","draining_rgb_residual"),*(("thresholds",k) for k in limits)]
        dest = STUDY/Path(old).name
        assert not dest.exists()
        torch.save(candidate,dest)
        actual = torch.load(dest,map_location="cpu",weights_only=False)
        assert identical(actual,candidate)
        for group,field in fields:
            if field in before[group]:
                actual[group][field]=before[group][field]
            else:
                del actual[group][field]
        assert identical(actual,before)
        replacements[old] = str(dest)
        plan["checkpoints"][str(dest)] = sha(dest)
        changes.append(dict(original=old,parent_sha256=parent_hash,candidate_sha256=sha(dest),fields=fields))
    assert all(sha(p)==h for p,h in plan["checkpoints"].items())
    frozen = STUDY/"frozen-controller.json"
    save(frozen,dict(passed=True,changed=changes,runtime_sources=sources,
         unchanged_checkpoints=len(plan["checkpoints"])-len(changes),parent_plan_sha256=completion["plan_sha256"]))
    seal_layouts(plan,parents,replacements,targets,"SA_A_wide_mixed_r27",2611274100,2611274700,2,
                 "Two bottle detectors and pair 09 camera stop calibration. Grasp and pour trajectories frozen.")
    for report in (PARENT/"plan.json",PARENT/"completion.json",STUDY/"neck-replay.json",STUDY/"stop-fit.json",
                   STUDY/"stop-diagnosis.json",frozen):
        plan["prerequisite_reports"][str(report)] = sha(report)
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    save(STUDY/"plan.json",plan)


if __name__=="__main__":
    main()
