"""Seal the targeted cup repair and fresh matched trials with all motion fields frozen."""
from pathlib import Path
import shutil

import torch

from fit_wide_mixed_r26_cup import KEY, PARENT, STUDY
from fit_wide_mixed_r6 import read, save
from pour_size_autonomy import sha
from prepare_wide_mixed_r8 import suffix
from prepare_wide_mixed_r17 import identical
from prepare_wide_mixed_r19 import seal_layouts


def main():
    plan = read(PARENT/"plan.json")
    prev = read(PARENT/"completion.json")
    assert prev["plan_sha256"]==sha(PARENT/"plan.json")
    failed, = [r for r in prev["test_cases"] if not r["strict_pass"]]
    assert suffix(failed)==KEY and failed["stop_reason"]=="localization_unconfirmed"
    assert failed["delivered_to_cup_ml"]==0 and prev["test"]["lost_particles"]==0
    diag = STUDY/"cup-diagnosis.json"
    replay = STUDY/"cup-startup-replay.json"
    proof = read(replay)
    assert proof["passed"] and proof["parent_plan_sha256"]==prev["plan_sha256"]
    parents = {suffix(s):s for s in plan["test"]}
    old = parents[KEY]["checkpoint"]
    assert proof["checkpoint"]==old and all(sha(p)==h for p,h in plan["checkpoints"].items())
    parent_digest = plan["checkpoints"].pop(old)
    before = torch.load(old,map_location="cpu",weights_only=False)
    artifact = STUDY/"cup-selected-07.pt"
    assert sha(artifact)==proof["artifact_sha256"]
    weights = torch.load(artifact,map_location="cpu",weights_only=False)
    cfg = before["sized_cup_perception"]
    assert not identical(weights,cfg["tracker"]["coarse_state"])
    fitted = dict(before,sized_cup_perception=dict(cfg,tracker=dict(cfg["tracker"],coarse_state=weights)))
    path = STUDY/Path(old).name
    torch.save(fitted,path)
    actual = torch.load(path,map_location="cpu",weights_only=False)
    assert identical(actual,fitted)
    # Restore just the fitted leaf in the reloaded object and compare the entire tree.
    actual["sized_cup_perception"]["tracker"]["coarse_state"] = cfg["tracker"]["coarse_state"]
    assert identical(actual,before)
    sources = {p.name:sha(p) for p in (STUDY/"src").glob("*.py")}
    assert sources==read(PARENT/"geometry.json")["sources"]
    unchanged = len(plan["checkpoints"])
    plan["checkpoints"][str(path)] = sha(path)
    frozen = STUDY/"frozen-controller.json"
    save(frozen,dict(passed=True,runtime_sources=sources,unchanged_checkpoints=unchanged,
         changed_field="sized_cup_perception.tracker.coarse_state",original=old,parent_sha256=parent_digest,
         candidate_sha256=sha(path),parent_plan_sha256=prev["plan_sha256"]))
    seal_layouts(plan,parents,{old:str(path)},(KEY,),"SA_A_wide_mixed_r26",2611244100,2611244700,2,
                 "Pair 07 coarse cup heatmap only. All motion, grasp, stop, level and bottle-localization fields frozen.")
    for report in (PARENT/"plan.json",PARENT/"completion.json",diag,replay,STUDY/"blend-selection.json",frozen):
        plan["prerequisite_reports"][str(report)] = sha(report)
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PARENT/name,STUDY/name)
    save(STUDY/"plan.json",plan)
    print("Sealed the original normal 07 placement and 15 fresh matched pairs",flush=True)


if __name__ == "__main__":
    main()
