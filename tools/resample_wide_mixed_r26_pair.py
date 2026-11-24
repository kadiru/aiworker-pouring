"""Replace a collision-screened pair before any autonomous trial, retaining the first plan."""
from copy import deepcopy
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from preflight_sized_autonomy import STUDY, collision_meshes, scan, sha
from prepare_wide_mixed_r10 import shift
from fit_wide_mixed_r6 import read, save

SEED = 2611244808


def main():
    assert STUDY.name=="wide-mixed-r26" and not list(STUDY.glob("*-batch.json"))
    assert not list(STUDY.glob("*-results.json"))
    torch.set_num_threads(1)
    plan, geometry = read(STUDY/"plan.json"), read(STUDY/"geometry.json")
    assert geometry["plan_sha256"] == sha(STUDY/"plan.json")
    failures = [c for c in geometry["cases"] if not c["passed"]]
    assert [(c["name"].split("_test_")[1],c["failures"]) for c in failures]==[("08_swap",["approach: arm/cup"])]
    parent = read(STUDY.parent/"wide-mixed-r25"/"plan.json")
    pair = [s for s in plan["test"] if s["pair_index"]==8]
    reference = [s for s in parent["test"] if s["pair_index"]==8]
    assert [s["swapped"] for s in pair]==[False,True]
    assert [s["swapped"] for s in reference]==[False,True]
    assert all(sha(s["checkpoint"])==plan["checkpoints"][s["checkpoint"]] for s in pair)
    states = [torch.load(s["checkpoint"],map_location="cpu",weights_only=False) for s in pair]
    stage, opened, closed = collision_meshes()
    rng = np.random.default_rng(SEED)
    attempts = []
    for draw in range(1,17):
        offsets = rng.uniform(.0011,.0025,(2,2))*rng.choice((-1.,1.),(2,2))
        scenes = deepcopy(pair)
        checks = []
        for s, base, state in zip(scenes,reference,states):
            for k in ("cup_world","source_xy"):
                s[k] = deepcopy(base[k])
            shift(s,*(offsets[::-1] if s["swapped"] else offsets))
            checks.append(scan(stage,s,state,opened,closed))
        attempts.append(dict(draw=draw,offsets_m=offsets.tolist(),checks=checks))
        if all(c["passed"] for c in checks):
            break
    else:
        raise RuntimeError("No collision-free matched placement in the scheduled draws")
    report = STUDY/"pair-08-clearance.json"
    save(report,dict(seed=SEED,initial_plan_sha256=geometry["plan_sha256"],initial_geometry_sha256=sha(STUDY/"geometry.json"),
                     passed=True,selected_draw=draw,attempts=attempts))
    for filename in ("plan.json","geometry.json","geometry-progress.jsonl"):
        shutil.copy2(STUDY/filename,STUDY/("first-"+filename))
    for old,new in zip(pair,scenes):
        assert all(v==new[k] for k,v in old.items() if k not in ("cup_world","source_xy"))
        old.update(new)
    for name in ("first-plan.json","first-geometry.json","pair-08-clearance.json"):
        p = STUDY/name
        plan["prerequisite_reports"][str(p)] = sha(p)
    updated = STUDY/"replacement-plan.json"
    save(updated,plan)
    os.replace(updated,STUDY/"plan.json")
    for name in ("geometry.json","geometry-progress.jsonl"):
        assert sha(STUDY/name)==sha(STUDY/("first-"+name))
        (STUDY/name).unlink()
    print("Selected matched placement",draw,"before autonomous testing",flush=True)


if __name__=="__main__":
    main()
