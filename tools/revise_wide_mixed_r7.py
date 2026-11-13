"""Replay the cup outline ambiguity and seal a candidate with guarded fusion."""
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "wide-mixed-r7"
STUDY = ROOT / "wide-mixed-r7b"
sys.path.insert(0, str(STUDY/"src"))
from pour_guard_vision import cup_crop, plane_point, project
from pour_size_autonomy import SizedCupPerception, sha
from fit_wide_mixed_r6 import decode, folder, h5py, read, save
sys.path.insert(0, str(STUDY/"src"))

NARROW_CUP = (1.,.9)
JUMP_GUARD = dict(max_outline_jump_px=10., max_neural_gap_px=6.)


def error(anchor, z, calibration, truth):
    if anchor is None:
        return None
    return float(np.linalg.norm(plane_point(anchor,z,calibration)[:2]-truth)*1000.)


def record(path, device):
    spec = read(path/"parameters.json")
    state = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    geometry = spec["declared_cup_geometry"]
    original = state["sized_cup_perception"]
    assert original["outline_cup_localization"]
    updated = copy.deepcopy(original)
    updated["tracker"]["outline_jump_guard"] = JUMP_GUARD
    baseline, guarded, missing = [SizedCupPerception(config,geometry,device)
                                  for config in (original,updated,updated)]
    cal = updated["tracker"]["calibration"]
    xy = np.asarray(spec["cup_world"][:2])
    z = cal["table_z"] + geometry["rim_z_m"]
    x,y = np.rint(project([*xy,z],cal)).astype(int)
    target = spec["name"].endswith("12_swap")
    missing_acceptances = 0
    with h5py.File(path/"episode.hdf5") as file:
        frames = file["data/demo_0/obs/cup_detail_image"]
        for i in range(15):
            frame = decode(frames[i],spec["image_encoding"])
            baseline.track(frame)
            guarded.track(frame)
            blank = frame.copy()
            blank[max(0,y-75):y+160,max(0,x-140):x+140] = 180
            missing_acceptances += int(missing.track(blank)["valid"])
        prior = error(baseline.tracker.anchor,z,cal,xy)
        new = error(guarded.tracker.anchor,z,cal,xy)
        result = dict(name=spec["name"], recording_sha256=sha(path/"episode.hdf5"),
                      old_error_mm=prior,new_error_mm=new,missing_acceptances=missing_acceptances,
                      target=target)
        if target and len(frames)>400 and new is not None:
            from perception_followup import predict_level
            decisions = read(path/"policy-decisions.json")
            samples = {round(s["t"],6):s["cup_fill_fraction"] for s in read(path/"samples.json")}
            for label, model in (("old",baseline),("new",guarded)):
                estimate = plane_point(model.tracker.anchor,z,cal)[:2]
                errors, gaps = [], []
                for i in (300,400,min(len(frames)-1,650)):
                    frame = decode(frames[i],spec["image_encoding"])
                    crop = cup_crop(frame,estimate,dict(cal,cup_geometry=geometry))
                    pixels = torch.as_tensor(crop,device=device)[None]
                    pair = [float(predict_level(pixels,estimate,geometry,branch)[0][0])
                            for branch in updated["models"].values()]
                    measured = samples[round(decisions[i]["t"],6)]
                    errors.extend(abs(v-measured) for v in pair)
                    gaps.append(abs(pair[0]-pair[1]))
                result[label+"_level_mae"] = float(np.mean(errors))
                result[label+"_maximum_branch_gap"] = max(gaps)
    return result


def replay(device):
    older = ROOT/"wide-mixed-r6"
    cases = read(older/"completion.json")["test_cases"]
    paths = [folder(c) for c in cases if tuple(c["cup_size"]) == NARROW_CUP]
    first_dev = Path("/workspace/outputs/size-autonomy/wide-r7-dev")
    targets = list(first_dev.glob("*/*12_swap*/episode.hdf5"))
    assert len(targets) == 1
    paths.append(targets[0].parent)
    results = [record(p,device) for p in paths]
    checks = [r for r in results if r["target"]]
    passes = (len(results) == 5 and len(checks) == 2
              and all(r["new_error_mm"] is not None and r["new_error_mm"]<=2.5
                      and not r["missing_acceptances"] for r in results)
              and all(r["new_error_mm"]<r["old_error_mm"] for r in checks)
              and results[-1]["new_level_mae"] < results[-1]["old_level_mae"]
              and results[-1]["new_maximum_branch_gap"] < .035)
    save(STUDY/"cup-anchor-replay.json",dict(passed=passes,settings=JUMP_GUARD,cases=results))
    assert passes,results


def seal():
    receipt = STUDY/"cup-anchor-replay.json"
    assert read(receipt)["passed"]
    plan_path = PARENT/"plan.json"
    plan = read(plan_path)
    original = plan["checkpoints"]
    checkpoints = {}
    current = {}
    predecessors = {}
    for part in ("development","test"):
        for spec in plan[part]:
            path = spec["checkpoint"]
            if path not in current:
                assert sha(path) == original[path]
                cup = spec["cup_height_scale"],spec["cup_diameter_scale"]
                if cup == NARROW_CUP:
                    state = torch.load(path,map_location="cpu",weights_only=False)
                    state["sized_cup_perception"]["tracker"]["outline_jump_guard"] = JUMP_GUARD
                    dest = STUDY/Path(path).name
                    assert not dest.exists()
                    torch.save(state,dest)
                    current[path] = str(dest)
                else:
                    current[path] = path
                checkpoints[current[path]] = sha(current[path])
            spec["checkpoint"] = current[path]
            if part == "development":
                previous = spec["name"]
                spec["name"] = previous.replace("_r7_development_", "_r7b_development_")
                predecessors[spec["name"]] = previous
    for file in ("neck-fusion.json","level-fit.json","empty-replay.json","calibration.json"):
        shutil.copy2(PARENT/file,STUDY/file)
    plan["development_empty_lag_cases"] = [s.replace("_r7_development_","_r7b_development_")
                                           for s in plan["development_empty_lag_cases"]]
    plan["checkpoints"] = checkpoints
    plan["clearance_predecessors"] = predecessors
    plan["prerequisite_reports"].update({str(p):sha(p) for p in (plan_path,receipt,STUDY/"neck-fusion.json",
                                                               STUDY/"level-fit.json")})
    plan["controller_scope"] += " with guarded narrow-cup outline refinement"
    save(STUDY/"plan.json",plan)


if __name__ == "__main__":
    torch.set_num_threads(1)
    if not (STUDY/"cup-anchor-replay.json").exists():
        replay("cuda" if torch.cuda.is_available() else "cpu")
    if "replay-only" not in sys.argv:
        seal()
