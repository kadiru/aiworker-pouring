"""Prepare one full and one empty pilot with simultaneous fixture changes."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CUPS = ROOT/"sized-autonomy-r1"
BOTTLES = ROOT/"bottle-autonomy"
EMPTY = ROOT/"bottle-empty-autonomy"
STUDY = ROOT/"mixed-vessel-autonomy"
sys.path.insert(0, str(STUDY/"src"))
from bottle_pour import initial_points
from pour_size_autonomy import seeded_prefill, sha


def read(path):
    return json.loads(path.read_text())


def load_checkpoint(plan, scene):
    path = scene["checkpoint"]
    assert sha(path) == plan["checkpoints"][path]
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    cup_plan = read(CUPS/"plan.json")
    bottle_plan = read(BOTTLES/"plan.json")
    accepted = [(CUPS/"test-results.json", 9), (BOTTLES/"test-results.json", 4),
                (EMPTY/"development-results.json", 4)]
    for path, count in accepted:
        report = read(path)
        assert report["completed"] == report["strict_passes"] == count
        assert report["plan_sha256"] == sha(path.with_name("plan.json"))
        assert all(row["measured_loss_particles"] == 0 for row in report["cases"])
    assert read(EMPTY/"completion.json")["final_zero_source_cases"] == 4

    plan = {k:v for k,v in cup_plan.items() if k not in ("development","test","checkpoints")}
    plan.update(development=[], test=[], checkpoints={},
                prerequisite_reports={str(path):sha(path) for path,_ in accepted},
                controller_scope="First combined-size and placement pilot, known-size bottle and cup",
                primary="Verified grasp, correct visual stop, ordered return and zero measured loss")
    receipt = STUDY/"calibration.json"
    assert not receipt.exists()
    shutil.copy2(CUPS/"calibration.json",receipt)
    assert sha(receipt) == plan["settled_rgb_calibration"]["receipt_sha256"]
    pilots = (
        ("full", (1.1,1.1), (.925,.95), (-.006,.007), (.004,-.004)),
        ("empty", (.9,1.), (1.075,1.), (.007,-.009), (-.003,.003)),
    )
    for index,(goal,cup_size,bottle_size,cup_shift,source_shift) in enumerate(pilots):
        cup = next(s for s in cup_plan["test"] if (s["cup_height_scale"],s["cup_diameter_scale"])==cup_size)
        bottle = next(s for s in bottle_plan["test"] if (s["bottle_height_scale"],s["bottle_diameter_scale"])==bottle_size)
        saved = load_checkpoint(cup_plan,cup)
        base = load_checkpoint(bottle_plan,bottle)
        assert all(np.array_equal(saved["calibration"][k],base["calibration"][k]) for k in ("K","T","table_z"))
        for key in ("source_bank","source_tracking_bank","bottle_geometry"):
            saved[key] = base[key]
        saved["inputs"] = base["inputs"]
        saved["model_version"] = "visual_guard_mixed_declared_vessels_v1"
        weight = STUDY/(goal+".pt")
        assert not weight.exists()
        torch.save(saved,weight)
        plan["checkpoints"][str(weight)] = sha(weight)
        scene = copy.deepcopy(cup)
        for key in ("bottle_size_autonomy","bottle_height_scale","bottle_diameter_scale","declared_bottle_geometry"):
            scene[key] = bottle[key]
        scene.update(name="SA_A_mixed_development_r0_"+goal, study_split="development",
                     expected_stop="cup_near_full" if goal=="full" else "bottle_empty",
                     initial_ml=80. if goal=="full" else 28.,
                     calibration_initial_height=.55 if goal=="full" else .26,
                     seed=2611121600+index, checkpoint=str(weight))
        for axis in (0,1):
            scene["cup_world"][axis] += cup_shift[axis]
            scene["source_xy"][axis] += source_shift[axis]
        quantum = scene["spacing"]**3*1e6
        scene["cup_initial_ml"] = len(seeded_prefill(scene))*quantum
        initial_points(scene)
        plan["development"].append(scene)
        print(json.dumps(dict(name=scene["name"],cup_size=cup_size,bottle_size=bottle_size,
                              cup_initial_ml=scene["cup_initial_ml"],source_ml=scene["initial_ml"])),flush=True)
    with (STUDY/"plan.json").open("x") as file:
        json.dump(plan,file,indent=2,allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
