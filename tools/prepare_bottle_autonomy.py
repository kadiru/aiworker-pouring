"""Freeze short and tall bottle fixtures using the accepted standard-cup policy."""
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT/"sized-autonomy-r1"
STUDY = ROOT/"bottle-autonomy"
sys.path.insert(0, str(STUDY/"src"))
from bottle_pour import LIP, START_X, initial_points, vessel_mask
from pour_bottle_geometry import declared_geometry, profile_for
from pour_guard_vision import plane_point, project
from pour_size_autonomy import sha


def read(path):
    return json.loads(path.read_text())


def retarget_source_bank(bank, calibration, lip_height):
    """Reproject the template search region to the new declared lip plane."""
    def shift(uv):
        xyz = plane_point(uv, calibration["table_z"] + LIP, calibration)
        return project(xyz + [0., 0., lip_height-LIP], calibration).tolist()

    moved = dict(bank)
    if bank.get("alternatives"):
        moved["alternatives"] = [retarget_source_bank(b, calibration, lip_height)
                                 for b in bank["alternatives"]]
        return moved
    x0,y0,x1,y1 = bank["bounds"]
    corners = np.asarray([shift([x,y]) for x in (x0,x1) for y in (y0,y1)])
    moved["bounds"] = [*map(math.floor,corners.min(axis=0)), *map(math.ceil,corners.max(axis=0))]
    moved["anchors"] = [shift(uv) for uv in bank["anchors"]]
    if bank.get("initial_region_uv") is not None:
        moved["initial_region_uv"] = [shift(uv) for uv in bank["initial_region_uv"]]
    return moved


def main():
    parent_plan = read(PARENT/"plan.json")
    parent_results = read(PARENT/"test-results.json")
    assert parent_results["plan_sha256"] == sha(PARENT/"plan.json") and parent_results["strict_passes"] == 9
    template = next(s for s in parent_plan["test"] if s["cup_height_scale"] == s["cup_diameter_scale"] == 1.)
    checkpoint = Path(template["checkpoint"])
    assert sha(checkpoint) == parent_plan["checkpoints"][str(checkpoint)]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved.get("sized_cup_perception") and saved.get("side_return") and saved.get("project_pour_kinematics")
    plan = {k:v for k,v in parent_plan.items() if k not in ("development","test","checkpoints","settled_rgb_calibration")}
    plan.update(development=[], test=[], checkpoints={},
                source_checkpoint_sha256=sha(checkpoint),
                source_task_results_sha256=sha(PARENT/"test-results.json"),
                development_requires_strict_zero=True,
                controller_scope="Known-size bottle grasp, visual near-full stop and side return to the table")
    for index,(height,h) in enumerate((("short",.925),("tall",1.075))):
        for width,d in (("nominal",1.),("narrow",.95)):
            tag = f"{height}_{width}"
            scene = copy.deepcopy(template)
            scene.update(bottle_size_autonomy=True, bottle_height_scale=h, bottle_diameter_scale=d,
                         initial_ml=80.)
            fixture = declared_geometry(scene)
            scene["declared_bottle_geometry"] = fixture
            grid = initial_points(scene).copy()
            grid[:,0] -= START_X
            assert vessel_mask(grid, profile_for(scene)).all()
            weight = STUDY/f"{tag}.pt"
            assert not weight.exists()
            variant = dict(saved, bottle_geometry=fixture,
                           model_version="visual_guard_declared_bottle_v1")
            variant["source_tracking_bank"] = saved.get("source_tracking_bank", saved["source_bank"])
            variant["source_bank"] = retarget_source_bank(saved["source_bank"],saved["calibration"],fixture["lip_height_m"])
            variant["inputs"] = [*saved["inputs"], "declared bottle dimensions"]
            torch.save(variant,weight)
            plan["checkpoints"][str(weight)] = sha(weight)
            if width == "nominal":
                development = copy.deepcopy(scene)
                development.update(name=f"SA_A_bottle_development_r0_{tag}", study_split="development",
                                   seed=2611121100+index, checkpoint=str(weight))
                plan["development"].append(development)
            scene.update(name=f"SA_A_bottle_test_r0_{tag}", study_split="test",
                         seed=2611121200+2*index+int(width=="narrow"), checkpoint=str(weight))
            direction = 1 if height == "short" else -1
            scene["cup_world"][0] += .001 * direction
            scene["source_xy"][1] += .001 * direction
            plan["test"].append(scene)
            print(json.dumps(dict(name=tag, geometry=fixture, initial_particles=len(grid))),flush=True)
    with (STUDY/"plan.json").open("x") as file:
        json.dump(plan,file,indent=2,allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
