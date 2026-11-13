"""Freeze a size-aware wide benchmark with fifteen exact position-swap pairs."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "wide-mixed-r1"
LATENCY = ROOT / "tall-aligned-r0"
MIXED = ROOT / "layout-repair-r3"
FINISH = ROOT.parent / "A-finish-20261106-v1" / "frozen-plan.json"
sys.path.insert(0, str(STUDY / "src"))

from bottle_pour import initial_points
from pour_size_autonomy import seeded_prefill, sha

BOTTLES = ((.925, 1.), (1.075, .95), (.925, .95), (1.075, 1.))
CUPS = tuple((h, d) for h in (.9, 1., 1.1) for d in (.9, 1., 1.1))
DEVELOPMENT = ((1, (1., 1.), (1.075, 1.)), (2, (1.1, .9), (.925, 1.)),
               (5, (.9, 1.1), (.925, .95)), (7, (1.1, .9), (1.075, .95)))


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def key(scene, vessel):
    return scene[vessel+"_height_scale"], scene[vessel+"_diameter_scale"]


def checkpoint(plan, scene):
    path = scene["checkpoint"]
    assert sha(path) == plan["checkpoints"][path]
    return torch.load(path, map_location="cpu", weights_only=False)


def make_scene(cup, bottle, layout, weight, pair, seed, split, cup_xy, bottle_xy, goal):
    scene = copy.deepcopy(cup)
    for field in ("bottle_size_autonomy", "bottle_height_scale", "bottle_diameter_scale",
                  "declared_bottle_geometry"):
        scene[field] = bottle[field]
    scene.update(name=f"SA_A_wide_mixed_r1_{split}_{pair:02}_{'swap' if layout['swapped'] else 'normal'}",
                 study_split=split, seed=seed, checkpoint=str(weight), pair_index=pair,
                 layout_family=layout["layout_family"], layout_label=layout["layout_label"],
                 parent_layout_family=layout["name"], swapped=layout["swapped"],
                 cup_world=[*cup_xy, cup["cup_world"][2]],
                 source_xy=[bottle_xy[1]-cup_xy[1], cup_xy[0]-bottle_xy[0]],
                 expected_stop=goal, initial_ml=80. if goal=="cup_near_full" else 28.,
                 calibration_initial_height=.55 if goal=="cup_near_full" else .26)
    scene["cup_initial_ml"] = len(seeded_prefill(scene))*scene["spacing"]**3*1e6
    initial_points(scene)
    return scene


def endpoints(layout):
    x, y = layout["cup_world"][:2]
    longitudinal, lateral = layout["source_xy"]
    return np.array([[x,y],[x-lateral,y+longitudinal]])


def main():
    torch.set_num_threads(2)
    timing_plan = LATENCY / "plan.json"
    timing_result = LATENCY / "development-results.json"
    timing = read(timing_result)
    assert timing["plan_sha256"] == sha(timing_plan) and timing["completed"] == timing["tolerance_passes"] == 1
    assert 0. <= timing["cases"][0]["empty_confirmation_lag_s"] <= 2.
    baseline = MIXED / "plan.json"
    previous = read(baseline)
    cup_plan = read(ROOT / "sized-autonomy-r1" / "plan.json")
    bottle_plan = read(ROOT / "bottle-autonomy" / "plan.json")
    reference = read(FINISH)
    cups = {key(scene, "cup"):scene for scene in cup_plan["test"]}
    bottles = {key(scene, "bottle"):scene for scene in bottle_plan["test"]}
    assert set(cups)==set(CUPS) and set(bottles)==set(BOTTLES)
    weights = {}
    plan = {k:v for k,v in previous.items() if k not in ("development","test","checkpoints")}
    plan.update(development=[], test=[], checkpoints={}, development_requires_strict_zero=False,
                controller_scope="Size-aware visual autonomy on five wider families and fifteen exact position swaps",
                prerequisite_reports={**previous["prerequisite_reports"], str(timing_plan):sha(timing_plan),
                                      str(timing_result):sha(timing_result), str(FINISH):sha(FINISH)})
    shutil.copy2(MIXED/"calibration.json", STUDY/"calibration.json")
    shutil.copy2(LATENCY/"empty-replay.json", STUDY/"empty-replay.json")
    plan["empty_confirmation_replay_sha256"]=sha(STUDY/"empty-replay.json")
    midfill = read(MIXED/"empty-replay.json")["midfill_level"]["bias"]
    prior_tall = checkpoint(previous, previous["test"][0])
    fallback = dict(prior_tall["source_initial_fallback"], presence_bounds_only=True)

    def weight(cup_size, bottle_size):
        pairing = (cup_size,bottle_size)
        if pairing in weights:
            return weights[pairing]
        c, b = cups[cup_size], bottles[bottle_size]
        saved, source = checkpoint(cup_plan,c), checkpoint(bottle_plan,b)
        assert all(np.array_equal(saved["calibration"][k],source["calibration"][k]) for k in ("K","T","table_z"))
        for k in ("source_bank","source_tracking_bank","bottle_geometry","inputs"):
            saved[k] = source[k]
        saved["sized_cup_perception"]["retry_initial_cup"] = True
        saved["side_return"] = dict(saved["side_return"],early_lift_arc=True)
        if cup_size == (1.1,.9):
            saved["sized_cup_perception"]["midfill_rgb_calibration"] = dict(
                geometry=c["declared_cup_geometry"], bias=midfill)
        if bottle_size[0] > 1.:
            saved.update(tall_empty_confirmation=True, tall_empty_aligned_confirmation=True,
                         source_initial_fallback=fallback)
        else:
            saved["short_empty_requires_drain"] = True
            if bottle_size == (.925,.95) and cup_size == (1.,1.1):
                saved["tail_setback_m"] = .008
        saved["model_version"] = "visual_guard_wide_mixed_r1"
        path = STUDY / f"cup-{CUPS.index(cup_size):02}-bottle-{BOTTLES.index(bottle_size):02}.pt"
        assert not path.exists()
        torch.save(saved,path)
        plan["checkpoints"][str(path)] = sha(path)
        weights[pairing] = path
        return path

    for dev, (index, cup_size, bottle_size) in enumerate(DEVELOPMENT):
        layout = reference["development"][index]
        cup_xy, bottle_xy = endpoints(layout)
        plan["development"].append(make_scene(cups[cup_size],bottles[bottle_size],layout,
            weight(cup_size,bottle_size),dev,2611125200+dev,"development",cup_xy,bottle_xy,layout["expected_stop"]))
    generator = np.random.default_rng(2611125300)
    for pair in range(15):
        normal, swapped = reference["test"][2*pair:2*pair+2]
        points = endpoints(normal)
        assert not normal["swapped"] and swapped["swapped"]
        assert np.allclose(points, endpoints(swapped)[::-1], rtol=0, atol=1e-12)
        # Fresh endpoint offsets are shared by the exact reversed scene.
        points += generator.uniform(.0015,.0035,points.shape)*generator.choice((-1,1),points.shape)
        cup_size, bottle_size = CUPS[pair%len(CUPS)], BOTTLES[pair%len(BOTTLES)]
        path = weight(cup_size,bottle_size)
        for orientation, layout in enumerate((normal,swapped)):
            cup_xy,bottle_xy = points[::(-1 if orientation else 1)]
            plan["test"].append(make_scene(cups[cup_size],bottles[bottle_size],layout,path,
                pair,2611125400+2*pair+orientation,"test",cup_xy,bottle_xy,layout["expected_stop"]))
    assert len(plan["test"]) == 30 and sum(s["swapped"] for s in plan["test"]) == 15
    assert sum(s["expected_stop"]=="bottle_empty" for s in plan["test"]) == 15
    assert len({key(s,"cup") for s in plan["test"]})==9 and len({key(s,"bottle") for s in plan["test"]})==4
    plan["test_shards"] = [[s["name"] for s in plan["test"][i:i+10]] for i in (0,10,20)]
    save(STUDY/"plan.json",plan)
    print(json.dumps(dict(development=len(plan["development"]),test=len(plan["test"]),
                          checkpoints=len(plan["checkpoints"]),shards=list(map(len,plan["test_shards"])))),
          flush=True)


if __name__ == "__main__":
    main()
