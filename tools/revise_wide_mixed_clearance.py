"""Replace the vessel combination in one unscreened pair, keeping both positions."""
import shutil

from prepare_wide_mixed_autonomy import ROOT, STUDY as PREVIOUS, endpoints, make_scene, read, save, sha

STUDY = ROOT / "wide-mixed-r2"


def main():
    previous_plan = PREVIOUS / "plan.json"
    previous_geometry = PREVIOUS / "geometry.json"
    plan, geometry = read(previous_plan), read(previous_geometry)
    assert geometry["plan_sha256"] == sha(previous_plan)
    rejected = [r for r in geometry["cases"] if not r["passed"]]
    assert len(rejected) == 1 and rejected[0]["name"].endswith("_test_14_swap")
    assert rejected[0]["failures"] == ["upright 90.0: arm/cup"]
    assert not (PREVIOUS/"development-results.json").exists()
    donor = plan["test"][0]
    assert (donor["cup_height_scale"],donor["cup_diameter_scale"]) == (.9,.9)
    for i,scene in enumerate(plan["test"]):
        if scene["pair_index"] != 14:
            continue
        cup_xy,bottle_xy=endpoints(scene)
        variant=make_scene(donor,donor,scene,donor["checkpoint"],14,scene["seed"],"test",
                           cup_xy,bottle_xy,scene["expected_stop"])
        variant["name"]=scene["name"]
        variant["source_xy"]=scene["source_xy"]
        plan["test"][i]=variant
    assert len({(s["cup_height_scale"],s["cup_diameter_scale"]) for s in plan["test"]})==9
    assert len({(s["bottle_height_scale"],s["bottle_diameter_scale"]) for s in plan["test"]})==4
    used={s["checkpoint"] for s in plan["development"]+plan["test"]}
    plan["checkpoints"]={path:digest for path,digest in plan["checkpoints"].items() if path in used}
    plan["prerequisite_reports"].update({str(previous_plan):sha(previous_plan),
                                         str(previous_geometry):sha(previous_geometry)})
    for file in ("calibration.json","empty-replay.json"):
        shutil.copy2(PREVIOUS/file,STUDY/file)
    save(STUDY/"plan.json",plan)
    print("Pair 14 retains both endpoints with the already calibrated small cup and short bottle.",flush=True)


if __name__ == "__main__":
    main()
