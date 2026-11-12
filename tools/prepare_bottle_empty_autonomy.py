"""Freeze the four low-inventory bottle-empty cases from the verified bottle study."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PARENT = ROOT/"bottle-autonomy"
STUDY = ROOT/"bottle-empty-autonomy"
sys.path.insert(0, str(STUDY/"src"))
from bottle_pour import initial_points
from pour_size_autonomy import seeded_prefill, sha


def main():
    source = PARENT/"plan.json"
    results = PARENT/"test-results.json"
    plan = json.loads(source.read_text())
    verified = json.loads(results.read_text())
    assert verified["plan_sha256"] == sha(source)
    assert verified["completed"] == verified["strict_passes"] == len(plan["test"]) == 4
    assert {case["name"] for case in verified["cases"] if case["strict_pass"]} == {
        case["name"] for case in plan["test"]}
    assert all(sha(name) == digest for name, digest in plan["checkpoints"].items())

    cases = []
    for index, template in enumerate(plan["test"]):
        scene = copy.deepcopy(template)
        geometry = scene["declared_cup_geometry"]
        scene.update(name=template["name"].replace("_test_r0_", "_empty_r0_"),
                     study_split="development", expected_stop="bottle_empty",
                     initial_ml=28., calibration_initial_height=.26, seed=2611121400+index)
        quantum = scene["spacing"]**3*1e6
        source_ml = len(initial_points(scene))*quantum
        scene["cup_initial_ml"] = len(seeded_prefill(scene))*quantum
        # Margin against the prior observed near-full inventory on this cup.
        assert scene["cup_initial_ml"] + source_ml < .7*geometry["legacy_brim_nominal_ml"]
        cases.append(scene)
        print(json.dumps(dict(name=scene["name"], source_ml=source_ml,
                              cup_initial_ml=scene["cup_initial_ml"])), flush=True)

    plan.update(development=cases, test=[],
                source_bottle_plan_sha256=sha(source), source_bottle_results_sha256=sha(results),
                controller_scope="First bottle-empty checks of all four verified bottle geometries",
                primary="Verified grasp, correct visual empty stop, ordered return and zero measured particle loss")
    with (STUDY/"plan.json").open("x") as file:
        json.dump(plan, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
