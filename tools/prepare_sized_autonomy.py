"""Build one deployment checkpoint per size from the frozen perception and side-return controller."""
import copy
import json
import sys
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "sized-autonomy"
PARENT = ROOT / "best.pt"
sys.path.insert(0, str(STUDY/"src"))
from pour_vessel_geometry import cup_geometry
from pour_size_autonomy import seeded_prefill, sha


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def main():
    selection = read(ROOT / "followup-selection.json")
    result = read(ROOT / "followup-fresh.json")
    assert selection["all_required_development_gates_passed"] and result["passed_cases"] == result["total_cases"] == 18
    assert result["passed"] and all(sha(ROOT/n) == h for n, h in selection["frozen_files"].items())
    assert sha(PARENT) == read(ROOT / "candidate.json")["checkpoint_sha256"]
    base = torch.load(PARENT, map_location="cpu", weights_only=False)
    locator = torch.load(ROOT/"followup-tracker.pt", map_location="cpu", weights_only=False)
    models = {k:torch.load(ROOT/f"pair-r17-{k}.pt", map_location="cpu", weights_only=False)
              for k in ("rgb","color")}
    assert all(model["accepted"] for model in models.values())
    assert base["cup_detail_calibration"]["K"] == locator["calibration"]["K"]
    assert base["cup_detail_calibration"]["T"] == locator["calibration"]["T"]
    protocol = read(ROOT/"followup-protocol.json")
    template = next(s for s in read("/workspace/learning/A-side-return-20261108-v1/plan.json")["development"]
                    if s["name"] == "SR_A_development_r1_04")
    assert template["policy_evaluation"] and not template["expert_recovery"]
    plan = dict(development=[], test=[], checkpoints={}, parent_sha256=sha(PARENT),
                perception_selection_sha256=sha(ROOT/"followup-selection.json"),
                perception_test_sha256=sha(ROOT/"followup-fresh.json"),
                controller_scope="Autonomous simulated grasp, transfer, visual full stop and side return",
                primary="Verified grasp and correct near-full stop with zero measured particle loss",
                final_height_min=.90, maximum_sample_height=.98, provisional_particle_limit=5)
    for index, (h, d) in enumerate((h,d) for h in (.9,1.,1.1) for d in (.9,1.,1.1)):
        seed = next(s for s in protocol["holdout"] if (s["cup_height_scale"],s["cup_diameter_scale"])==(h,d)
                    and s["coverage_role"]=="moving_low")
        geometry = cup_geometry(seed)
        checkpoint = STUDY / f"cup-{index:02d}.pt"
        saved = copy.copy(base)
        saved.update(calibration=dict(base["calibration"],cup_geometry=geometry),
                     cup_localization_calibration=dict(locator["calibration"],cup_geometry=geometry),
                     cup_detail_calibration=dict(locator["calibration"],cup_geometry=geometry),
                     sized_cup_perception=dict(tracker=locator,models=models,filters=selection["filters"],
                                               agreement_limit=protocol["gates"]["crosscheck_max_disagreement"]),
                     side_return=dict(base["side_return"],rim_z_m=geometry["rim_z_m"]),
                     pour_mouth_z=.054 + .07*max(0.,h-1.) + .02*max(0.,d-1.),
                     thresholds=dict(base["thresholds"],full_settled_target=.925),
                     size_curriculum_native_accepted=True,
                     model_version="visual_guard_sized_cup_v1")
        saved["inputs"]=["task RGB", "wrist RGB", "1024px cup-detail RGB", "joint positions and velocities",
                         "declared cup dimensions", "fixed calibration and controller history"]
        assert not checkpoint.exists()
        torch.save(saved,checkpoint)
        plan["checkpoints"][str(checkpoint)] = sha(checkpoint)
        for split in (["development","test"] if index in (0,8) else ["test"]):
            scene=copy.deepcopy(template)
            name=f"SA_A_{split}_r0_{index:02d}"
            delta=(index-4)*.00045
            cup=scene["cup_world"]
            cup[0]+=delta + (.003 if split=="test" else -.003)
            cup[1]+=(-1 if index%2 else 1)*.002
            source=scene["source_xy"]
            source[1]+=.002 if split=="test" else -.002
            scene.update(name=name,study_split=split,checkpoint=str(checkpoint),seed=2611115300+index+100*(split=="test"),
                         cup_height_scale=h,cup_diameter_scale=d,declared_cup_geometry=geometry,
                         brim_nominal_ml=geometry["legacy_brim_nominal_ml"],mouth_z=saved["pour_mouth_z"],
                         calibration_particle_seed=seed["calibration_particle_seed"],
                         calibration_particle_seed_sha256=seed["calibration_particle_seed_sha256"],
                         calibration_initial_height=.55,size_autonomy_prefill=True,initial_ml=92.,
                         expected_stop="cup_near_full",record_overview=False,save_particle_trajectory=False,
                         save_scene=False)
            scene["cup_initial_ml"]=len(seeded_prefill(scene))*scene["spacing"]**3*1e6
            plan[split].append(scene)
    assert len(plan["development"])==2 and len(plan["test"])==9
    write(STUDY/"plan.json",plan)
    print("Prepared", len(plan["checkpoints"]), "size-aware checkpoints", flush=True)


if __name__=="__main__":
    main()
