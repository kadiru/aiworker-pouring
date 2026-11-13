"""Enable the audited rim refinement and keep the thirty reserved layouts frozen."""
import copy
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "wide-mixed-r4"
PREVIOUS = ROOT / "wide-mixed-r2"
sys.path.insert(0, str(STUDY/"src"))
from prepare_wide_mixed_autonomy import read, save, sha


def main():
    torch.set_num_threads(2)
    source_plan = PREVIOUS/"plan.json"
    results_path = PREVIOUS/"development-results.json"
    plan, results = read(source_plan), read(results_path)
    assert results["plan_sha256"]==sha(source_plan) and results["completed"]==len(plan["development"])
    failures = {row["name"] for row in results["cases"] if not row["within_particle_tolerance"]}
    assert any(name.endswith("_01_normal") for name in failures)
    replay = STUDY/"cup-outline-replay.json"
    evidence = read(replay)
    assert len(evidence["cases"])==28 and evidence["max_outline_error_mm"]<=2.5
    assert evidence["missing_cup_acceptances"]==0
    consumed = {s["name"] for s in plan["test"]}
    assert not any(read(path)["name"] in consumed
                   for path in Path("/workspace/outputs/size-autonomy").glob("*/*/*SA_A_*/parameters.json"))
    # Replay the observed failures and one completed swapped case.
    selected = failures | {plan["development"][0]["name"]}
    development = []
    for i, scene in enumerate(plan["development"]):
        if scene["name"] not in selected:
            continue
        spec = copy.deepcopy(scene)
        spec.update(name=scene["name"].replace("_r1_development_","_r4_development_"),seed=2611125600+i)
        development.append(spec)
    plan["development"] = development
    weights = {}
    for scene in plan["development"]+plan["test"]:
        previous = scene["checkpoint"]
        if previous not in weights:
            assert sha(previous) == plan["checkpoints"][previous]
            saved = torch.load(previous,map_location="cpu",weights_only=False)
            saved["sized_cup_perception"]["outline_cup_localization"] = True
            saved["preflow_occlusion_guard"] = True
            saved["side_return"] = dict(saved["side_return"],retime_upright=True)
            saved["model_version"] = "visual_guard_wide_outline_r4"
            path = STUDY/Path(previous).name
            assert not path.exists()
            torch.save(saved,path)
            weights[previous] = str(path)
        scene["checkpoint"] = weights[previous]
    plan["checkpoints"] = {path:sha(path) for path in weights.values()}
    plan["prerequisite_reports"].update({str(source_plan):sha(source_plan),str(results_path):sha(results_path),
                                         str(replay):sha(replay)})
    for name in ("calibration.json","empty-replay.json"):
        shutil.copy2(PREVIOUS/name, STUDY/name)
    save(STUDY/"plan.json",plan)
    print(f"Prepared {len(development)} development repeats and {len(plan['test'])} untouched reserved layouts.",
          flush=True)


if __name__=="__main__":
    main()
