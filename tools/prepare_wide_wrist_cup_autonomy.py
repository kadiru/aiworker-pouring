"""Replay the wide-view wrist and cup confirmation, then seal one live repeat."""
import copy
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "wide-mixed-r4"
STUDY = ROOT / "wide-mixed-r5"
HOST = Path("/home/kadir/aiworker-pouring-4070")
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import sha
from pour_source_watchdog import TallBottleEmptyGate
from prepare_wide_mixed_autonomy import read,save

ARCHIVES = (
    ("wide-mixed-r4","development"), ("wide-mixed-r2","development"),
    ("layout-repair-r3","test"), ("layout-repair-r2","development"),
    ("tall-empty-r1","test"), ("tall-empty-r1","development"),
    ("bottle-empty-autonomy","development"), ("mixed-vessel-autonomy","development"),
    ("bottle-autonomy","test"), ("bottle-autonomy","development"),
)


def replay(case):
    folder=Path("/workspace")/Path(case["path"]).relative_to(HOST)
    decisions,samples=folder/"policy-decisions.json",folder/"samples.json"
    labels={round(r["t"],6):r["bottle_ml"] for r in read(samples)}
    zero=next((t for t, ml in labels.items() if ml==0.),None)
    volume=read(folder/"parameters.json")["declared_cup_geometry"]["legacy_brim_nominal_ml"]
    gate=TallBottleEmptyGate(use_aligned_scene=True,use_wrist_and_cup=True)
    rows=0
    candidate=route=None
    for row in read(decisions):
        views=row.get("empty_views") or {}
        cup=row.get("sized_cup_perception") or {}
        if not views:
            continue
        rows=rows+1 if cup.get("valid") else 0
        if gate.update(held=bool(views.get("encoder_hold")),level_ready=rows>=9 and cup.get("agreement",False),
                       present=row["source_present_probability"],wrist_empty=views["wrist_crop_empty"],
                       scene_empty=views["raw_empty"],aligned_empty=views["aligned_empty"],
                       tilt=row["estimated_tilt_from_joints"],flow_ml_s=row["visual_flow_fraction_s"]*volume):
            candidate=round(row["t"],6)
            route=gate.scene_view
            assert zero is not None and labels[candidate]==0. and candidate>=zero,(case["name"],candidate)
            break
        if row["stop_latched"]:
            break
    lag=round(candidate-zero,6) if candidate is not None else None
    return dict(name=case["name"], recorded_stop=case["stop_reason"], first_zero_s=zero,
                candidate_lag_s=lag, candidate_route=route, source_at_candidate_ml=labels.get(candidate),
                samples_sha256=sha(samples),decisions_sha256=sha(decisions))


def main():
    torch.set_num_threads(2)
    prior_plan, prior_results=PREVIOUS/"plan.json",PREVIOUS/"development-results.json"
    plan, development=read(prior_plan),read(prior_results)
    assert development["plan_sha256"]==sha(prior_plan) and development["completed"]==3
    assert development["tolerance_passes"]==2
    rows=[]
    for study,part in ARCHIVES:
        path=ROOT/study/f"{part}-results.json"
        report=read(path)
        assert report["plan_sha256"]==sha(path.with_name("plan.json"))
        rows.extend(replay(row) for row in report["cases"]
                    if row["bottle_geometry"]["lip_height_m"]>=.14 and row["grasp_verified"])
    for suffix in ("00_swap","03_swap"):
        case=next(r for r in rows if r["name"]==f"SA_A_wide_mixed_r4_development_{suffix}")
        assert 0.<=case["candidate_lag_s"]<=2. and case["candidate_route"]=="wrist_and_cup"
    replay_path=STUDY/"empty-replay.json"
    previous_replay=read(PREVIOUS/"empty-replay.json")
    save(replay_path,dict(aligned_gate=previous_replay,wide_wrist_cup_replay=rows,
                          method="Fourteen consecutive frames with 0.94 wrist probability, 120° tilt, grip and tracked low-flow cup"))
    plan["empty_confirmation_replay_sha256"]=sha(replay_path)
    prior_geometry=PREVIOUS/"geometry.json"
    plan["prerequisite_reports"].update({str(prior_plan):sha(prior_plan),str(prior_geometry):sha(prior_geometry),
                                         str(prior_results):sha(prior_results)})
    scene=copy.deepcopy(next(s for s in plan["development"] if s["name"].endswith("_03_swap")))
    scene.update(name="SA_A_wide_mixed_r5_development_03_swap",seed=2611125703)
    plan["development"]=[scene]
    weights={}
    for spec in plan["development"]+plan["test"]:
        source=spec["checkpoint"]
        assert sha(source)==plan["checkpoints"][source]
        if spec["bottle_height_scale"]<=1.:
            weights[source]=sha(source)
            continue
        path=STUDY/Path(source).name
        if str(path) not in weights:
            assert not path.exists()
            saved=torch.load(source,map_location="cpu",weights_only=False)
            saved.update(tall_empty_wrist_cup_confirmation=True,model_version="visual_guard_wide_wrist_cup_r5")
            torch.save(saved,path)
            weights[str(path)]=sha(path)
        spec["checkpoint"]=str(path)
    plan["checkpoints"]=weights
    shutil.copy2(PREVIOUS/"calibration.json",STUDY/"calibration.json")
    save(STUDY/"plan.json",plan)
    print(json.dumps(dict(replayed=len(rows),confirmation="zero particles at every replayed latch")),flush=True)


if __name__=="__main__":
    main()
