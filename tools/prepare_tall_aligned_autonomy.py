"""Replay tall-bottle depletion gates and seal a same-layout timing verification."""
import copy
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parent
HOST = Path("/home/kadir/aiworker-pouring-4070")
STUDY = ROOT / "tall-aligned-r0"
PARENT = ROOT / "layout-repair-r3"
sys.path.insert(0, str(STUDY / "src"))

from pour_size_autonomy import MIDFILL_RGB_KNOTS, settled_rgb_weight, sha
from pour_source_watchdog import TallBottleEmptyGate

SOURCES = (
    ("layout-repair-r3", "test"), ("layout-repair-r2", "development"),
    ("tall-empty-r1", "test"), ("tall-empty-r1", "development"),
    ("bottle-empty-autonomy", "development"), ("mixed-vessel-autonomy", "development"),
    ("bottle-autonomy", "test"), ("bottle-autonomy", "development"),
)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def replay(case, midfill_bias):
    folder = Path("/workspace") / Path(case["path"]).relative_to(HOST)
    decisions, samples = folder / "policy-decisions.json", folder / "samples.json"
    labels = {round(x["t"], 6): x["bottle_ml"] for x in read(samples)}
    zero = next((t for t, ml in labels.items() if ml == 0.), None)
    geometry = read(folder / "parameters.json")["declared_cup_geometry"]
    scale = geometry["legacy_brim_nominal_ml"]
    corrected = case["name"] == "SA_A_tall_empty_test_r1_02_empty"
    gate, level_count, accepted, view = TallBottleEmptyGate(use_aligned_scene=True), 0, None, None
    for frame in read(decisions):
        cup = frame.get("sized_cup_perception") or {}
        views = frame.get("empty_views") or {}
        if frame["stop_latched"] and not views:
            break
        if not views:
            continue
        level_count = level_count+1 if cup.get("valid") else 0
        agrees = cup.get("agreement", False)
        if corrected and cup.get("valid") and frame["own_pour_phase_deg"] >= 115.:
            rgb, color = cup["filtered"]["rgb"], cup["filtered"]["color"]
            agrees = abs(rgb-midfill_bias*settled_rgb_weight(rgb, MIDFILL_RGB_KNOTS)-color) < .035
        if gate.update(held=bool(views.get("encoder_hold")), level_ready=level_count>=9 and agrees,
                       present=frame["source_present_probability"], wrist_empty=views["wrist_crop_empty"],
                       scene_empty=views["raw_empty"], aligned_empty=views["aligned_empty"],
                       tilt=frame["estimated_tilt_from_joints"], flow_ml_s=frame["visual_flow_fraction_s"]*scale):
            accepted, view = round(frame["t"], 6), gate.scene_view
            break
        if frame["stop_latched"]:
            break
    lag = round(accepted-zero, 6) if accepted is not None and zero is not None else None
    if case["stop_reason"] == "bottle_empty":
        assert lag is not None and 0. <= lag <= 2. and labels[accepted] == 0., (case["name"], lag)
    else:
        assert case["stop_reason"] == "cup_near_full" and accepted is None, (case["name"], accepted)
    return dict(name=case["name"], stop=case["stop_reason"], first_zero_s=zero, candidate_stop_s=accepted,
                candidate_lag_s=lag, recorded_lag_s=case.get("empty_confirmation_lag_s"),
                source_at_candidate_ml=labels[accepted] if accepted is not None else None,
                scene_view=view, replay_applied_midfill_calibration=corrected,
                decisions_sha256=sha(decisions), samples_sha256=sha(samples))


def main():
    torch.set_num_threads(2)
    parent_path = PARENT / "plan.json"
    plan = read(parent_path)
    parent_results = PARENT / "test-results.json"
    assert read(parent_results)["strict_passes"] == 6
    bias = read(PARENT / "empty-replay.json")["midfill_level"]["bias"]
    reports, rows = {}, []
    for study, part in SOURCES:
        path = ROOT / study / f"{part}-results.json"
        results = read(path)
        assert results["plan_sha256"] == sha(path.with_name("plan.json"))
        reports[str(path)] = sha(path)
        rows.extend(replay(case, bias) for case in results["cases"]
                    if case["bottle_geometry"]["lip_height_m"] >= .14
                    and case["grasp_verified"] and case["policy_completed"])
    positives = [row for row in rows if row["stop"] == "bottle_empty"]
    assert len(positives) == 9 and len(rows)-len(positives) == 7
    late = next(row for row in positives if row["name"] == "SA_A_layout_r2_test_02_empty")
    assert late["recorded_lag_s"] == 5.85 and late["candidate_lag_s"] <= 1.25
    receipt = STUDY / "empty-replay.json"
    save(receipt, dict(positive_episodes=len(positives), near_full_episodes=len(rows)-len(positives),
                       cases=rows, scope="Archived camera/joint decisions with independent simulator labels"))
    shutil.copy2(PARENT / "calibration.json", STUDY / "calibration.json")
    scene = copy.deepcopy(next(s for s in plan["test"] if s["name"] == late["name"]))
    source = scene["checkpoint"]
    assert sha(source) == plan["checkpoints"][source]
    saved = torch.load(source, map_location="cpu", weights_only=False)
    assert saved["tall_empty_confirmation"]
    saved.update(tall_empty_aligned_confirmation=True, model_version="visual_guard_tall_aligned_r0")
    weight = STUDY / "verify.pt"
    assert not weight.exists()
    torch.save(saved, weight)
    scene.update(name="SA_A_tall_aligned_r0_development", study_split="development",
                 seed=2611125102, checkpoint=str(weight))
    plan.update(development=[scene], test=[], checkpoints={str(weight):sha(weight)},
                development_requires_strict_zero=True, development_empty_lag_limit_s=2.,
                prerequisite_reports={**plan["prerequisite_reports"], **reports},
                empty_confirmation_replay_sha256=sha(receipt),
                controller_scope="Same layout with a new seed to verify earlier tall-bottle depletion")
    save(STUDY / "plan.json", plan)
    print(json.dumps(dict(positives=len(positives), negatives=len(rows)-len(positives), prior_late=late)),
          flush=True)


if __name__ == "__main__":
    main()
