"""Fit the short-cup settled RGB correction and freeze the next autonomy batch."""
import copy
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "sized-autonomy"
STUDY = ROOT / "sized-autonomy-r1"
sys.path.insert(0, str(STUDY / "src"))
from pour_size_autonomy import MAX_SETTLED_RGB_BIAS, SETTLED_RGB_KNOTS, settled_pair, settled_rgb_weight, sha
from perception_followup import LevelFilter
from evaluate_landmark_holdout import error_metrics, level_ok

SHORT_STANDARD = (.9, 1.)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def size(scene):
    return scene["cup_height_scale"], scene["cup_diameter_scale"]


def settled_blocks(decisions, samples):
    levels = {round(row["t"], 6): row["cup_fill_fraction"] for row in samples}
    blocks = []
    for mode, group in itertools.groupby(decisions, key=lambda row: row["fill_mode"]):
        rows = list(group)
        if mode != "settle" or rows[0]["stop_latched"]:
            continue
        frames = []
        for row in rows:
            if row["stop_latched"] or row["t"] - rows[0]["t"] < 1.:
                continue
            perception = row["sized_cup_perception"]
            if perception and perception["valid"]:
                filtered = perception["filtered"]
                frames.append((filtered["rgb"], filtered["color"], levels[round(row["t"], 6)]))
        if frames:
            blocks.append(np.asarray(frames))
    return blocks


def replay(before, bias, target, tolerance, stability):
    rgb, color, measured = before.T
    adjusted = rgb - bias * np.array([settled_rgb_weight(x) for x in rgb])
    old_pair = settled_pair(list(zip(rgb, color)), target, tolerance, stability)
    new_pair = settled_pair(list(zip(adjusted, color)), target, tolerance, stability)
    return dict(frames=len(rgb), rgb_mae_before=float(np.abs(rgb-measured).mean()),
                rgb_mae_after=float(np.abs(adjusted-measured).mean()),
                color_mae=float(np.abs(color-measured).mean()),
                last_six_gap_before=float(np.abs(rgb[-6:]-color[-6:]).max()),
                last_six_gap_after=float(np.abs(adjusted[-6:]-color[-6:]).max()),
                last_six_before=old_pair, last_six_after=new_pair)


def archived_checks(path, filters, gates, bias):
    checked = changed = 0
    max_mae = max_disagreement = 0.
    for row in read(path)["rows"]:
        if size(row) != SHORT_STANDARD:
            continue
        frames = row["frames"]
        series = {}
        for branch, config in filters.items():
            level_filter = LevelFilter(config)
            series[branch] = np.array([level_filter.update(f[branch]) for f in frames])
        rgb = series["rgb"]
        adjusted = rgb - bias * np.array([settled_rgb_weight(x) for x in rgb])
        changed += int(np.count_nonzero(adjusted != rgb))
        checked += 1
        accuracy = error_metrics(adjusted, [f["target"] for f in frames])
        disagreement = float(np.abs(adjusted - series["color"]).max())
        assert level_ok(accuracy, gates), (row["name"], accuracy)
        assert disagreement < gates["crosscheck_max_disagreement"], (row["name"], disagreement)
        max_mae = max(max_mae, accuracy["mae"])
        max_disagreement = max(max_disagreement, disagreement)
    assert checked
    return dict(cases=checked, changed_frames=changed, max_rgb_mae=max_mae,
                max_disagreement=max_disagreement, sha256=sha(path))


def main():
    base_plan_path = BASELINE / "plan.json"
    base_results_path = BASELINE / "test-results.json"
    plan = read(base_plan_path)
    results = read(base_results_path)
    assert results["plan_sha256"] == sha(base_plan_path)
    template = next(scene for scene in plan["test"] if size(scene) == SHORT_STANDARD)
    failure = next(row for row in results["cases"] if row["name"] == template["name"])
    assert failure["stop_reason"] == "level_unconfirmed" and failure["measured_loss_particles"] == 0
    recording = Path("/workspace") / Path(failure["path"]).relative_to("/home/kadir/aiworker-pouring-4070")
    decisions, samples = recording / "policy-decisions.json", recording / "samples.json"
    fit, check = settled_blocks(read(decisions), read(samples))
    rgb, _, measured = fit.T
    weight = np.array([settled_rgb_weight(x) for x in rgb])
    assert len(fit) >= 4 and len(check) >= 6 and weight @ weight > 0.
    bias = float(np.clip(weight @ (rgb-measured) / (weight @ weight), 0., MAX_SETTLED_RGB_BIAS))
    assert bias > 0.

    saved = torch.load(template["checkpoint"], map_location="cpu", weights_only=False)
    config = saved["sized_cup_perception"]
    gates = read(ROOT/"followup-protocol.json")["gates"]
    agreement = config["agreement_limit"]
    assert agreement == gates["crosscheck_max_disagreement"]
    target = saved["thresholds"]["full_settled_target"]
    stability = saved["thresholds"]["settled_stability_std"]
    fit_stats = replay(fit, bias, target, agreement, stability)
    check_stats = replay(check, bias, target, agreement, stability)
    assert not check_stats["last_six_before"][1] and check_stats["last_six_after"][1]
    assert not check_stats["last_six_after"][2] and check_stats["rgb_mae_after"] < check_stats["rgb_mae_before"]
    regressions = {n: archived_checks(ROOT/n, config["filters"], gates, bias)
                   for n in ("followup-development-raw.json", "followup-fresh-raw.json")}
    config["settled_rgb_calibration"] = dict(geometry=template["declared_cup_geometry"], bias=bias)
    unchanged = {}
    checkpoints = {}
    for scene in plan["test"]:
        source = Path(scene["checkpoint"])
        dest = STUDY / source.name
        assert not dest.exists() and sha(source) == plan["checkpoints"][str(source)]
        if size(scene) == SHORT_STANDARD:
            torch.save(saved, dest)
        else:
            shutil.copy2(source, dest)
            assert sha(source) == sha(dest)
            unchanged[source.name] = sha(dest)
        checkpoints[str(dest)] = sha(dest)

    receipt = STUDY / "calibration.json"
    save(receipt, dict(cup_size=SHORT_STANDARD, bias=bias, knots=SETTLED_RGB_KNOTS,
                      agreement_limit=agreement, source_case=failure["name"],
                      source_recording_sha256=failure["recording_sha256"],
                      decisions_sha256=sha(decisions), samples_sha256=sha(samples),
                      baseline_plan_sha256=sha(base_plan_path), baseline_results_sha256=sha(base_results_path),
                      fit_method="Bounded least squares against measured liquid height in the first settled block",
                      fit=fit_stats, same_episode_replay=check_stats, archived_regressions=regressions,
                      unchanged_checkpoints=unchanged))
    development = copy.deepcopy(template)
    development.update(name="SA_A_development_r1_01", study_split="development",
                       seed=2611120101, checkpoint=str(STUDY/Path(template["checkpoint"]).name))
    for index, scene in enumerate(plan["test"]):
        scene.update(name=f"SA_A_test_r1_{index:02d}", seed=2611120200+index,
                     checkpoint=str(STUDY/Path(scene["checkpoint"]).name))
        direction = -1 if index % 2 else 1
        scene["cup_world"][0] += .001 * direction
        scene["cup_world"][1] += .00075 * direction
        scene["source_xy"][1] += .001 * direction
    plan.update(development=[development], checkpoints=checkpoints,
                development_requires_strict_zero=True,
                settled_rgb_calibration=dict(receipt_sha256=sha(receipt), cup_size=SHORT_STANDARD,
                                             agreement_limit=agreement))
    assert len(plan["test"]) == len({size(s) for s in plan["test"]}) == 9
    save(STUDY/"plan.json", plan)
    print(json.dumps(dict(bias=bias, replay=check_stats, archived_regressions=regressions)), flush=True)


if __name__ == "__main__":
    main()
