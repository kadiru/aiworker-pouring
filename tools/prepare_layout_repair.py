"""Fit and replay targeted visual repairs before sealing new autonomous layouts."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
HOST = Path("/home/kadir/aiworker-pouring-4070")
BASELINE = ROOT / "tall-empty-r1"
STUDY = ROOT / "layout-repair-r2"
sys.path[:0] = [str(STUDY / "src"), "/workspace"]

from pour_guard_vision import (bottle_base_from_neck, locate, locate_initial_source,
                               make_bank, plane_point, project)
from pour_size_autonomy import (MAX_SETTLED_RGB_BIAS, MIDFILL_RGB_KNOTS, SizedCupPerception,
                                settled_rgb_weight, sha)
from pour_temporal_localization import consensus
from validate_pour_recording import decode, h5py

TALL_NARROW_CUP = (1.1, .9)
SOURCES = (("tall-empty-r1", "test"), ("mixed-vessel-autonomy", "development"),
           ("bottle-autonomy", "test"), ("bottle-empty-autonomy", "development"))
FRESH_LAYOUTS = (
    ((.483, -.150), (-.118, -.008)),
    ((.499, -.190), (-.107, .003)),
    ((.488, -.184), (-.117, .007)),
    ((.502, -.163), (-.123, -.011)),
    ((.479, -.180), (-.110, -.005)),
    ((.495, -.144), (-.118, .004)),
)


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def folder_for(case):
    return Path("/workspace") / Path(case["path"]).relative_to(HOST)


def recorded_frames(case, key, count):
    folder = folder_for(case)
    scene = read(folder / "parameters.json")
    with h5py.File(folder / "episode.hdf5") as file:
        data = file[f"data/demo_0/obs/{key}"]
        frames = [decode(data[i], scene["image_encoding"]).copy() for i in range(count)]
    return scene, frames


def erase_bottle(image, uv):
    x, y = np.rint(uv).astype(int)
    x0, x1 = max(1, x - 34), min(383, x + 36)
    y0, y1 = max(0, y - 13), min(384, y + 124)
    result = image.copy()
    left, right = image[y0:y1, x0-1:x0], image[y0:y1, x1:x1+1]
    weight = np.linspace(0, 1, x1-x0)[None, :, None]
    result[y0:y1, x0:x1] = left * (1-weight) + right * weight
    return result


def train_neck():
    records = []
    for study, part in SOURCES:
        for case in read(ROOT / study / f"{part}-results.json")["cases"]:
            lip = case["bottle_geometry"]["lip_height_m"]
            if lip < .14:
                continue
            spec, frames = recorded_frames(case, "task_image", 12)
            saved = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
            cal = saved["calibration"]
            source_x, source_y = spec["source_xy"]
            base = np.asarray(spec["cup_world"]) + [-source_y, source_x, 0.]
            uv = project(base + [0., 0., lip], cal)
            records.append((case, saved, frames, lip, base, uv))
    training = [record for i, record in enumerate(records)
                if i % 3 != 2 or record[0]["name"].endswith("02_empty")]
    training = [record for record in training if not record[0]["name"].endswith("05_full")]
    fit_names = {record[0]["name"] for record in training}
    anchors = np.array([record[-1] for record in training])
    bank = make_bank([record[2][0] for record in training], anchors, "source",
                     aggregation="max", gradient_clip=.04, source_mask_box=(10, 7, 38, 39))
    bank["bounds"] = [*np.floor(anchors.min(0)-14).astype(int).tolist(),
                      *np.ceil(anchors.max(0)+14).astype(int).tolist()]
    fallback = dict(bank=bank, threshold=.50, presence_floor=.33)
    receipt = []
    for case, saved, frames, lip, base, uv in records:
        name = case["name"]
        model = dict(saved, source_initial_fallback=fallback)
        limit = saved["thresholds"]["source_localization_score"]
        config = saved["localization_consensus"]["source"]
        detections = [locate_initial_source(frame, model) for frame in frames]
        accepted = consensus([(point.tolist(), score) for point, score, _ in detections], limit, config)
        missing = [erase_bottle(frame, uv) for frame in frames]
        absent_max = max(locate(frame, bank)[1] for frame in missing)
        absent_detections = [locate_initial_source(frame, model) for frame in missing]
        absent_confirmed = consensus([(point.tolist(), score) for point, score, _ in absent_detections],
                                     limit, config)
        assert accepted is not None and absent_max < fallback["presence_floor"] and absent_confirmed is None, (
            name, accepted, absent_max, absent_confirmed)
        localized = bottle_base_from_neck(accepted["uv"], saved["calibration"], lip)
        error = float(np.linalg.norm(localized[:2]-base[:2])*1000)
        fallback_frames = sum(method == "neck_fallback" for _, _, method in detections)
        assert error <= 2.5, (name, error)
        if name.endswith("05_full"):
            assert name not in fit_names and fallback_frames >= config["votes"]
        receipt.append(dict(name=name, split="template_fit" if name in fit_names else "archive_replay",
                            recording_sha256=case["recording_sha256"], frames=len(frames),
                            consensus_error_mm=error, fallback_frames=fallback_frames,
                            max_erased_neck_score=absent_max, erased_confirmed=absent_confirmed is not None))
    return fallback, dict(method="Signed clipped RGB-gradient correlation against labeled neck templates",
                          fallback_threshold=fallback["threshold"], presence_floor=fallback["presence_floor"],
                          bounds=bank["bounds"],
                          source_mask_box=bank["source_mask_box"], gradient_clip=bank["gradient_clip"],
                          fit_episodes=len(training), replay_episodes=len(receipt)-len(training), cases=receipt)


def check_cup_recovery(case):
    scene, frames = recorded_frames(case, "cup_detail_image", 40)
    saved = torch.load(scene["checkpoint"], map_location="cpu", weights_only=False)
    config = saved["sized_cup_perception"]
    g = scene["declared_cup_geometry"]
    baseline = SizedCupPerception(config, g)
    recovered = SizedCupPerception(dict(config, retry_initial_cup=True), g)
    missing = SizedCupPerception(dict(config, retry_initial_cup=True), g)
    world = np.asarray(scene["cup_world"])
    cal = config["tracker"]["calibration"]
    rim = project(world + [0., 0., g["rim_z_m"]], cal)
    x, y = np.rint(rim).astype(int)
    absent_frames = []
    for frame in frames:
        erased = frame.copy()
        erased[max(0, y-65):y+150, max(0, x-135):x+135] = 180
        absent_frames.append(erased)
    original = [baseline.track(frame) for frame in frames]
    results = [recovered.track(frame) for frame in frames]
    assert not any(row["valid"] for row in original)
    assert not any(missing.track(frame)["valid"] for frame in absent_frames)
    first = next(i for i, row in enumerate(results) if row["valid"])
    point = plane_point(results[first]["uv"], world[2]+g["rim_z_m"], cal)
    error = float(np.linalg.norm(point[:2]-world[:2])*1000)
    assert first < 40 and error <= 2.5
    return dict(name=case["name"], recording_sha256=case["recording_sha256"], frames=len(frames),
                first_recovered_frame=first, consensus_error_mm=error, initial_failure_frame=next(
                    i for i, row in enumerate(original) if row["reason"] == "cup_absent_or_uncertain"),
                valid_frames_after_recovery=sum(r["valid"] for r in results),
                accepted_erased_frames=0, erased_challenge="Synthetic cup-area mask")


def fit_level(case):
    folder = folder_for(case)
    decisions, samples = folder / "policy-decisions.json", folder / "samples.json"
    labels = {round(r["t"], 6): r["cup_fill_fraction"] for r in read(samples)}
    start = case["first_zero_source_s"]
    fit, replay = [], []
    for row in read(decisions):
        if not row.get("sized_cup_perception") or row["t"] < start:
            continue
        if row["stop_latched"] and row["mode"] != "Returning upright":
            break
        values = row["sized_cup_perception"]["filtered"]
        if not all(value is not None for value in values.values()):
            continue
        group = fit if row["t"] < start+1.55 else replay
        group.append((values["rgb"], values["color"], labels[round(row["t"], 6)]))
        if row["stop_latched"]:
            break
    rgb, _, truth = np.asarray(fit).T
    w = np.array([settled_rgb_weight(v, MIDFILL_RGB_KNOTS) for v in rgb])
    assert len(fit) >= 20 and len(replay) >= 30 and w @ w > 0.
    bias = float(np.clip(w @ (rgb-truth) / (w @ w), 0., MAX_SETTLED_RGB_BIAS))

    def metrics(rows):
        rgb, color, target = np.asarray(rows).T
        corrected = rgb - bias*np.array([settled_rgb_weight(v, MIDFILL_RGB_KNOTS) for v in rgb])
        return dict(frames=len(rows), mae_before=float(np.abs(rgb-target).mean()),
                    mae_after=float(np.abs(corrected-target).mean()),
                    max_gap_before=float(np.abs(rgb-color).max()),
                    max_gap_after=float(np.abs(corrected-color).max()),
                    last_gap_before=float(abs(rgb[-1]-color[-1])),
                    last_gap_after=float(abs(corrected[-1]-color[-1])))

    check = metrics(replay)
    assert check["mae_after"] < check["mae_before"]
    assert check["last_gap_before"] >= .035 and check["max_gap_after"] < .035, check
    return bias, dict(name=case["name"], bias=bias, knots=MIDFILL_RGB_KNOTS,
                      method="Bounded least squares against measured rim-height fraction",
                      recording_sha256=case["recording_sha256"], decisions_sha256=sha(decisions),
                      samples_sha256=sha(samples), fit=metrics(fit), later_same_episode_replay=check)


def main():
    torch.set_num_threads(3)
    plan_path, previous = BASELINE / "plan.json", BASELINE / "test-results.json"
    parent, old = read(plan_path), read(previous)
    assert old["plan_sha256"] == sha(plan_path) and old["completed"] == old["planned"] == 6
    cases = old["cases"]
    fallback, source = train_neck()
    cup = check_cup_recovery(cases[1])
    bias, level = fit_level(cases[2])
    short = cases[4]
    assert "source_not_depleted_at_stop" in short["failure_reasons"]
    receipt = STUDY / "empty-replay.json"
    save(receipt, dict(source_localization=source, cup_localization=cup, midfill_level=level,
                       short_empty=dict(name=short["name"], remaining_at_stop_ml=short["source_at_stop_ml"],
                                        remaining_after_return_ml=short["source_remaining_ml"],
                                        verification="Wet development uses the existing cap and flow dwell",
                                        recording_sha256=short["recording_sha256"])))
    shutil.copy2(BASELINE / "calibration.json", STUDY / "calibration.json")
    plan = {k: v for k, v in parent.items()
            if k not in ("development", "test", "checkpoints", "development_empty_lag_limit_s")}
    plan.update(layout_repair=True, development=[], test=[], checkpoints={},
                controller_scope="Four repaired development fixtures, then six fresh mixed-layout pours",
                empty_confirmation_replay_sha256=sha(receipt))
    plan["prerequisite_reports"] = dict(parent["prerequisite_reports"], **{str(previous): sha(previous)})
    for i, spec in enumerate(parent["test"]):
        path = spec["checkpoint"]
        assert sha(path) == parent["checkpoints"][path]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        config = saved["sized_cup_perception"]
        config["retry_initial_cup"] = True
        size = spec["cup_height_scale"], spec["cup_diameter_scale"]
        if size == TALL_NARROW_CUP:
            config["midfill_rgb_calibration"] = dict(geometry=spec["declared_cup_geometry"], bias=bias)
        if saved["bottle_geometry"]["lip_height_m"] >= .14:
            saved["source_initial_fallback"] = fallback
        else:
            saved["short_empty_requires_drain"] = True
        saved["model_version"] = "visual_guard_layout_repair_r2"
        weight = STUDY / f"mixed-{i:02}.pt"
        assert not weight.exists()
        torch.save(saved, weight)
        plan["checkpoints"][str(weight)] = sha(weight)
        goal = "empty" if spec["expected_stop"] == "bottle_empty" else "full"
        for part in ("development", "test"):
            if part == "development" and i not in (1, 2, 4, 5):
                continue
            scene = copy.deepcopy(spec)
            seed_base = 2611123100 if part == "development" else 2611123200
            scene.update(name=f"SA_A_layout_r2_{part}_{i:02}_{goal}", study_split=part,
                         seed=seed_base+i, checkpoint=str(weight))
            if part == "test":
                cup_xy, source_xy = FRESH_LAYOUTS[i]
                scene["cup_world"][:2] = cup_xy
                scene["source_xy"] = list(source_xy)
            plan[part].append(scene)
    save(STUDY / "plan.json", plan)
    print(json.dumps(dict(source=source, cup=cup, level=level,
                          development=len(plan["development"]), fresh=len(plan["test"]))), flush=True)


if __name__ == "__main__":
    main()
