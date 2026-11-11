"""Shared development and fresh evaluation for cup presence, tracking and level."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from evaluate_landmark_holdout import error_metrics, level_ok
from fit_dynamic_pair import scores, passes
from fit_dynamic_rim import ROOT, protected_unchanged, read, sha
from fit_followup_presence import decode_detail, prior_scenes, source_path
from perception_followup import CupPresence, LevelFilter, VerifiedCupTracker, erased_coarse, predict_level
from perception_midfill_data import entries
from pour_guard_vision import cup_crop, plane_point, project
from validate_pour_recording import h5py


def save(name, value):
    with (ROOT / name).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def load(name):
    return torch.load(ROOT / name, map_location="cpu", weights_only=False)


def world_error(uv, spec, cal):
    z = cal["table_z"] + spec["declared_cup_geometry"]["rim_z_m"]
    xy = plane_point(uv, z, cal)[:2]
    return float(np.linalg.norm(xy - spec["cup_world"][:2]) * 1000)


def shifted_views(image, world, cal):
    origin = project(world, cal)
    for dx, dy in ((.0035, 0), (-.0035, 0), (0, .0035), (0, -.0035)):
        offset = np.rint(project(world + (dx, dy, 0), cal) - origin).astype(int)
        yield np.asarray(Image.fromarray(image).transform(
            (1024, 1024), Image.Transform.AFFINE, (1, 0, -int(offset[0]), 0, 1, -int(offset[1])),
            Image.Resampling.NEAREST, fillcolor=(180, 180, 180))).copy()


def replay(spec, record, split, locator, models, gates, gpu_presence, cpu_presence):
    geometry = spec["declared_cup_geometry"]
    cal = dict(locator["calibration"], cup_geometry=geometry)
    tracker = VerifiedCupTracker(locator, geometry, presence=gpu_presence)
    cpu = VerifiedCupTracker(locator, geometry, "cpu", cpu_presence) if split != "train" else None
    moving = bool(spec.get("calibration_motion"))
    path = source_path(spec, record)
    samples = read(path.parent / "samples.json") if moving else None
    sample_times = np.array([s["t"] for s in samples]) if moving else None
    physical = record["recording_validation"]["valid"] and record["label"]["accepted"]
    if moving:
        physical &= record["motion"]["passed"] and record["motion"]["cup_inventory_range_ml"] == 0
        physical &= all(v == 0 for v in record["containment"].values())
        if spec["coverage_role"].endswith("high"):
            physical &= gates["high_fill_min"] <= record["label"]["target"] <= gates["high_fill_max"]
    row = dict(name=spec["name"], split=split, moving=moving, physical=physical,
               cup_height_scale=spec["cup_height_scale"], cup_diameter_scale=spec["cup_diameter_scale"],
               recording_sha256=record["sha256"], raw_initial_errors_mm=[], initial=[], frames=[],
               challenges=[], locator_parity=[], tracker_status_parity=True)
    truth = np.asarray(spec["cup_world"], dtype=float)
    truth[2] = cal["table_z"] + geometry["rim_z_m"]
    truth_pixel = project(truth, cal)

    def update(image):
        result = tracker.update(image)
        if cpu is not None:
            other = cpu.update(image)
            row["tracker_status_parity"] &= result["valid"] == other["valid"]
            if result["valid"] and other["valid"]:
                row["locator_parity"].append(float(np.linalg.norm(np.array(result["uv"]) - other["uv"])))
        return result

    with h5py.File(path) as f:
        data = f["data/demo_0"]
        assert np.allclose(f["calibration/cup_detail/intrinsics"][:], cal["K"])
        assert np.allclose(data["obs/cup_detail_world_T_camera"][0], cal["T"])
        initial = [decode_detail(data, "obs", i) for i in range(12)]
        reference = None
        for image in initial:
            result = update(image)
            point = result.get("raw_uv")
            if point is None:
                point, _ = tracker.detect(image)
            row["raw_initial_errors_mm"].append(world_error(point, spec, cal))
            if result["valid"] and reference is None:
                reference = image
            row["initial"].append(dict(valid=result["valid"], reason=result["reason"],
                                       error_mm=world_error(result["uv"], spec, cal) if result["valid"] else None))
        if tracker.anchor is None:
            return row
        xy = plane_point(tracker.anchor, truth[2], cal)[:2]
        row["anchor_error_mm"] = world_error(tracker.anchor, spec, cal)
        if reference is not None and split != "train":
            base = np.asarray(spec["cup_world"], dtype=float)
            base[2] = cal["table_z"] + geometry["bottom_z_m"]
            erased = np.asarray(Image.fromarray(erased_coarse(reference, truth_pixel)).resize((1024,1024))).copy()
            for name, challenge in [("erased", erased)] + [("shifted", im) for im in shifted_views(reference, base, cal)]:
                probe = copy.copy(tracker)
                row["challenges"].append(dict(kind=name, rejected=not probe.update(challenge)["valid"]))
        times = data["next_time"][:]
        if moving:
            indices = np.flatnonzero((times >= .5) & (times <= 2.5))
        else:
            indices = np.flatnonzero(times >= times[-1] - 1 - 1e-7)[::2]
        assert len(indices) == (40 if moving else 11)
        crops = []
        for index in indices:
            image = decode_detail(data, "next_obs", int(index))
            result = update(image)
            target = record["label"]["target"]
            if moving:
                matched = int(np.abs(sample_times - times[index]).argmin())
                assert abs(sample_times[matched] - times[index]) < 1e-6
                target = samples[matched]["cup_fill_fraction"]
            crops.append(cup_crop(image, xy, cal))
            row["frames"].append(dict(index=int(index), t=float(times[index]), target=target,
                                      valid=result["valid"], score=result["score"], reason=result["reason"],
                                      error_mm=world_error(result["uv"], spec, cal) if result["valid"] else None,
                                      absent=tracker.absent_score(erased_coarse(image, truth_pixel))))
    # Score the production single-image path on both devices when checking parity.
    stacked = torch.as_tensor(np.stack(crops))
    for branch, artifact in models.items():
        if cpu is None:
            estimate, valid = predict_level(stacked.cuda(), xy, geometry, artifact)
            values = estimate.cpu().tolist()
            for frame, value, accepted in zip(row["frames"], values, valid.cpu().tolist()):
                frame[branch] = value if accepted else None
        else:
            for image, frame in zip(stacked.split(1), row["frames"]):
                value, good = predict_level(image.cuda(), xy, geometry, artifact)
                cvalue, cgood = predict_level(image, xy, geometry, artifact)
                frame[branch] = float(value[0]) if good[0] else None
                frame["cpu_" + branch] = float(cvalue[0]) if cgood[0] else None
    return row


def train_filters(rows):
    """Fit one convex blend per branch against its own supervised error."""
    filters = {}
    for branch in ("rgb", "color"):
        raw, medians, labels, weights = [], [], [], []
        for row in rows:
            if row["split"] not in ("train", "regression") or not row["moving"]:
                continue
            frame = row["frames"]
            assert len(frame) == 40 and all(f[branch] is not None for f in frame)
            median = LevelFilter(dict(window=7, current_weight=0.))
            raw.extend(f[branch] for f in frame)
            medians.extend(median.update(f[branch]) for f in frame)
            labels.extend(f["target"] for f in frame)
            weights.extend([1 / len(frame)] * len(frame))
        raw, medians, labels, weights = [torch.tensor(v, device="cuda") for v in (raw, medians, labels, weights)]
        # The optimum is the weighted least-squares coefficient with a fixed
        # maximum instantaneous contribution. Each branch sees only its own estimates.
        delta = raw - medians
        weight = ((weights * delta * (labels - medians)).sum() /
                  (weights * delta.square()).sum().clamp_min(1e-12)).clamp(0, .25)
        filters[branch] = dict(window=7, current_weight=float(weight), training_frames=len(raw))
    return filters


def score_case(row, filters, gates, threshold):
    frames = row["frames"]
    expected = 40 if row["moving"] else 11
    stats = dict(name=row["name"], split=row["split"], moving=row["moving"], frames=len(frames),
                 raw_max_mm=max(row["raw_initial_errors_mm"]), anchor_error_mm=row.get("anchor_error_mm"),
                 tracked_frames=sum(f["valid"] for f in frames),
                 tracked_max_mm=max((f["error_mm"] for f in frames if f["valid"]), default=None),
                 negative_max=max((f["absent"] for f in frames), default=None),
                 present_min=min((f["score"] for f in frames), default=None),
                 locator_cpu_gpu_pixels=max(row["locator_parity"], default=0.),
                 challenges=len(row["challenges"]), challenges_rejected=sum(c["rejected"] for c in row["challenges"]))
    limits = gates["localization_max_error_mm"]
    valid_initial = [r["valid"] for r in row["initial"]]
    checks = dict(physical=row["physical"], initial=any(valid_initial) and all(valid_initial[valid_initial.index(True):]) and stats["raw_max_mm"] < limits and stats["anchor_error_mm"] < limits,
                  tracking=len(frames) == expected and stats["tracked_frames"] == expected and stats["tracked_max_mm"] < limits,
                  absence=len(frames) == expected and stats["negative_max"] < threshold and stats["present_min"] - stats["negative_max"] > gates["localization_negative_margin"],
                  shifted_and_erased=stats["challenges_rejected"] == stats["challenges"] == (0 if row["split"]=="train" else 5),
                  locator_parity=row["tracker_status_parity"] and stats["locator_cpu_gpu_pixels"] < .1)
    series = {}
    for branch in ("rgb","color"):
        series[branch] = []
        filtered = LevelFilter(filters[branch])
        cpu_filter = LevelFilter(filters[branch])
        parity = []
        for f in frames:
            value = filtered.update(f[branch] if f[branch] is not None else float("nan"))
            series[branch].append(value)
            if row["split"] != "train":
                cpu = cpu_filter.update(f["cpu_"+branch] if f["cpu_"+branch] is not None else float("nan"))
                parity.append(abs(value-cpu) if np.isfinite([value,cpu]).all() else float("inf"))
        metric = error_metrics(series[branch], [f["target"] for f in frames]) if frames else dict(valid=False)
        stats[branch] = metric
        stats[branch+"_cpu_gpu"] = max(parity, default=0.)
        checks[branch+"_accuracy"] = level_ok(metric,gates)
        checks[branch+"_parity"] = stats[branch+"_cpu_gpu"] < gates["cpu_gpu_level_difference"]
        if not np.isfinite(stats[branch+"_cpu_gpu"]):
            stats[branch+"_cpu_gpu"] = None
    agreement = np.abs(np.asarray(series["rgb"]) - series["color"])
    stats["max_disagreement"] = float(agreement.max()) if len(agreement) and np.isfinite(agreement).all() else None
    checks["agreement"] = stats["max_disagreement"] is not None and stats["max_disagreement"] < gates["crosscheck_max_disagreement"]
    stats.update(checks=checks, passed=all(checks.values()))
    return stats, series


def summarize(rows, filters, gates, threshold):
    cases, predictions = [], {branch: [] for branch in ("rgb","color")}
    for row in rows:
        result, series = score_case(row, filters, gates, threshold)
        cases.append(result)
        for branch in predictions:
            predictions[branch].append(series[branch])
    groups = {}
    for branch in predictions:
        groups[branch] = {}
        for split in sorted({r["split"] for r in rows}):
            indices = [i for i, row in enumerate(rows) if row["split"] == split]
            if any(not cases[i]["checks"][branch+"_accuracy"] for i in indices):
                groups[branch][split] = dict(passed=False)
                continue
            data = dict(y=torch.tensor([f["target"] for i in indices for f in rows[i]["frames"]],device="cuda"),
                        meta=[{k:rows[i][k] for k in ("name","cup_height_scale","cup_diameter_scale","moving")} for i in indices for f in rows[i]["frames"]])
            values = torch.tensor([value for i in indices for value in predictions[branch][i]],device="cuda")
            group_scores = scores(values, data)
            groups[branch][split] = dict(passed=passes(group_scores,gates), scores=group_scores)
    return dict(passed=all(r["passed"] for r in cases) and all(v["passed"] for g in groups.values() for v in g.values()),
                passed_cases=sum(r["passed"] for r in cases), total_cases=len(cases), cases=cases, group_accuracy=groups,
                filters=filters)


def missing_color_rejected(models, locator, geometry):
    for device in ("cpu", "cuda"):
        image = torch.full((1,96,96,3),180,dtype=torch.uint8,device=device)
        value, visible = predict_level(image, np.array([.5,-.2]), geometry, models["color"])
        if bool(visible[0]) or not torch.isnan(value[0]):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    assert not (args.fresh and args.rescore)
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(4)
    protected_unchanged()
    protocol_path = ROOT / "followup-protocol.json"
    protocol = read(protocol_path)
    assert protocol["previous_protocol_sha256"] == sha(ROOT/"protocol.json")
    locator = load("followup-tracker.pt")
    models = {b:load("pair-r17-"+b+".pt") for b in ("rgb","color")}
    historical = read(ROOT/"component-selection.json")
    for name in ("pair-r17-rgb.pt", "pair-r17-color.pt", "tracker-t3.pt"):
        assert historical["frozen_files"][name] == sha(ROOT/name)
    if args.fresh:
        selected = read(ROOT/"followup-selection.json")
        assert selected["all_required_development_gates_passed"]
        assert all(sha(ROOT/n)==v for n,v in selected["frozen_files"].items())
        audit_path = ROOT/"holdout-r1-audit.json"
        audit = read(audit_path)
        specs = {s["name"]:s for s in protocol["holdout"]}
        assert audit["complete"] and audit["protocol_sha256"] == sha(protocol_path)
        assert specs.keys() == {r["name"] for r in audit["records"]}
        records = [(specs[r["name"]],r,"fresh") for r in audit["records"]]
        assert not (ROOT/"followup-fresh-raw.json").exists()
        with (ROOT/"followup-evaluation-started.json").open("x") as f:
            json.dump(dict(selection_sha256=sha(ROOT/"followup-selection.json"), audit_sha256=sha(audit_path)),f)
    else:
        records = entries()+prior_scenes() if not args.rescore else []
        assert not list(Path("/workspace/outputs/fountain").glob("*/*SD_A_motion_holdout_r1_*/parameters.json"))
    prefix = "followup-fresh" if args.fresh else "followup-development"
    sources = ["followup-protocol.json", "followup-tracker.pt", "pair-r17-rgb.pt", "pair-r17-color.pt",
               "perception_followup.py", "evaluate_perception_followup.py", "fit_followup_presence.py"]
    hashes = {n:sha(ROOT/n) for n in sources}
    if args.rescore:
        cached = read(ROOT/(prefix+"-raw.json"))
        assert all(cached["sources"][n] == h for n,h in hashes.items() if n != "evaluate_perception_followup.py")
        rows = cached["rows"]
    else:
        gpu = CupPresence().cuda().eval()
        cpu = CupPresence().eval()
        for model in (gpu, cpu):
            model.load_state_dict(locator["presence_state"])
        rows = []
        for index,(spec,record,split) in enumerate(records,1):
            result = replay(spec,record,split,locator,models,protocol["gates"],gpu,cpu)
            rows.append(result)
            if index%25==0 or args.fresh:
                print(index,spec["name"],sum(f["valid"] for f in result["frames"]),flush=True)
        save(prefix+"-raw.json", dict(rows=rows,sources=hashes))
    filters = selected["filters"] if args.fresh else train_filters(rows)
    report = summarize(rows,filters,protocol["gates"],locator["threshold"])
    geometry = protocol["holdout"][0]["declared_cup_geometry"]
    report["missing_color_rejected"] = missing_color_rejected(models,locator,geometry)
    report["passed"] &= report["missing_color_rejected"]
    report.update(frozen_files=hashes,raw_sha256=sha(ROOT/(prefix+"-raw.json")),autonomy_accepted=False)
    assert all(sha(ROOT/n)==v for n,v in hashes.items())
    protected_unchanged()
    save(prefix+".json",report)
    for case in report["cases"]:
        if not case["passed"]:
            print(case["name"],[k for k,v in case["checks"].items() if not v],flush=True)
    print(json.dumps({k:report[k] for k in ("passed","passed_cases","total_cases","filters")}),flush=True)


if __name__=="__main__":
    main()
