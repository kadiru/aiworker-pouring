"""Fit the two observed spill phases and final-pause confirmation from archived trials."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "wide-mixed-r15/src"))
from pour_size_autonomy import pulse_limit_level, sha
from fit_wide_mixed_r6 import folder, read, save
from fit_wide_mixed_r14_stream import ROOT, brake, coarse, sample, trace

PARENT = ROOT / "wide-mixed-r14"
STUDY = ROOT / "wide-mixed-r15"


def replay(case, settings, tolerance):
    path = folder(case)
    height = {round(f["t"], 6): f["cup_fill_fraction"] for f in read(path/"samples.json")}
    group, rows = [], []
    last_pause = None
    for frame in read(path/"policy-decisions.json"):
        if frame["fill_mode"] == "settle":
            t = round(frame["t"], 6)
            pause = frame["pause_count"]
            if pause != last_pause:
                last_pause, start, group = pause, t, []
            view = frame.get("sized_cup_perception") or {}
            if t >= start + .35 - 1e-6:
                if t in height and view.get("valid"):
                    group.append((frame["cup_predicted_fraction"], frame["surface_geometry_fraction"], height[t]))
                else:
                    group.clear()
            if len(group) >= 8 and t >= start + settings["settle_seconds"] - 1e-6:
                last = group[-8:]
                value = pulse_limit_level([x[:2] for x in last], tolerance, settings["settled_stability_std"])
                if value is not None:
                    rows.append(dict(t=t, pause=pause, at_limit=pause >= settings["max_pauses"],
                                     level=value, minimum_height=min(x[2] for x in last)))
        if frame["stop_latched"]:
            break
    return rows


def confirm(target):
    path = folder(target)
    spec = read(path/"parameters.json")
    saved = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    settings = saved["thresholds"]
    assert "fitted_full_at_pulse_limit" not in settings and target["stop_reason"] == "fill_unconfirmed"
    tolerance = saved["sized_cup_perception"].get("agreement_limit", .035)
    training, validation = [], []
    for studies, dataset in ((("wide-mixed-r9b", "wide-mixed-r12", PARENT.name), training),
                             (("wide-mixed-r8", "wide-mixed-r10b", "wide-mixed-r11", "wide-mixed-r13b"), validation)):
        for study in studies:
            for case in read(ROOT/study/"completion.json")["test_cases"]:
                if case["cup_size"] != target["cup_size"] or not case["grasp_verified"]:
                    continue
                windows = replay(case, settings, tolerance)
                if windows:
                    dataset.append((case, windows))
    failed = next(rows for case, rows in training if case["name"] == target["name"])
    final = failed[-1]
    assert final["at_limit"] and final["minimum_height"] >= .90 and final["level"] < settings["full_settled_target"]
    grid = torch.linspace(.915, settings["full_settled_target"], 101, device="cuda", dtype=torch.float64)
    negatives = [row["level"] for _, rows in training for row in rows if row["minimum_height"] < .90]
    assert negatives
    underfill = torch.tensor(negatives, device="cuda", dtype=torch.float64)
    eligible = grid[(grid <= final["level"]) & (underfill[:, None] < grid).all(0)]
    assert len(eligible), (final, max(negatives))
    threshold = round(float(eligible[len(eligible)//2]), 4)
    audits = []
    for split, batch in (("train", training), ("holdout", validation)):
        for case, rows in batch:
            accepted = [r for r in rows if r["level"] >= threshold]
            audits.append(dict(name=case["name"], split=split, windows=len(rows), supported=len(accepted),
                               underfilled=sum(r["minimum_height"] < .90 for r in rows),
                               false_confirmations=sum(r["minimum_height"] < .90 for r in accepted),
                               minimum_supported_height=min((r["minimum_height"] for r in accepted), default=None),
                               sidecar_sha256=sha(folder(case)/"policy-decisions.json")))
    passed = bool(validation and any(x["supported"] for x in audits if x["split"] == "holdout")
                  and any(x["underfilled"] for x in audits if x["split"] == "holdout")
                  and all(x["false_confirmations"] == 0 for x in audits))
    report = dict(passed=passed, final=final, old_threshold=settings["full_settled_target"],
                  fitted_final_threshold=threshold, agreement_tolerance=tolerance, cases=audits)
    save(STUDY/"near-full-fit.json", report)
    return dict(passed=passed, checkpoint=spec["checkpoint"], mouth_options={},
                threshold_options=dict(fitted_full_at_pulse_limit=threshold))


def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    targets = {key: sample(PARENT.name, key) for key in ("00_swap","10_normal","04_normal")}
    traces = {k: trace(c) for k, c in targets.items() if c["measured_loss_particles"]}
    assert set(traces) == {"00_swap","10_normal"}
    ordered = read(PARENT/"test-01-recording-archives.json")
    assert ordered["complete"] and ordered["report_sha256"] == sha(PARENT/"test-01-results.json")
    case = targets["00_swap"]
    archive = next(a for a in ordered["records"] if a["source"] == str(Path(case["path"])/"episode.hdf5"))
    fits = {"00_swap": coarse(case, traces["00_swap"], archive, layout="00_swap",
                              span=(116.,130.), focus=(120.,126.), rebound_min_vz=.04,
                              finish_mouth_pose=(124., 128., .002, .004)),
            "10_normal": brake(targets["10_normal"], layout="10_normal", parent=PARENT.name),
            "04_normal": confirm(targets["04_normal"])}
    passed = all(f["passed"] for f in fits.values())
    save(STUDY/"spill-trace.json", dict(cases=traces))
    save(STUDY/"stream-fit.json", dict(passed=passed, device=torch.cuda.get_device_name(), cases=fits))
    for key, f in fits.items():
        print(key, f["passed"], f["mouth_options"], f.get("threshold_options"), flush=True)
    assert passed


if __name__ == "__main__":
    main()
