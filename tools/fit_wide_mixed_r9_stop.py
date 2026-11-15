"""Fit small-cup confirmation at the final allowed top-up pulse."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "wide-mixed-r9"
sys.path.insert(0, str(STUDY/"src"))
from pour_size_autonomy import calibrated_rgb_height, pulse_limit_level, sha
from fit_wide_mixed_r6 import folder, read, save

TARGET = "SA_A_wide_mixed_r8_test_09_swap"
SIZE = [.9,.9]
MIN_HEIGHT = .90
PARENT_THRESHOLD = .925
LOWER_BOUND = .920


def load_cases(study):
    root = ROOT/study
    return read(root/"completion.json")["test_cases"] + read(root/"development-results.json")["cases"]


def windows(case, settings, calibration):
    path = folder(case)
    assert sha(path/"episode.hdf5")==case["recording_sha256"]
    samples={round(s["t"],6):s["cup_fill_fraction"] for s in read(path/"samples.json")}
    group = []
    pause = None
    for frame in read(path/"policy-decisions.json"):
        value=frame.get("sized_cup_perception") or {}
        t=round(frame["t"],6)
        if frame["fill_mode"] == "settle":
            if pause != frame["pause_count"]:
                group=[]
                pause=frame["pause_count"]
                start=t
            if t >= start+.35-1e-6:
                if value.get("valid") and t in samples:
                    v=value.get("guard_filtered",value["filtered"])
                    group.append((calibrated_rgb_height(v["rgb"],calibration),v["color"],samples[t]))
                else:
                    group=[]
            if len(group)>=8 and t>=start+settings["settle_seconds"]-1e-6:
                last=group[-8:]
                level=pulse_limit_level([(r[0],r[1]) for r in last],.035,
                                        settings["settled_stability_std"])
                if level is not None:
                    yield dict(time_s=t,pulse_count=pause,level=level,
                               height_lower=min(r[2] for r in last),
                               at_limit=pause>=settings["max_pauses"])
        if frame["stop_latched"]:
            return


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    named = {}
    for split,studies in (("train",("wide-mixed-r8",)),("validation",("wide-mixed-r7b","wide-mixed-r6"))):
        for study in studies:
            for case in load_cases(study):
                if case["cup_size"] != SIZE or not case["grasp_verified"]:
                    continue
                spec = read(folder(case)/"parameters.json")
                parent = ROOT/"wide-mixed-r8"
                chosen = next(s for s in read(parent/"plan.json")["test"] if s["name"]==TARGET)
                saved = torch.load(chosen["checkpoint"],map_location="cpu",weights_only=False)
                settings = torch.load(spec["checkpoint"],map_location="cpu",weights_only=False)["thresholds"]
                calibration = saved["sized_cup_perception"]["rgb_height_calibration"]["settling"]
                data = list(windows(case,settings,calibration))
                if data:
                    named[case["name"]]=dict(split=split,case=case,data=data)
    assert TARGET in named
    target = named[TARGET]["data"][-1]
    assert target["at_limit"] and target["height_lower"]>=MIN_HEIGHT and target["level"]<PARENT_THRESHOLD, target
    grid = torch.linspace(LOWER_BOUND,PARENT_THRESHOLD,51,device="cuda",dtype=torch.float64)
    underfill = [d["level"] for r in named.values() if r["split"]=="train"
                 for d in r["data"] if d["height_lower"]<MIN_HEIGHT]
    negatives=torch.tensor(underfill,device="cuda",dtype=torch.float64)
    assert len(negatives)
    supporting=grid[(target["level"]>=grid) & (negatives[:,None]<grid).all(dim=0)]
    assert len(supporting), (target, float(negatives.max()))
    # Center the feasible interval to leave headroom on both sides.
    threshold=round(float(supporting[len(supporting)//2]),4)
    reports=[]
    for key,row in named.items():
        data=row["data"]
        accepted=[d for d in data if d["level"]>=threshold]
        reports.append(dict(name=key,split=row["split"],recording_sha256=row["case"]["recording_sha256"],
                            windows=len(data),first_supported=accepted[0] if accepted else None,
                            smallest_supported_height=min((d["height_lower"] for d in accepted),default=None),
                            final=data[-1]))
    assert all(r["smallest_supported_height"] is None or r["smallest_supported_height"]>=MIN_HEIGHT
               for r in reports), reports
    save(STUDY/"near-full-fit.json",dict(passed=True,device=torch.cuda.get_device_name(0),cup_size=SIZE,
        prior_threshold=PARENT_THRESHOLD,pulse_limit_threshold=threshold,grid_points=len(grid),
        train_underfill_windows=len(negatives),max_train_underfill_level=float(negatives.max()),
        fit_range=[LOWER_BOUND,PARENT_THRESHOLD],target_case=TARGET,target_final=target,cases=reports))
    print(threshold,target,len(reports),flush=True)


if __name__=="__main__":
    main()
