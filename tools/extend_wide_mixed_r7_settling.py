"""Extend the small cup's settling correction using newly measured heights."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT/"wide-mixed-r7b"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import (MAX_SETTLED_RGB_BIAS, calibrated_rgb_height,
                                settled_pair, settled_rgb_weight, sha, validate_rgb_calibration)
from prepare_wide_mixed_r7 import observations
from fit_wide_mixed_r6 import folder, read, save

KNOTS = [.739,.787,.911,.95]
HOST = Path("/home/kadir/aiworker-pouring-4070")


def high_fill_fit():
    trial = list(Path("/workspace/outputs/size-autonomy/wide-r7-dev").glob("*/*09_swap*/episode.hdf5"))
    assert len(trial) == 1
    current = dict(path=str(HOST/trial[0].parent.relative_to("/workspace")))
    previous = read(ROOT/"wide-mixed-r6"/"completion.json")["test_cases"]
    archive = [r for r in previous if r["name"].endswith(("00_normal","09_swap","14_swap"))]
    fit, checks = [], []
    for case in (current,*archive):
        spec, history = observations(case)
        rows = [r for r in history if r[0]=="settling" and r[2]>=.78]
        checks.append((spec,rows,sha(folder(case)/"episode.hdf5")))
        for i,(_,_,rgb,color,actual,_) in enumerate(rows[:-6]):
            if i%12 < 6:
                # Newly observed high fills extend the former fit support.
                weight = (4. if case is current and rgb>.89 else 1.)/len(rows)
                fit.append((rgb,actual,weight))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb,height,weight = torch.tensor(fit,dtype=torch.float64,device=device).T
    basis = torch.as_tensor([settled_rgb_weight(v,KNOTS) for v in rgb.cpu().numpy()],device=device)
    bias = float(torch.clamp((weight*basis*(rgb-height)).sum()/(weight*basis.square()).sum(),
                             0.,MAX_SETTLED_RGB_BIAS))
    geometry = checks[0][0]["declared_cup_geometry"]
    calibration = dict(geometry=geometry,knots=KNOTS,bias=bias)
    validate_rgb_calibration(calibration,geometry,settling=True)
    audited = []
    for spec,history,digest in checks:
        readings = np.asarray([(r[2],r[3],r[4]) for r in history[-6:]])
        rgb,color,height = readings.T
        corrected = np.asarray([calibrated_rgb_height(v,calibration) for v in rgb])
        level,stable,full = settled_pair(list(zip(corrected,color)),.925,.035,.012)
        audited.append(dict(name=spec["name"],recording_sha256=digest,
                            final_six_gap_before=float(max(abs(rgb-color))),
                            final_six_gap_after=float(max(abs(corrected-color))),
                            rgb_mae_before=float(abs(rgb-height).mean()),
                            rgb_mae_after=float(abs(corrected-height).mean()),
                            final_level=level,stable=stable,full=full))
    passed = all(r["stable"] for r in audited) and audited[1]["full"]
    passed = passed and all(not r["full"] and r["rgb_mae_after"]<r["rgb_mae_before"]
                            for r in (audited[0],*audited[2:]))
    return dict(passed=passed,device=device,calibration=calibration,train_frames=len(fit),cases=audited)


def main():
    torch.set_num_threads(1)
    plan_path = STUDY/"plan.json"
    plan = read(plan_path)
    receipt = STUDY/"high-fill-replay.json"
    report = high_fill_fit()
    save(receipt,report)
    assert report["passed"], report
    new_paths = {}
    for part in ("development","test"):
        for spec in plan[part]:
            if (spec["cup_height_scale"],spec["cup_diameter_scale"]) != (.9,.9):
                continue
            source = spec["checkpoint"]
            if source not in new_paths:
                assert sha(source) == plan["checkpoints"].pop(source)
                state = torch.load(source,map_location="cpu",weights_only=False)
                state["sized_cup_perception"]["rgb_height_calibration"]["settling"] = report["calibration"]
                dest = STUDY/(Path(source).stem+"-higher-settle.pt")
                assert not dest.exists()
                torch.save(state,dest)
                new_paths[source] = str(dest)
                plan["checkpoints"][str(dest)] = sha(dest)
            spec["checkpoint"] = new_paths[source]
    plan["prerequisite_reports"][str(receipt)] = sha(receipt)
    staging = plan_path.with_suffix(".revised.json")
    save(staging,plan)
    staging.replace(plan_path)


if __name__ == "__main__":
    main()
