"""Read the shared pair 09 checkpoint and trace the original stop decisions."""
from fit_wide_mixed_r27_neck import PARENT, STUDY, state_for, sample, folder, save
from fit_wide_mixed_r6 import read


def main():
    path, state=state_for("09_normal",PARENT)
    cfg=state["sized_cup_perception"]
    keys=("rgb_height_calibration","draining_color_calibration","settled_rgb_calibration","midfill_rgb_calibration","agreement_limit")
    rows=[]
    for side in ("normal","swap"):
        case=sample(PARENT.name,"09_"+side)
        frames=read(folder(case)/"policy-decisions.json")
        seen={}
        for f in frames:
            v=f.get("sized_cup_perception") or {}
            if v.get("valid") and (f["fill_mode"]=="settle" or f["stop_latched"]):
                seen[f["pause_count"]]=dict(t=f["t"],stop=f["stop_latched"],view=v)
            if f["stop_latched"]:break
        rows.append(dict(name=case["name"],recording_sha256=case["recording_sha256"],windows=seen))
    save(STUDY/"stop-diagnosis.json",dict(checkpoint=path,calibrations={k:cfg.get(k) for k in keys},
         thresholds={k:v for k,v in state["thresholds"].items() if any(x in k for x in ("settle","pulse","trim","full","pause","brake"))},cases=rows))


if __name__=="__main__":
    main()
