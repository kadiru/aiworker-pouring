"""Fit two camera-only stop corrections on archived pair 09 and verify held-out episodes."""
import math
import numpy as np
import torch

from fit_wide_mixed_r27_neck import PARENT, STUDY, state_for, sample, folder, save
from fit_wide_mixed_r6 import read
from pour_size_autonomy import calibrated_rgb_height, settled_pair, settled_rgb_weight, sha

TRAIN = (20, 22, 24, 26)
VALID = (11, 12, 21, 23, 25)
KNOTS = (.695, .735, .76, .80)


def observations(revision, side, config, thresholds):
    case = sample(f"wide-mixed-r{revision}", "09_"+side)
    path = folder(case)
    decisions = read(path/"policy-decisions.json")
    samples = read(path/"samples.json")
    stop = next(i for i,d in enumerate(decisions) if d["stop_latched"])
    stop_frame = decisions[stop]
    times = np.array([s["t"] for s in samples])
    def height(frame):
        i = int(np.abs(times-frame["t"]).argmin())
        assert abs(times[i]-frame["t"]) < .051
        return samples[i]["cup_fill_fraction"]
    def pair(frame, phase):
        view = frame["sized_cup_perception"]
        assert view["valid"]
        raw = view.get("guard_filtered", view["filtered"])
        rgb = calibrated_rgb_height(raw["rgb"], config["rgb_height_calibration"][phase])
        return raw["rgb"], rgb, raw["color"]
    evidence = dict(name=case["name"], split="train" if revision in TRAIN else "validation",
                    samples_sha256=sha(path/"samples.json"), decisions_sha256=sha(path/"policy-decisions.json"),
                    recording_sha256=case["recording_sha256"])
    if side=="normal":
        assert stop_frame["stop_latched"]=="bottle_empty"
        return dict(evidence, drain=[[*pair(d,"draining"),height(d)] for d in decisions[stop-5:stop+1]])
    assert stop_frame["stop_latched"]=="cup_near_full"
    windows = []
    for pause in range(1,stop_frame["pause_count"]+1):
        frames = [d for d in decisions[:stop+1] if d["fill_mode"]=="settle" and d["pause_count"]==pause]
        if len(frames)<6:
            continue
        last = frames[-6:]
        levels = [pair(d,"settling")[1:] for d in last]
        low,stable,_ = settled_pair(levels,1.,config["agreement_limit"],thresholds["settled_stability_std"])
        windows.append(dict(pause=pause, low=low, stable=stable, height=height(last[-1])))
    assert windows and windows[-1]["stable"]
    return dict(evidence, settled=windows, final_height=case["cup_surface_height_fraction"])


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    checkpoint, state = state_for("09_normal",PARENT)
    cfg, thresholds = state["sized_cup_perception"], state["thresholds"]
    rows = [observations(i,side,cfg,thresholds) for i in TRAIN+VALID for side in ("normal","swap")]
    train = [v for r in rows if r["split"]=="train" for v in r.get("drain",[])]
    raw,rgb,_,truth = torch.tensor(train,dtype=torch.float64,device="cuda").T
    weights = raw.new_tensor([settled_rgb_weight(x,KNOTS) for x in raw.tolist()])
    # Constrain the correction below the existing 3.5 point calibration bound.
    grid = torch.arange(31,device=raw.device,dtype=raw.dtype)[:,None]*.001
    bias = float(grid[((rgb+grid*weights-truth).abs().mean(1)+.1*grid[:,0]).argmin(),0])
    high = [(w["low"]-w["height"]) for r in rows if r["split"]=="train"
            for w in r.get("settled",[]) if w["stable"] and w["low"]>=.89]
    target = math.ceil((.90 + max(high) + .002)*1000)/1000
    assert thresholds["full_settled_target"] < target < .98
    passed = True
    for r in rows:
        if "drain" in r:
            raw,rgb,color,truth = np.array(r["drain"]).T
            corrected = rgb + bias*np.array([settled_rgb_weight(x,KNOTS) for x in raw])
            mae = [float(np.abs(x-truth).mean()) for x in (rgb,corrected)]
            gaps = [float(abs(x[-1]-color[-1])) for x in (rgb,corrected)]
            r.update(rgb_mae=mae,stop_disagreement=gaps)
            passed &= mae[1] < mae[0] and gaps[1] < cfg["agreement_limit"]
        else:
            accepted = [w for w in r["settled"] if w["stable"] and w["low"]>=target]
            more = not accepted
            r.update(requests_another_trim=more)
            passed &= all(w["height"]>=.90 for w in accepted)
            if more:
                passed &= r["settled"][-1]["pause"] < thresholds["max_pauses"]
            if "r26_test" in r["name"]:
                passed &= more and r["final_height"]<.9
    result = dict(passed=bool(passed),checkpoint=checkpoint, parent_sha256=sha(checkpoint),
                  draining_rgb_residual=dict(geometry=cfg["rgb_height_calibration"]["draining"]["geometry"],
                  knots=list(KNOTS),bias=bias), full_settled_target=target,fitted_full_at_pulse_limit=target,
                  agreement_limit=cfg["agreement_limit"],rows=rows)
    print("RGB bias, settled stop target, replay gate",bias,target,bool(passed),flush=True)
    save(STUDY/"stop-fit.json",result)
    assert passed


if __name__=="__main__":
    main()
