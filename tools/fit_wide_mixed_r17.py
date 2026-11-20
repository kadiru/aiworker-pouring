"""Fit post-lift confirmation and phase-local mouth corrections with r16 localization frozen."""
from itertools import product
from math import dist
from pathlib import Path

import torch

from fit_wide_mixed_r14_stream import HOLDOUT, ROOT, TRAIN, sample, scores, trace
from fit_wide_mixed_r12_stream import archived_stream, counterfactual, recorded_stream
from fit_wide_mixed_r13_stream import project
from fit_wide_mixed_r6 import HOST, folder, read, save
from pour_size_autonomy import sha
from refine_wide_mixed_r13_stream import rebound_at_rim

PARENT = ROOT / "wide-mixed-r16"
STUDY = ROOT / "wide-mixed-r17"
LAYOUTS = ("05_normal", "05_swap", "10_swap", "14_swap")


def grasp_windows(case):
    path = folder(case)
    frames = read(path / "policy-decisions.json")
    truth = {round(s["t"], 3): s for s in read(path / "samples.json")}
    attempts = {}
    for f in frames:
        v = f.get("grasp_check")
        if not v or not f["mode"].endswith("verify"):
            continue
        s = truth[round(f["t"], 3)]
        attempts.setdefault(f["grasp_attempt"], []).append((
            dist(v["neck_uv"], v["expected_uv"]), v["score"] > .23 and v["moved"],
            v["finger_obstruction"] and v["arm_lift_m"] > .035,
            bool(s["both_fingers"] and s["bottle_base_height_m"] > .03)))
    return attempts


def votes(frames, radius, window):
    history = []
    for error, visual, physical, held in frames:
        history = (history + [visual and error < radius])[-window:] if physical else []
        if sum(history) >= 6:
            return held
    return None


def fit_grasp(target):
    spec = read(folder(target)/"parameters.json")
    saved = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    settings = saved["thresholds"]
    old, window = settings["grasp_follow_radius_px"], settings["grasp_visual_window"]
    target_attempts = grasp_windows(target)
    candidates = [old+i*.25 for i in range(7)]
    eligible = [r for r in candidates if all(votes(fs, r, window) is True for fs in target_attempts.values())]
    assert eligible and min(eligible) > old
    radius = min(eligible)
    audits, controls = [], 0
    for study in (*HOLDOUT, *TRAIN, "wide-mixed-r14", "wide-mixed-r15", PARENT.name):
        for key in ("05_normal", "05_swap"):
            case = sample(study, key)
            for attempt, frames in grasp_windows(case).items():
                before, after = (votes(frames, r, window) for r in (old, radius))
                audits.append(dict(case=case["name"], attempt=attempt, parent=before, candidate=after,
                                   decisions_sha256=sha(folder(case)/"policy-decisions.json")))
                # Ablate each necessary piece of evidence independently on real trajectories.
                for missing in ("physical", "visual", "near"):
                    negative = [(radius+1 if missing=="near" else e,
                                 v and missing!="visual", p and missing!="physical", False)
                                for e,v,p,_ in frames]
                    assert votes(negative, radius, window) is None
                    controls += 1
    passed = all(a["candidate"] is not False and (a["parent"] is not True or a["candidate"] is True)
                 for a in audits) and len(target_attempts)==2
    return dict(passed=passed, checkpoint=spec["checkpoint"], mouth_options={},
                threshold_options={"grasp_follow_radius_px":radius}, previous_radius_px=old,
                six_vote_window=window, ablation_controls=controls, cases=audits)


def verified_record(target, parent=PARENT, study=STUDY):
    for manifest in sorted(parent.glob("test-*-recording-archives.json")):
        data = read(manifest)
        for entry in data["records"]:
            if Path(entry["source"]).parent != Path(target["path"]):
                continue
            assert data["complete"] and data["report_sha256"] == sha(str(manifest).replace("recording-archives","results"))
            p = study/"inputs"/Path(entry["archive"]).name
            return dict(entry, archive=str(HOST / p.relative_to("/workspace")))
    raise ValueError(target["name"])


def candidates(saved, early):
    start,end = saved.get("tail_window_deg",(123.,126.))
    back,drop,advance = (saved.get(k,0.) for k in ("tail_setback_m","tail_drop_m","late_mouth_shift_m"))
    # A finish pose equal to the settled incumbent offset/drop is an identity
    # beyond the initial ramps when the checkpoint has no terminal blend.
    assert end <= 126.
    finish = saved.get("finish_mouth_pose") or (126.,130.,advance-back,drop)
    base = (start,end,back,drop,advance,*finish)
    if early:
        assert not saved.get("finish_mouth_pose")
        options = [(on,off,x,z,advance,116.,123.,advance-back,drop)
                   for (on,off),x,z in product(((104.,109.),(104.,111.),(106.,112.)),
                                               (.006,.008,.010),(.003,.005,.007,.008))]
    else:
        options = [(*base[:5],on,off,x,z)
                   for (on,off),x,z in product(((121.,125.),(123.,127.),(124.,128.)),
                                               (-.010,-.008,-.006,-.004,-.002,0.,.002),
                                               (.002,.004,.006,.008))]
    return torch.tensor([base,*options],device="cuda",dtype=torch.float64)


def fit_stream(target, key, traces):
    early = key=="05_normal"
    focus = (106.,114.) if early else (120.,131.)
    span = (104.,127.) if early else (114.,132.)
    train = [archived_stream(study,key,span,80,True) for study in TRAIN]
    train.append(recorded_stream(target,span,80,True,verified_archive=verified_record(target)))
    validate = [archived_stream(study,key,span,80,True) for study in HOLDOUT]
    spec = train[-1][1]
    state = torch.load(spec["checkpoint"],map_location="cpu",weights_only=False)
    grid = candidates(state,early)
    rows = []
    for track in traces["particles"].values():
        bounce = next(p for p in track if p["velocity"][2] > .05)
        rows.append([bounce["tilt"],*bounce["xyz"],*bounce["velocity"]])
    obs = grid.new_tensor(rows).T
    zeros = torch.zeros_like(obs[0])
    base_x,base_z = counterfactual(obs[0],zeros,zeros,grid[:1])
    geometry = spec["declared_cup_geometry"]
    event = torch.cat((obs,-base_x,-base_z)),geometry["rim_z_m"],geometry["rim_radius_m"]
    ratios = rebound_at_rim(event,grid)
    cost = torch.stack([project(d,grid,focus)[0] for _,_,d in train]).mean(0)
    objective = cost + .6*ratios.square().mean(1)
    # The first geometry screen found hand/bench contact above 128 degrees
    # with 8 mm of terminal lowering. Keep at least the incumbent end height.
    feasible = torch.ones_like(objective,dtype=torch.bool)
    if not early:
        feasible &= grid[:,8] <= min(.002, float(grid[0,8]))
    best = int(objective.masked_fill(~feasible,torch.inf).argmin())
    audits=[]
    for split,batch in (("train",train),("holdout",validate)):
        for case,_,d in batch:
            obj,impact,rim,radius = project(d,grid[[0,best]],focus)
            audits.append(dict(scores(case,d,split,obj),outside=(rim>radius).sum(1).tolist(),
                               impact_p95_ratio=torch.quantile(impact/radius,.95,dim=1).tolist()))
    start,end,back,drop,advance,*finish = grid[best].tolist()
    passed = (best>0 and bool((ratios[best]<ratios[0]).all()) and
              all(a["outside"][1]<=a["outside"][0] and a["impact_p95_ratio"][1]<.85 for a in audits))
    return dict(passed=passed,checkpoint=spec["checkpoint"],candidates=len(grid),
                rebound_rim_ratio=ratios[[0,best]].tolist(),cases=audits,
                mouth_options=dict(tail_window_deg=[start,end],tail_setback_m=back,tail_drop_m=drop,
                                   late_mouth_shift_m=advance,finish_mouth_pose=finish))


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    traces, fits = {}, {}
    for key in LAYOUTS:
        target = sample(PARENT.name,key)
        if key=="05_swap":
            fits[key] = fit_grasp(target)
        else:
            traces[key] = trace(target)
            fits[key] = fit_stream(target,key,traces[key])
        print(key,fits[key]["passed"],fits[key].get("mouth_options"),fits[key].get("threshold_options"),flush=True)
    save(STUDY/"spill-trace.json",dict(cases=traces))
    passed = all(f["passed"] for f in fits.values())
    save(STUDY/"stream-fit.json",dict(passed=passed,device=torch.cuda.get_device_name(),cases=fits))
    assert passed


if __name__=="__main__":
    main()
