"""Fit corroborating neck silhouettes and local mid-pour trajectories to archived RGB and particle tracks."""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "wide-mixed-r19"
STUDY = ROOT / "wide-mixed-r20"
sys.path.insert(0, str(STUDY/"src"))
from pour_guard_vision import bottle_base_from_neck, make_bank
from pour_size_autonomy import sha
from fit_wide_mixed_r19 import neck_rows, replay
from fit_wide_mixed_r7 import erased, shifted
from fit_wide_mixed_r6 import folder, read, save
from fit_wide_mixed_r12_stream import archived_stream, counterfactual, recorded_stream
from fit_wide_mixed_r13_stream import project
from fit_wide_mixed_r14_stream import HOLDOUT, TRAIN, sample, scores, trace
from fit_wide_mixed_r17 import verified_record

NECK = ("12_swap", "14_swap")
MOTION = ("11_normal", "13_normal")


def state_for(key):
    case = sample(PARENT.name, key)
    path = read(folder(case)/"parameters.json")["checkpoint"]
    assert sha(path) == case["checkpoint_sha256"]
    return path, torch.load(path, map_location="cpu", weights_only=False)


def add_refinement(state, bank):
    primary = state["source_heatmap"]
    alternatives = dict(alternatives=[primary["refinement"], bank])
    return dict(state, source_heatmap=dict(primary, refinement=alternatives))


@torch.inference_mode()
def fit_neck(key, rows):
    path, state = state_for(key)
    target = next(r for r in rows if r["name"].endswith("r19_test_"+key))
    assert replay(target["frames"], state)[0] is None
    indices = (0, 3, 6, 9)
    templates = [r for r in rows if any(r["name"].endswith(rev+"_test_"+key) for rev in ("r18","r19"))]
    bank = make_bank([r["frames"][i] for r in templates for i in indices],
                     [r["uv"] for r in templates for i in indices],
                     "source", aggregation="max", gradient_clip=.04, source_mask_box=(10,7,38,39))
    candidate = add_refinement(state, bank)
    records = []
    for row in rows:
        for dx, dy in ((0,0),(-4,3),(3,-4)):
            images = row["frames"] if (dx,dy)==(0,0) else [shifted(im,dx,dy) for im in row["frames"]]
            truth = bottle_base_from_neck(row["uv"]+[dx,dy], row["cal"], row["lip"])
            point, method, frames = replay(images,candidate)
            error = None if point is None else float(1000*np.linalg.norm(
                bottle_base_from_neck(point,row["cal"],row["lip"])[:2]-truth[:2]))
            records.append(dict(name=row["name"],shift=[dx,dy],frames=frames,method=method,error_mm=error))
        absent = [erased(im,row["uv"],i%2) for i,im in enumerate(row["frames"])]
        parent_hit = replay(absent,state)[0] is not None
        candidate_hit = replay(absent,candidate)[0] is not None
        records.append(dict(name=row["name"],shift="erased",parent_accept=parent_hit,candidate_accept=candidate_hit))
    failures = [r for r in records if (r["candidate_accept"] if r["shift"]=="erased" else
                                         r["error_mm"] is None or r["error_mm"]>2.5)]
    artifact = STUDY/("neck-"+key+".pt")
    torch.save(bank,artifact)
    print(key,"neck replay failures",failures,flush=True)
    return dict(passed=not failures,checkpoint=path,bank_sha256=sha(artifact),
                template_trials=[r["name"] for r in templates],template_indices=indices,
                trials={r["name"]:r["recording_sha256"] for r in rows},cases=records)


def motion_fit(key):
    case = sample(PARENT.name,key)
    observed = trace(case)
    train = [archived_stream(s,key,(108.,130.),80,True) for s in TRAIN]
    train.append(recorded_stream(case,(108.,130.),80,True,verified_archive=verified_record(case,PARENT,STUDY)))
    validation = [archived_stream(s,key,(108.,130.),80,True) for s in HOLDOUT]
    spec = train[-1][1]
    path,state = state_for(key)
    start,end = state.get("tail_window_deg",(123.,126.))
    back,drop,advance = (state.get(k,0.) for k in ("tail_setback_m","tail_drop_m","late_mouth_shift_m"))
    assert not state.get("finish_mouth_pose") and end<=126.
    base = (start,end,back,drop,advance,126.,130.,advance-back,drop)
    # Return to the incumbent terminal offset and height after the observed escape.
    options = [(on,off,b,z,a,124.,128.,advance-back,drop)
               for (on,off),b,z,a in product(((108.,116.),(112.,117.),(113.,119.),(114.,120.)),
                                             (.004,.006,.008,.010),(.002,.004,.006,.008),(0.,.002,.004,.006,.008))]
    grid = torch.tensor([base,*options],device="cuda",dtype=torch.float64)
    events = []
    for track in observed["particles"].values():
        if key=="11_normal":
            frame = next(f for f in track if f["velocity"][2]>.2 and f["xyz"][0]>0)
        else:
            # This pair skimmed the side wall. Use its last above-rim sample
            # before the descent, which includes transverse velocity.
            frame = next(f for f in reversed(track) if f["xyz"][2]>=spec["declared_cup_geometry"]["rim_z_m"])
        events.append([frame["tilt"],*frame["xyz"],*frame["velocity"]])
    tilt,x,y,z,vx,vy,vz = grid.new_tensor(events).T
    zeros = torch.zeros_like(tilt)
    old_x,old_z = counterfactual(tilt,zeros,zeros,grid[:1])
    dx,dz = counterfactual(tilt,-old_x,-old_z,grid)
    geom = spec["declared_cup_geometry"]
    flight = (vz+(vz.square()+2*9.81*(z+dz-geom["rim_z_m"]).clamp_min(0.)).sqrt())/9.81
    rad = torch.hypot(x+dx+vx*flight,y+vy*flight)
    focus = (117.,125.) if key=="11_normal" else (115.,123.)
    loss = torch.stack([project(data,grid,focus)[0] for _,_,data in train]).mean(0)
    worst = rad.amax(1)
    eligible = worst<=max(geom["rim_radius_m"]-.0015,float(worst.min())+.00035)
    best = int(loss.masked_fill(~eligible,torch.inf).argmin())
    audit = []
    for split,batch in (("train",train),("holdout",validation)):
        for c,_,data in batch:
            cost,impact,rim,radius = project(data,grid[[0,best]],focus)
            audit.append(dict(scores(c,data,split,cost),outside=(rim>radius).sum(1).tolist(),
                              impact_p95_ratio=torch.quantile(impact/radius,.95,dim=1).tolist()))
    passed = (best>0 and bool((rad[0]-rad[best]>=.001).all()) and
              all(a["outside"][1]<=a["outside"][0] and a["impact_p95_ratio"][1]<.85 for a in audit))
    on,off,b,z,a,*finish = grid[best].tolist()
    result = dict(passed=passed,checkpoint=path,candidates=len(grid),rim_ratios=(rad[[0,best]]/geom["rim_radius_m"]).tolist(),
           mouth_options=dict(tail_window_deg=[on,off],tail_setback_m=b,tail_drop_m=z,late_mouth_shift_m=a,finish_mouth_pose=finish),cases=audit)
    print(key,passed,result["rim_ratios"],result["mouth_options"],flush=True)
    return result,observed


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    fits,traces={},{}
    for key in MOTION:
        fits[key],traces[key]=motion_fit(key)
    save(STUDY/"spill-trace.json",dict(cases=traces))
    save(STUDY/"stream-fit.json",dict(passed=all(f["passed"] for f in fits.values()),device=torch.cuda.get_device_name(),cases=fits))
    rows=list(neck_rows(STUDY))
    neck={}
    for key in NECK:
        pairs=("12",) if key=="12_swap" else ("00","14")
        neck[key]=fit_neck(key,[r for r in rows if r["name"].split("_test_")[1][:2] in pairs])
    save(STUDY/"neck-replay.json",dict(passed=all(f["passed"] for f in neck.values()),cases=neck,
                                      startup_sha256=sha(STUDY/"inputs/neck-startup.h5")))
    assert all(f["passed"] for f in [*fits.values(),*neck.values()])


if __name__=="__main__":
    main()
