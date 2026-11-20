"""Fit a late-tilt pose from the observed descending rim exits of both particles."""
from itertools import product

import torch

from fit_wide_mixed_r12_stream import archived_stream, counterfactual, recorded_stream
from fit_wide_mixed_r13_stream import project
from fit_wide_mixed_r14_stream import HOLDOUT, ROOT, TRAIN, sample, scores, trace
from fit_wide_mixed_r17 import verified_record
from fit_wide_mixed_r6 import read, save
from pour_size_autonomy import sha

PARENT = ROOT / "wide-mixed-r17"
STUDY = ROOT / "wide-mixed-r18"
TARGET = "02_swap"
SPAN, FOCUS = (115.,132.), (122.,130.)


def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    target = sample(PARENT.name, TARGET)
    assert target["measured_loss_particles"] == 2
    observed = trace(target)
    archive = verified_record(target, PARENT, STUDY)
    train = [archived_stream(s,TARGET,SPAN,80,True) for s in TRAIN]
    train.append(recorded_stream(target,SPAN,80,True,verified_archive=archive))
    validation = [archived_stream(s,TARGET,SPAN,80,True) for s in HOLDOUT]
    spec = train[-1][1]
    saved = torch.load(spec["checkpoint"],map_location="cpu",weights_only=False)
    normal = sample(PARENT.name,"02_normal")
    assert normal["strict_pass"] and normal["checkpoint_sha256"] == target["checkpoint_sha256"]
    start,end = saved.get("tail_window_deg",(123.,126.))
    back,drop,advance = (saved.get(k,0.) for k in ("tail_setback_m","tail_drop_m","late_mouth_shift_m"))
    assert end<=126. and not saved.get("finish_mouth_pose")
    x0 = advance-back
    base = (start,end,back,drop,advance,126.,130.,x0,drop)
    # Only the terminal mouth pose may change. Limit additional lowering to
    # 2 mm and leave the entire motion before 122 degrees untouched.
    options = [(*base[:5],on,off,x/1000,z/1000)
               for (on,off),x,z in product(((122.,125.),(123.,126.),(124.,127.)),range(-10,11),
                                           range(0, int(round(min(.008,drop+.002)*1000))+1))]
    grid = torch.tensor([base,*options],device="cuda",dtype=torch.float64)
    rim, radius = (spec["declared_cup_geometry"][k] for k in ("rim_z_m","rim_radius_m"))
    exits = []
    for track in observed["particles"].values():
        first, after = track[:2]
        assert first["xyz"][0]<0<after["xyz"][0] and first["velocity"][2]<0
        assert after["xyz"][2]>rim and after["velocity"][2]<0 and after["velocity"][1]>.15
        exits.append([after["tilt"],*after["xyz"],*after["velocity"]])
    tilt,x,y,z,vx,vy,vz = grid.new_tensor(exits).T
    zeros = torch.zeros_like(tilt)
    bx,bz = counterfactual(tilt,zeros,zeros,grid[:1])
    dx,dz = counterfactual(tilt,-bx,-bz,grid)
    fall = (z+dz-rim).clamp_min(0.)
    t = (vz+(vz.square()+2*9.81*fall).sqrt())/9.81
    ratio = torch.hypot(x+dx+vx*t,y+vy*t)/radius
    cost = torch.stack([project(d,grid,FOCUS)[0] for _,_,d in train]).mean(0)
    movement = torch.hypot(grid[:,7]-x0,grid[:,8]-drop)
    # Choose the smallest end-pose displacement to place both replayed
    # exits one particle radius inside the rim. Break ties by incoming cost.
    clearance = .0015
    safe = ratio.amax(1) <= 1.-clearance/radius
    if safe.any():
        least = movement[safe].min()
        eligible = safe & (movement<=least+1e-9)
    else:
        # At the actuator bounds the proxy may only support partial recovery.
        # Keep candidates within half a millimetre of the best clearance.
        risk = ratio.amax(1)
        supported = risk <= risk.min()+.0005/radius
        least = movement[supported].min()
        eligible = supported & (movement<=least+1e-9)
    best = int(cost.masked_fill(~eligible,torch.inf).argmin())
    audits = []
    for split,batch in (("train",train),("holdout",validation)):
        for case,_,data in batch:
            obj,impact,cross,r = project(data,grid[[0,best]],FOCUS)
            audits.append(dict(scores(case,data,split,obj),outside=(cross>r).sum(1).tolist(),
                               impact_p95_ratio=torch.quantile(impact/r,.95,dim=1).tolist()))
    passed = (best>0 and bool(((ratio[0]-ratio[best])*radius>=.0015).all())
              and all(a["outside"][1]<=a["outside"][0] and a["impact_p95_ratio"][1]<.85 for a in audits))
    finish = grid[best,5:].tolist()
    fit = dict(passed=passed,checkpoint=spec["checkpoint"],mouth_options=dict(finish_mouth_pose=finish),
               candidates=len(grid),rebound_rim_ratio=ratio[[0,best]].tolist(),cases=audits,
               parent_terminal_pose_m=[x0,drop],end_displacement_mm=float(1000*movement[best]),
               desired_clearance_m=clearance,both_replayed_inside_margin=bool(safe[best]),
               parent_completion_sha256=sha(PARENT/"completion.json"))
    save(STUDY/"spill-trace.json",dict(cases={TARGET:observed}))
    save(STUDY/"stream-fit.json",dict(passed=passed,device=torch.cuda.get_device_name(),cases={TARGET:fit}))
    print("passed",passed,"parent pose",x0,drop,"candidate",finish,"rim ratios",fit["rebound_rim_ratio"],flush=True)
    assert passed


if __name__ == "__main__":
    main()
