"""Fit the normal-01 mouth path using observed incoming droplets on CUDA."""
from itertools import product
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parent
STUDY=ROOT/"wide-mixed-r12"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import sha
from pour_wide_motion import late_mouth_offset, tail_progress
from fit_wide_mixed_r6 import folder,read,save

SOURCE="01_normal"
TRAIN=("wide-mixed-r7b","wide-mixed-r9b","wide-mixed-r11")
VALIDATE=("wide-mixed-r8","wide-mixed-r10b")


def archived_stream(study, source=SOURCE, tilt_range=(104.,124.), min_particles=501, include_advance=False):
    case=next(r for r in read(ROOT/study/"completion.json")["test_cases"] if r["name"].endswith(source))
    return recorded_stream(case, tilt_range, min_particles, include_advance)


def recorded_stream(case, tilt_range, min_particles, include_advance, *, verified_archive=None):
    path=folder(case)
    if verified_archive is None:
        assert sha(path/"episode.hdf5")==case["recording_sha256"]
    else:
        assert verified_archive["source"] == str(Path(case["path"]) / "episode.hdf5")
        assert verified_archive["original_sha256"] == case["recording_sha256"]
        assert verified_archive["original_bytes"] == case["recording_bytes"]
        from fit_wide_mixed_r6 import HOST
        archive = Path("/workspace") / Path(verified_archive["archive"]).relative_to(HOST)
        assert sha(archive) == verified_archive["archive_sha256"]
    spec=read(path/"parameters.json")
    geometry=spec["declared_cup_geometry"]
    rim,bottom,radius=(geometry[k] for k in ("rim_z_m","bottom_z_m","rim_radius_m"))
    saved=torch.load(spec["checkpoint"],map_location="cpu",weights_only=False)
    setback=saved.get("tail_setback_m",0.)
    drop=saved.get("tail_drop_m",0.)
    window=saved.get("tail_window_deg",(123.,126.))
    seen=set()
    rows=[]
    for frame in read(path/"samples.json"):
        tilt=frame["bottle_tilt_deg"]
        if not tilt_range[0]<=tilt<=tilt_range[1] or frame["stop_reason"]:
            continue
        progress=tail_progress(tilt,window)
        old_x=setback*progress
        if include_advance:
            old_x-=late_mouth_offset(tilt,saved.get("late_mouth_shift_m",0.))
        old_z = drop * progress
        if saved.get("finish_mouth_pose") is not None:
            start, end, final_offset, final_drop = saved["finish_mouth_pose"]
            t = min(1., max(0., (tilt-start)/(end-start)))
            weight = t*t*(3.-2.*t)
            old_x += weight * (-final_offset-old_x)
            old_z += weight * (final_drop-old_z)
        surface=bottom+(rim-bottom)*frame["cup_fill_fraction"]
        for particle in frame["diagnostic_outside_particles"]:
            index=particle["index"]
            x,y,z=particle["xyz_cup_frame_m"]
            vx,vy,vz=particle["velocity_m_s"]
            if (index not in seen and particle["origin"]=="bottle" and abs(y)<.008
                    and -.033<x<.013 and rim+.0015<z<rim+.025 and vx>.12 and vz<-.05):
                seen.add(index)
                rows.append((tilt,x,y,z,vx,vy,vz,rim,surface,radius,old_x,old_z))
    assert len(rows)>=min_particles,(case["name"],len(rows))
    return case, spec, torch.tensor(rows,device="cuda",dtype=torch.float64)


def smooth_ramp(value, start, end):
    fraction=((value-start)/(end-start)).clamp(0.,1.)
    return fraction.square()*(3.-2.*fraction)


def counterfactual(tilt,old_x,old_z,grid):
    start,end,setback,drop=grid[:, :4].T.unsqueeze(-1).unbind(0)
    progress=smooth_ramp(tilt,start,end)
    dx=old_x-setback*progress
    if grid.shape[1]>=5:
        dx=dx+grid[:,4,None]*smooth_ramp(tilt,116.,122.)
    dz = old_z-drop*progress
    if grid.shape[1]==9:
        start, end, offset, down = grid[:, 5:].T.unsqueeze(-1).unbind(0)
        blend = smooth_ramp(tilt, start, end)
        dx += blend * (old_x + offset - dx)
        dz += blend * (old_z - down - dz)
    return dx,dz


def evaluate(data,grid,focus=(108.,116.)):
    """Approximate the incoming stream between observation and impact."""
    tilt,x,y,z,vx,vy,vz,rim,surface,radius,old_x,old_z=data.T[None].unbind(1)
    dx,dz=counterfactual(tilt,old_x,old_z,grid)

    def crossing(height):
        depth=(z+dz-height).clamp(min=0.)
        flight=(vz+torch.sqrt(vz.square()+2*9.81*depth))/9.81
        return torch.hypot(x+dx+vx*flight,y+vy*flight)

    at_rim=crossing(rim)
    at_surface=crossing(surface)
    margin=at_surface/radius
    rim_ratio=at_rim/radius
    weights=1.+((tilt>=focus[0]) & (tilt<=focus[1])).to(data.dtype)
    # Penalize near-wall impacts and predicted rim crossings, with a
    # smaller motion cost to avoid unnecessary changes to the path.
    cost=(margin-.62).clamp(min=0.).square()+.12*margin.square()+2*(rim_ratio-.85).clamp(min=0.).square()
    cost += .10*(dx.square()+dz.square())/radius.square()
    objective=(cost*weights).sum(dim=1)/weights.sum()
    return objective,at_surface,at_rim


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    training=[archived_stream(n) for n in TRAIN]
    validation=[archived_stream(n) for n in VALIDATE]
    target,target_spec,_=training[-1]
    assert target["measured_loss_particles"]==1
    incumbent=torch.load(target_spec["checkpoint"],map_location="cpu",weights_only=False)
    options=list(product((104.,106.,108.),(109.,111.,113.),(.006,.008,.009,.010),(.004,.005,.006)))
    base=(*incumbent["tail_window_deg"],incumbent["tail_setback_m"],incumbent["tail_drop_m"])
    grid=torch.tensor([base,*options],device="cuda",dtype=torch.float64)
    train_loss=torch.stack([evaluate(d,grid)[0] for _,_,d in training]).mean(0)
    best=int(train_loss.argmin())
    assert best>0
    audits=[]
    for set_name,entries in (("train",training),("holdout",validation)):
        for case,_,data in entries:
            objective,radius_at_surface,radius_at_rim=evaluate(data,grid[[0,best]])
            before,after=objective.tolist()
            audits.append(dict(name=case["name"],split=set_name,recording_sha256=case["recording_sha256"],
                               incoming_particles=len(data),loss_before=before,loss_after=after,
                               impact_p95_mm_before=float(torch.quantile(radius_at_surface[0],.95)*1000.),
                               impact_p95_mm_after=float(torch.quantile(radius_at_surface[1],.95)*1000.),
                               predicted_beyond_rim_before=int((radius_at_rim[0]>data[:,9]).sum()),
                               predicted_beyond_rim_after=int((radius_at_rim[1]>data[:,9]).sum())))
    passed=all(r["loss_after"]<r["loss_before"] and
               r["impact_p95_mm_after"]<r["impact_p95_mm_before"] for r in audits)
    start,end,setback,drop=grid[best].tolist()
    report=dict(passed=passed,model="gravity-only incoming stream projection",
                device=torch.cuda.get_device_name(0),candidate_count=len(options),
                mouth_options=dict(tail_window_deg=[start,end],tail_setback_m=setback,tail_drop_m=drop),
                selected_training_objective=float(train_loss[best]),
                incumbent_training_objective=float(train_loss[0]),cases=audits)
    save(STUDY/"stream-fit.json",report)
    print(report,flush=True)
    assert passed


if __name__=="__main__":
    main()
