"""Fit two phase-specific trajectories and a corroborated bottle-neck fallback."""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/"wide-mixed-r18"
STUDY=ROOT/"wide-mixed-r19"
sys.path.insert(0,str(STUDY/"src"))
from pour_guard_vision import bottle_base_from_neck, locate_initial_source, make_bank, project as project_pixel
from pour_size_autonomy import sha
from pour_temporal_localization import consensus
from fit_wide_mixed_r6 import decode, erased, folder, h5py, read, save
from fit_wide_mixed_r7 import shifted
from fit_wide_mixed_r12_stream import archived_stream, recorded_stream, counterfactual
from fit_wide_mixed_r13_stream import project
from fit_wide_mixed_r14_stream import HOLDOUT, TRAIN, sample, scores, trace
from fit_wide_mixed_r17 import verified_record
from refine_wide_mixed_r13_stream import rebound_at_rim


def fit_motion(key):
    case=sample(PARENT.name,key)
    observed=trace(case)
    early=key=="06_swap"
    span,focus=((105.,127.),(108.,115.)) if early else ((114.,132.),(122.,131.))
    train=[archived_stream(s,key,span,80,True) for s in TRAIN]
    train.append(recorded_stream(case,span,80,True,verified_archive=verified_record(case,PARENT,STUDY)))
    validation=[archived_stream(s,key,span,80,True) for s in HOLDOUT]
    spec=train[-1][1]
    ck=torch.load(spec["checkpoint"],map_location="cpu",weights_only=False)
    start,end=ck["tail_window_deg"]
    back,drop,advance=(ck[k] for k in ("tail_setback_m","tail_drop_m","late_mouth_shift_m"))
    assert not ck.get("finish_mouth_pose")
    base=(start,end,back,drop,advance,126.,130.,advance-back,drop)
    if early:
        options=[(on,off,s,d,advance,118.,123.,advance-back,drop)
                 for (on,off),s,d in product(((104.,110.),(106.,112.),(108.,114.)),
                                             (.004,.006,.008,.010),(.002,.004,.006,.008))]
    else:
        options=[(*base[:5],on,off,x/1000,d/1000)
                 for (on,off),x,d in product(((122.,126.),(124.,128.),(126.,129.)),range(-10,5,2),(2,4,6))]
    grid=torch.tensor([base,*options],device="cuda",dtype=torch.float64)
    obs=[]
    for track in observed["particles"].values():
        rebound=next(p for p in track if p["velocity"][2]>.01 and p["xyz"][0]>0.)
        obs.append([rebound["tilt"],*rebound["xyz"],*rebound["velocity"]])
    points=grid.new_tensor(obs).T
    zero=torch.zeros_like(points[0])
    bx,bz=counterfactual(points[0],zero,zero,grid[:1])
    geometry=spec["declared_cup_geometry"]
    event=torch.cat((points,-bx,-bz)),geometry["rim_z_m"],geometry["rim_radius_m"]
    ratios=rebound_at_rim(event,grid)
    cost=torch.stack([project(d,grid,focus)[0] for _,_,d in train]).mean(0)
    worst=ratios.amax(1)
    radius=geometry["rim_radius_m"]
    acceptable=worst<=max(1.-.0015/radius,float(worst.min())+.0005/radius)
    best=int(cost.masked_fill(~acceptable,torch.inf).argmin())
    audits=[]
    for split,batch in (("train",train),("holdout",validation)):
        for c,_,d in batch:
            obj,impact,rim,r=project(d,grid[[0,best]],focus)
            audits.append(dict(scores(c,d,split,obj),outside=(rim>r).sum(1).tolist(),
                               impact_p95_ratio=torch.quantile(impact/r,.95,dim=1).tolist()))
    on,off,s,d,adv,*finish=grid[best].tolist()
    options=dict(finish_mouth_pose=finish)
    if early:
        options.update(tail_window_deg=[on,off],tail_setback_m=s,tail_drop_m=d)
    passed=best>0 and bool(((ratios[0]-ratios[best])*radius>=.0015).all()) and all(
        a["outside"][1]<=a["outside"][0] and a["impact_p95_ratio"][1]<.85 for a in audits)
    return dict(passed=passed,checkpoint=spec["checkpoint"],mouth_options=options,
                candidates=len(grid),rim_ratios=ratios[[0,best]].tolist(),cases=audits),observed


def neck_rows(study=STUDY, filename="neck-startup.h5"):
    with h5py.File(study/"inputs"/filename) as archive:
        for name,g in archive.items():
            revision,key=name.removeprefix("SA_A_wide_mixed_").split("_test_")
            c=sample("wide-mixed-"+revision,key)
            assert c["recording_sha256"]==g.attrs["recording_sha256"]
            spec=read(folder(c)/"parameters.json")
            base=np.array(spec["cup_world"])+[-spec["source_xy"][1],spec["source_xy"][0],0.]
            lip=spec["declared_bottle_geometry"]["lip_height_m"]
            cal=dict(K=g["K"][:],T=g["T"][:],table_z=base[2])
            uv=project_pixel(base+[0.,0.,lip],cal)
            frames=[decode(g[str(i)][:],spec["image_encoding"]).copy() for i in range(41)]
            yield dict(name=name,recording_sha256=c["recording_sha256"],uv=uv,cal=cal,lip=lip,frames=frames)


def replay(frames,state):
    threshold=state["thresholds"]["source_localization_score"]
    settings=state["localization_consensus"]["source"]
    history=[]
    for i,image in enumerate(frames):
        pixel,score,method=locate_initial_source(image,state)
        history.append((pixel.tolist(),score))
        hit=consensus(history,threshold,settings)
        if hit:
            return hit["uv"],method,i+1
    return None,None,len(frames)


def historical_views():
    for study in ("wide-mixed-r11","wide-mixed-r12"):
        for key in ("01_normal","01_swap"):
            case=sample(study,key)
            path=folder(case)
            spec=read(path/"parameters.json")
            source=path/"episode.hdf5"
            assert sha(source)==case["recording_sha256"]
            base=np.array(spec["cup_world"])+[-spec["source_xy"][1],spec["source_xy"][0],0.]
            lip=spec["declared_bottle_geometry"]["lip_height_m"]
            with h5py.File(source) as h:
                obs=h["data/demo_0/obs"]
                cal=dict(K=h["calibration/task/intrinsics"][:],T=obs["task_world_T_camera"][0],table_z=base[2])
                frames=[decode(obs["task_image"][i], spec["image_encoding"]).copy() for i in range(41)]
            yield dict(name=case["name"],frames=frames,cal=cal,lip=lip,
                       uv=project_pixel(base+[0.,0.,lip],cal),recording_sha256=case["recording_sha256"])


@torch.inference_mode()
def fit_neck():
    rows=[*neck_rows(),*historical_views()]
    target=next(r for r in rows if r["name"].endswith("r18_test_01_normal"))
    spec=read(folder(sample(PARENT.name,"01_normal"))/"parameters.json")
    state=torch.load(spec["checkpoint"],map_location="cpu",weights_only=False)
    assert not state.get("source_heatmap_fallback") and replay(target["frames"],state)[0] is None
    ids=(0,3,6,9)
    templates=[r for r in rows if r["name"].endswith(("r17_test_01_normal","r18_test_01_normal"))]
    bank=make_bank([r["frames"][i] for r in templates for i in ids],[r["uv"] for r in templates for i in ids],
                   "source",aggregation="max",gradient_clip=.04,source_mask_box=(10,7,38,39))
    artifact=dict(proposal=state["source_heatmap"]["proposal"],proposal_floor=.53,
                  refinement=bank,refinement_floor=.80,max_refine_px=4.)
    candidate=dict(state,source_heatmap_fallback=artifact)
    results=[]
    for row in rows:
        for dx,dy in ((0,0),(-4,3),(3,-4)):
            imgs=row["frames"] if dx==dy==0 else [shifted(im,dx,dy) for im in row["frames"]]
            truth=row["uv"]+[dx,dy]
            hit,method,n=replay(imgs,candidate)
            err=None if hit is None else float(1000*np.linalg.norm(
                bottle_base_from_neck(hit,row["cal"],row["lip"])[:2]-bottle_base_from_neck(truth,row["cal"],row["lip"])[:2]))
            results.append(dict(name=row["name"],variant=[dx,dy],method=method,frames=n,error_mm=err))
        missing=[erased(im,row["uv"],i%2) for i,im in enumerate(row["frames"])]
        old=replay(missing,state);new=replay(missing,candidate)
        results.append(dict(name=row["name"],variant="erased",parent_accept=old[0] is not None,candidate_accept=new[0] is not None,
                            parent_method=old[1],candidate_method=new[1]))
    positives=[r for r in results if r["variant"]!="erased"]
    negatives=[r for r in results if r["variant"]=="erased"]
    passed=all(r["error_mm"] is not None and r["error_mm"]<=2.5 for r in positives) and all(not r["candidate_accept"] for r in negatives)
    torch.save(artifact,STUDY/"neck-fallback.pt")
    save(STUDY/"neck-replay.json",dict(passed=passed,cases=results,fit_frames=list(ids),template_trials=[r["name"] for r in templates],
         trials={r["name"]:r["recording_sha256"] for r in rows},startup_sha256=sha(STUDY/"inputs/neck-startup.h5"),
         checkpoint=spec["checkpoint"],artifact_sha256=sha(STUDY/"neck-fallback.pt")))
    print("neck replay",passed,[r for r in results if r.get("candidate_accept") or (r["variant"]!="erased" and (r["error_mm"] is None or r["error_mm"]>2.5))],flush=True)
    return passed


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    fits,traces={},{}
    for key in ("06_swap","12_swap"):
        fits[key],traces[key]=fit_motion(key)
        print(key,fits[key]["passed"],fits[key]["mouth_options"],fits[key]["rim_ratios"],flush=True)
    save(STUDY/"spill-trace.json",dict(cases=traces))
    passed=all(f["passed"] for f in fits.values())
    save(STUDY/"stream-fit.json",dict(passed=passed,device=torch.cuda.get_device_name(),cases=fits))
    neck_ok=fit_neck()
    assert passed and neck_ok


if __name__=="__main__":
    main()
