"""Supervise bottle-neck heatmaps and exact depletion from native archived recordings."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "wide-mixed-r6"
HOST = Path("/home/kadir/aiworker-pouring-4070")
sys.path[:0] = [str(STUDY/"src"), "/workspace"]
from pour_cup_locator import CupLocator, centers
from pour_guard_vision import bottle_base_from_neck, locate, locate_initial_source, make_bank, project
from pour_size_autonomy import sha
from pour_source_watchdog import TrainedEmptyGate
from pour_wide_empty import EmptyWrist, augment, wrist_crop
from validate_pour_recording import decode, h5py

PARENT = ROOT/"wide-mixed-r5"
SLOW = ("SA_A_wide_mixed_r1_test_01_normal", "SA_A_wide_mixed_r1_test_09_normal")
ARCHIVES = (("layout-repair-r3","test-results.json"), ("wide-mixed-r2","development-results.json"),
            ("wide-mixed-r4","development-results.json"), ("wide-mixed-r5","development-results.json"),
            ("bottle-empty-autonomy","development-results.json"), ("tall-empty-r1","test-results.json"),
            ("tall-empty-r1","development-results.json"), ("mixed-vessel-autonomy","development-results.json"),
            ("bottle-autonomy","test-results.json"))


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as file:
        json.dump(value,file,indent=2,allow_nan=False)
        file.write("\n")


def folder(case):
    return Path("/workspace")/Path(case["path"]).relative_to(HOST)


def examples():
    recent = read(PARENT/"completion.json")["test_cases"]
    entries = [(c,"wide_benchmark") for c in recent]
    for study,report in ARCHIVES:
        entries.extend((c,study) for c in read(ROOT/study/report)["cases"])
    return entries


def erased(image, uv, variant):
    x,y=np.rint(uv).astype(int)
    lo,hi=max(1,x-38),min(383,x+38)
    top,bottom=max(0,y-15),min(384,y+122)
    patch=image.copy()
    a,b=image[top:bottom,lo-1:lo],image[top:bottom,hi:hi+1]
    if variant == 0:
        mix=np.linspace(0.,1.,hi-lo)[None,:,None]
        patch[top:bottom,lo:hi]=(1.-mix)*a+mix*b
    else:
        patch[top:bottom,lo:hi]=np.median(np.concatenate((a,b),axis=1),axis=1)[:,None]
    return patch


def neck_examples():
    rows=[]
    for case, origin in examples():
        p=folder(case)
        spec=read(p/"parameters.json")
        if origin != "wide_benchmark" and origin not in ("layout-repair-r3","wide-mixed-r2","wide-mixed-r4",
                                                         "wide-mixed-r5","bottle-empty-autonomy"):
            continue
        record=p/"episode.hdf5"
        assert sha(record)==case["recording_sha256"]
        base=np.array(spec["cup_world"])+[-spec["source_xy"][1],spec["source_xy"][0],0.]
        lip=spec["declared_bottle_geometry"]["lip_height_m"]
        with h5py.File(record) as file:
            data=file["data/demo_0"]
            cal=dict(K=file["calibration/task/intrinsics"][:],T=data["obs/task_world_T_camera"][0],
                     table_z=spec["cup_world"][2])
            uv=project(base+[0.,0.,lip],cal)
            frames=np.stack([decode(data["obs/task_image"][i],spec["image_encoding"]) for i in (0,3,6,9,12,15)])
        assert np.all((uv>24)&(uv<355)),(spec["name"],uv)
        rows.append(dict(name=spec["name"],origin=origin,pair=spec["pair_index"],base=base,lip=lip,
                         uv=uv,frames=frames,cal=cal,sha256=case["recording_sha256"],
                         checkpoint=spec["checkpoint"],failed=case["stop_reason"]=="localization_unconfirmed"))
    return rows


def locator_data(rows):
    images,labels,present=[],[],[]
    for row in rows:
        for index,image in enumerate(row["frames"]):
            images.append(image)
            labels.append(row["uv"])
            present.append(True)
            if index%2==0:
                images.append(erased(image,row["uv"],index%4!=0))
                labels.append(row["uv"])
                present.append(False)
    return (torch.tensor(np.stack(images),device="cuda"),
            torch.tensor(np.stack(labels),device="cuda",dtype=torch.float32),
            torch.tensor(present,device="cuda"))


@torch.inference_mode()
def score_locator(model,rows,bank=None):
    result=[]
    model.eval()
    for row in rows:
        images=row["frames"]
        negatives=[erased(image,row["uv"],variant) for image in images[::2] for variant in (0,1)]
        native=np.concatenate((images,np.stack(negatives)))
        batch=torch.tensor(native,device="cuda")
        pixels,scores=centers(model(batch))
        points=pixels[:len(images)].cpu().numpy()
        if bank is not None:
            result_scores=scores.cpu().numpy().copy()
            points=[]
            for i,(image,point,score) in enumerate(zip(native,pixels.cpu().numpy(),result_scores.copy())):
                if score>=.15:
                    point,refine=locate(image,bank,expected_uv=point)
                    result_scores[i]=min(score,refine)
                points.append(point)
            points=points[:len(images)]
            scores=result_scores
        estimates=[bottle_base_from_neck(uv,row["cal"],row["lip"])[:2] for uv in points]
        error=np.linalg.norm(np.asarray(estimates)-row["base"][:2],axis=1)*1000
        result.append(dict(name=row["name"],failed_before=row["failed"],sha256=row["sha256"],
                           max_error_mm=float(error.max()),median_error_mm=float(np.median(error)),
                           min_score=float(scores[:len(images)].min()),max_erased_score=float(scores[len(images):].max())))
    return result


def fit_locator(rows):
    training=[r for r in rows if r["origin"] not in ("layout-repair-r3","bottle-empty-autonomy")]
    training_names={t["name"] for t in training}
    validation=[r for r in rows if r["name"] not in training_names]
    assert len(validation)>=4 and sum(r["failed"] for r in training)==7
    images,uv,present=locator_data(training)
    bank=make_bank([r["frames"][i] for r in training for i in (0,3)],
                   [r["uv"] for r in training for _ in range(2)],"source",
                   aggregation="max",gradient_clip=.04,source_mask_box=(10,7,38,39))
    model=CupLocator().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4)
    yy,xx=torch.meshgrid(torch.arange(96,device="cuda"),torch.arange(96,device="cuda"),indexing="ij")
    best=None
    for epoch in range(1,141):
        model.train()
        if epoch in (76,111):
            optimizer.param_groups[0]["lr"]/=2.
        for ids in torch.randperm(len(images),device="cuda").split(16):
            dx,dy=np.random.randint(-14,15,size=2)
            x=F.pad(images[ids].permute(0,3,1,2),(14,14,14,14),value=180)
            x=x[:,:,14-dy:398-dy,14-dx:398-dx].permute(0,2,3,1).float()
            x=(x*(.86+.28*torch.rand(len(x),1,1,1,device="cuda"))).clamp(0,255)
            u=uv[ids]+uv.new_tensor([dx,dy])
            has=present[ids]
            cell=(u/4).floor().long()
            gaussian=torch.exp(-((xx-cell[:,0,None,None]).square()+(yy-cell[:,1,None,None]).square())/4.5)*has[:,None,None]
            output=model(x)
            logits=output[:,0]
            probability=logits.sigmoid()
            pos=gaussian==1
            heat=(-(1-probability).square()*F.logsigmoid(logits)*pos
                  -probability.square()*F.logsigmoid(-logits)*(1-gaussian)**4*~pos).sum()/has.sum().clamp_min(1)
            offset=output[torch.arange(len(x),device="cuda"),1:,cell[:,1],cell[:,0]].sigmoid()
            offset_loss=(F.smooth_l1_loss(offset[has],(u/4-cell)[has]) if has.any() else output.sum()*0)
            loss=heat+3*offset_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        if epoch%10:
            continue
        check=score_locator(model,validation,bank)
        fit=score_locator(model,training,bank)
        minimum=min(r["min_score"] for r in fit+check)
        maximum=max(r["max_erased_score"] for r in fit+check)
        error=max(r["max_error_mm"] for r in check)
        target_error=max(r["max_error_mm"] for r in fit if r["failed_before"])
        print("NECK",epoch,round(error,3),round(target_error,3),round(minimum,3),round(maximum,3),flush=True)
        score=(sum(r["max_error_mm"]>2.5 for r in fit+check), max(0.,maximum+.03-minimum), error,target_error)
        if best is None or score<best[0]:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best=(score,state,dict(epoch=epoch,train=fit,validation=check,threshold=(minimum+maximum)/2,
                                   margin=minimum-maximum,train_images=len(images),validation_episodes=len(validation)))
    _,state,report=best
    identity=hashlib.sha256(b"".join(t.numpy().tobytes() for t in state.values())).hexdigest()
    proposal=dict(kind="cup_heatmap",state=state,identity=identity)
    artifact=dict(proposal=proposal,refinement=bank)
    torch.save(artifact,STUDY/"neck-heatmap.pt")
    report["passed"]=best[0][0]==0 and report["margin"]>=.03
    report["checkpoint_sha256"]=sha(STUDY/"neck-heatmap.pt")
    save(STUDY/"neck-fit.json",report)


def validate_fusion(rows):
    from pour_temporal_localization import consensus
    artifact_path=STUDY/"neck-heatmap.pt"
    raw=read(STUDY/"neck-fit.json")
    artifact=dict(torch.load(artifact_path,map_location="cpu",weights_only=False),
                  proposal_floor=.65,refinement_floor=.68,max_refine_px=6.)
    report=[]
    for row in rows:
        saved=torch.load(row["checkpoint"],map_location="cpu",weights_only=False)
        saved["source_heatmap"]=artifact
        limit=saved["thresholds"]["source_localization_score"]
        settings=saved["localization_consensus"]["source"]
        frames=row["frames"]
        candidates=[locate_initial_source(im,saved) for im in frames]
        # These frames were sampled three frames apart. Check their spatial
        # consistency here, then verify the real eight-vote rule in simulation.
        check=dict(settings,votes=4,window=6)
        accepted=consensus([(p.tolist(),s) for p,s,_ in candidates],limit,check)
        neck_only=0
        for i,im in enumerate(frames):
            _,_,method=locate_initial_source(erased(im,row["uv"],i%2),saved)
            neck_only+=method=="neck_heatmap"
        estimate=(bottle_base_from_neck(accepted["uv"],row["cal"],row["lip"])[:2]
                  if accepted else None)
        error=float(np.linalg.norm(estimate-row["base"][:2])*1000) if estimate is not None else None
        report.append(dict(name=row["name"],source_sha256=row["sha256"],failed_before=row["failed"],
                           error_mm=error,erased_heatmap_acceptances=neck_only,
                           heatmap_frames=sum(m=="neck_heatmap" for _,_,m in candidates)))
    failures=[r for r in report if r["failed_before"]]
    passed=all(r["error_mm"] is not None and r["error_mm"]<=2.5 for r in report)
    passed=passed and all(r["erased_heatmap_acceptances"]==0 for r in report) and len(failures)==7
    save(STUDY/"neck-fusion.json",dict(passed=passed,cases=report,fit_epoch=raw["epoch"],
                                      checkpoint_sha256=sha(artifact_path),
                                      operating_point={k:artifact[k] for k in
                                                       ("proposal_floor","refinement_floor","max_refine_px")}))
    print(json.dumps(dict(passed=passed,failures=[r for r in report if r["error_mm"] is None
                     or r["error_mm"]>2.5 or r["erased_heatmap_acceptances"]>0])),flush=True)


def wrist_examples():
    recent=read(PARENT/"plan.json")
    sample=next(s for s in recent["test"] if s["bottle_height_scale"]>1.)
    parent=torch.load(sample["checkpoint"],map_location="cpu",weights_only=False)
    roi=parent["wide_empty_roi"]
    rows=[]
    for case,origin in examples():
        if not case["grasp_verified"] or case["bottle_geometry"]["lip_height_m"]<.14:
            continue
        p=folder(case)
        decisions=read(p/"policy-decisions.json")
        samples={round(x["t"],6):x for x in read(p/"samples.json")}
        spec=read(p/"parameters.json")
        record=p/"episode.hdf5"
        assert sha(record)==case["recording_sha256"]
        frames=[]
        with h5py.File(record) as file:
            data=file["data/demo_0"]
            for i,(row,time) in enumerate(zip(decisions,data["time"][:])):
                assert abs(row["t"]-time)<1e-5
                if row["stop_latched"]:
                    break
                cup=row.get("sized_cup_perception") or {}
                views=row.get("empty_views") or {}
                if not views or not cup.get("valid") or row["estimated_tilt_from_joints"]<112.:
                    continue
                sample=samples[round(time,6)]
                image=wrist_crop(decode(data["obs/wrist_image"][i],spec["image_encoding"]),roi)
                frames.append((float(time),float(sample["bottle_ml"]),image,dict(
                    held=bool(views["encoder_hold"]),present=row["source_present_probability"],
                    tilt=row["estimated_tilt_from_joints"],cup_ready=True,
                    flow_ml_s=row["visual_flow_fraction_s"]*spec["declared_cup_geometry"]["legacy_brim_nominal_ml"],
                    upper_level=max(*cup.get("guard_filtered",cup["filtered"]).values(),*cup["raw"].values()))))
        if frames:
            zero=next((t for t,s in samples.items() if s["bottle_ml"]==0.),None)
            rows.append(dict(name=spec["name"],origin=origin,pair=spec["pair_index"],frames=frames,
                             zero=zero,sha256=case["recording_sha256"]))
    return rows,roi,parent


@torch.inference_mode()
def infer_wrist(model,rows):
    probabilities={}
    model.eval()
    for row in rows:
        x=torch.tensor(np.stack([f[2] for f in row["frames"]]),device="cuda")
        probabilities[row["name"]]=torch.cat([model(a).sigmoid() for a in x.split(96)]).cpu().numpy()
    return probabilities


def replay_wrist(rows,probs,config):
    result=[]
    for row in rows:
        gate=TrainedEmptyGate(config)
        first=None
        last_time=None
        for (time,volume,_,evidence),p in zip(row["frames"],probs[row["name"]]):
            if last_time is not None and time-last_time>.051:
                gate.votes=0
            last_time=time
            if gate.update(probability=float(p),**evidence):
                first=(round(time,6),volume)
                break
        lag=(round(first[0]-row["zero"],6) if first and row["zero"] is not None else None)
        result.append(dict(name=row["name"],sha256=row["sha256"],first_zero_s=row["zero"],
                           candidate_s=first[0] if first else None,source_at_candidate_ml=first[1] if first else None,
                           lag_s=lag,frames=len(row["frames"])))
    return result


def fit_wrist(rows,roi,parent):
    validation=[r for r in rows if r["origin"]=="wide_benchmark" and r["pair"]%4==3]
    validation_names={v["name"] for v in validation}
    train=[r for r in rows if r["name"] not in validation_names]
    assert set(SLOW)<={r["name"] for r in train}
    images,labels,weights=[],[],[]
    for row in train:
        frames=row["frames"][::2]
        counts={k:sum((f[1]==0.)==k for f in frames) for k in (True,False)}
        for _,volume,image,_ in frames:
            empty=volume==0.
            images.append(image)
            labels.append(float(empty))
            weights.append((2. if row["name"] in SLOW else 1.)/counts[empty])
    x=torch.tensor(np.stack(images),device="cuda")
    y=torch.tensor(labels,device="cuda")
    w=torch.tensor(weights,device="cuda")
    w/=w.mean()
    model=EmptyWrist().cuda()
    model.load_state_dict(parent["wide_empty_model"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    best=None
    for epoch in range(1,91):
        model.train()
        if epoch in (46,71):
            optimizer.param_groups[0]["lr"]/=2.
        for ids in torch.randperm(len(y),device="cuda").split(80):
            loss=(F.binary_cross_entropy_with_logits(model(augment(x[ids])),y[ids],reduction="none")*w[ids]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        if epoch%5:
            continue
        probs=infer_wrist(model,rows)
        for probability in (.82,.88,.92,.95,.97):
            for votes in (8,10,12):
                config=dict(tilt=120.,flow=1.25,probability=probability,votes=votes)
                outcome=replay_wrist(rows,probs,config)
                bad=sum(r["source_at_candidate_ml"] not in (None,0.) for r in outcome)
                target=[r for r in outcome if r["name"] in SLOW]
                misses=sum(r["lag_s"] is None or r["lag_s"]>2. for r in target)
                quick=[r["lag_s"] for r in outcome if r["lag_s"] is not None]
                val_ok=sum(r["name"] in validation_names and r["lag_s"] is not None and r["lag_s"]<=2. for r in outcome)
                score=(bad,misses,-val_ok,-sum(0<=t<=2. for t in quick),sum(quick))
                if best is None or score<best[0]:
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    best=(score,state,dict(epoch=epoch,gate=config,replay=outcome,train_frames=len(y),
                                          train_episodes=len(train),validation_names=[r["name"] for r in validation]))
        print("WRIST",epoch,best[0],best[2]["gate"],flush=True)
    score,state,report=best
    torch.save(dict(state=state,roi=roi,gate=report["gate"]),STUDY/"exact-empty-wrist.pt")
    report["passed"]=score[:2]==(0,0)
    report["checkpoint_sha256"]=sha(STUDY/"exact-empty-wrist.pt")
    save(STUDY/"wrist-fit.json",report)


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name(0)
    torch.set_num_threads(3)
    torch.manual_seed(26111306)
    np.random.seed(26111306)
    if not (STUDY/"neck-fit.json").exists():
        fit_locator(neck_examples())
    if not (STUDY/"wrist-fit.json").exists():
        fit_wrist(*wrist_examples())
    if not (STUDY/"neck-fusion.json").exists():
        validate_fusion(neck_examples())
    assert read(STUDY/"neck-fusion.json")["passed"] and read(STUDY/"wrist-fit.json")["passed"]


if __name__=="__main__":
    main()
