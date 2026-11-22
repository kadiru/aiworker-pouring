"""Gate r16 on exact camera replay, position accuracy, absent-cup controls and frozen assets."""
import inspect
from pathlib import Path

import numpy as np
import torch

from diagnose_r16_startup import ROOT, STUDY, SizedCupPerception, old_tracker, plane_point, project, read, save, setup, sha, trials

PARENT = ROOT / "wide-mixed-r15"


def erased(frame, center, fill):
    x,y = np.rint(center).astype(int)
    x0,x1 = max(0,x-140),min(frame.shape[1],x+140)
    y0,y1 = max(0,y-75),min(frame.shape[0],y+160)
    replaced = frame.copy()
    if fill == "flat":
        replaced[y0:y1,x0:x1] = 180
    else:
        # Fill with neighboring real pixels from this frame, preserving local vertical texture.
        col = x1 if x1 < frame.shape[1] else x0-1
        replaced[y0:y1,x0:x1] = frame[y0:y1,col:col+1]
    return replaced


@torch.inference_mode()
def measure(case, frame, parent_type, study=STUDY, target_suffix="02_normal", frame_limit=42):
    cfg,geom,cal,z,xy = setup(case)
    truth_uv = project([*xy,z],cal)
    names = ["parent","candidate","flat","background"]
    target = case["name"].endswith(target_suffix)
    if target:
        names += ["translated_left","translated_right"]
    trackers = {name:SizedCupPerception(cfg,geom) for name in names}
    assert Path(inspect.getfile(type(trackers["candidate"].tracker))).resolve() == study/"src/pour_cup_outline.py"
    trackers["parent"].tracker = parent_type(cfg["tracker"],geom,"cuda")
    observed = {}
    for i in range(frame_limit):
        image = frame(i)
        for name, model in trackers.items():
            if name in observed:
                continue
            view, expected_xy = image, xy
            if name in ("flat","background"):
                view = erased(image,truth_uv,name)
            elif name.startswith("translated"):
                dx,dy = (-5,3) if name.endswith("left") else (6,-4)
                padded=np.pad(image,((6,6),(6,6),(0,0)),mode="edge")
                view=padded[6-dy:1030-dy,6-dx:1030-dx]
                expected_xy = plane_point(truth_uv + [dx,dy],z,cal)[:2]
            result = model.track(view)
            if result["valid"]:
                uv=np.asarray(result["uv"])
                error=np.linalg.norm(plane_point(uv,z,cal)[:2]-expected_xy)
                observed[name] = dict(frame=i,error_mm=float(error*1000),uv=uv.tolist())
    return dict(name=case["name"],recording_sha256=case["recording_sha256"],
                **{name:observed.get(name) for name in names})


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    before={p.name:sha(p) for p in (PARENT/"src").glob("*.py")}
    after={p.name:sha(p) for p in (STUDY/"src").glob("*.py")}
    assert set(before)==set(after)
    changed=[n for n in before if before[n]!=after[n]]
    assert changed==["pour_cup_outline.py"]
    plan=read(PARENT/"plan.json")
    assert all(sha(p)==h for p,h in plan["checkpoints"].items())
    parent_type=old_tracker()
    rows=[]
    for case, frame in trials():
        row=measure(case,frame,parent_type)
        rows.append(row)
        print(row["name"], row["candidate"], flush=True)
    target=next(r for r in rows if r["name"]=="SA_A_wide_mixed_r15_test_02_normal")
    positives=[r[k] for r in rows for k in ("candidate","translated_left","translated_right") if k in r]
    missing=sum(r[k] is not None for r in rows for k in ("flat","background"))
    changed_passes=[r["name"] for r in rows if r["parent"] is not None and r["candidate"]!=r["parent"]]
    passed=(len(rows)==32 and target["parent"] is None and missing==0
            and all(p is not None and p["error_mm"]<=2.5 for p in positives)
            and all(r["candidate"] is not None for r in rows if r["parent"] is not None))
    result=dict(passed=passed,device=torch.cuda.get_device_name(),frozen_checkpoints=plan["checkpoints"],
                parent_plan_sha256=sha(PARENT/"plan.json"),startup_frames_sha256=sha(STUDY/"startup.h5"),
                changed_runtime_sources=changed,unchanged_runtime_sources=len(before)-len(changed),
                locator_sha256=after[changed[0]],positive_sequences=len(positives),negative_sequences=2*len(rows),
                false_acceptances=missing,changed_previously_confirmed=changed_passes,cases=rows)
    save(STUDY/"cup-startup-replay.json",result)
    print(passed,missing,changed_passes,flush=True)
    assert passed


if __name__=="__main__":
    main()
