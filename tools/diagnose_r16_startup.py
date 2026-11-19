"""Inspect initial proposals, presence scores, reference establishment and projected errors."""
from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT/"wide-mixed-r16"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import SizedCupPerception, sha
from dynamic_cup_tracker import CupTracker
from pour_guard_vision import project, plane_point
from validate_pour_recording import decode,h5py
from fit_wide_mixed_r6 import folder,read,save
sys.path.insert(0,str(STUDY/"src"))


def trials():
    rows=read(ROOT/"wide-mixed-r15/completion.json")["test_cases"]
    with h5py.File(STUDY/"startup.h5") as images:
        for row in rows:
            group=images[row["name"]]
            assert group.attrs["recording_sha256"] == row["recording_sha256"]
            yield row, lambda i: decode(group[str(i)][:],"jpeg95")
    for study in ("wide-mixed-r11","wide-mixed-r12"):
        row=next(c for c in read(ROOT/study/"completion.json")["test_cases"] if c["name"].endswith("02_normal"))
        path=folder(row)/"episode.hdf5"
        assert sha(path)==row["recording_sha256"]
        with h5py.File(path) as file:
            frames=file["data/demo_0/obs/cup_detail_image"]
            yield row, lambda i: decode(frames[i],"jpeg95")


def setup(row):
    spec=read(folder(row)/"parameters.json")
    state=torch.load(spec["checkpoint"],map_location="cpu",weights_only=False)
    cfg=state["sized_cup_perception"]
    geom=spec["declared_cup_geometry"]
    cal=cfg["tracker"]["calibration"]
    z=cal["table_z"]+geom["rim_z_m"]
    truth=np.asarray(spec["cup_world"][:2])
    return cfg,geom,cal,z,truth


def old_tracker():
    path = ROOT/"wide-mixed-r15/src/pour_cup_outline.py"
    spec = importlib.util.spec_from_file_location("parent_cup_outline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OutlineCupTracker


@torch.inference_mode()
def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    rows=[]
    for case, frame in trials():
        if not case["name"].endswith("02_normal"):
            continue
        cfg,geom,cal,z,xy=setup(case)
        perception=SizedCupPerception(cfg,geom)
        tracker=old_tracker()(cfg["tracker"],geom,"cuda")
        perception.tracker=tracker
        uv=project([*xy,z],cal)
        log=[]
        for i in range(42):
            im=frame(i)
            center,coarse=CupTracker.detect(tracker,im,False)
            fine,score=CupTracker.detect(tracker,im,True)
            presence=tracker.presence_score(im,fine)
            result=perception.track(im)
            log.append(dict(frame=i,coarse=coarse,fine_score=score,presence=presence,reason=result["reason"],
                            valid=result["valid"], combined=result["score"], coarse_uv=center.tolist(),
                            fine_uv=fine.tolist(), uv=result.get("raw_uv"), history=len(tracker.history)))
            if result["valid"]:break
        row=dict(name=case["name"],truth_uv=uv.tolist(),threshold=tracker.artifact["threshold"],
                 retry=perception.retry_initial_cup,settings=tracker.artifact["settings"],
                 consensus=tracker.artifact["consensus"],frames=log)
        rows.append(row)
        print(row["name"],row["threshold"],row["retry"],log[0],log[-1],flush=True)
    save(STUDY/"startup-diagnosis.json",dict(cases=rows))


if __name__=="__main__":
    main()
