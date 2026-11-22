"""Trace the learned detector and its startup votes on the original normal 12 recordings."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "wide-mixed-r22"
PARENT = ROOT / "wide-mixed-r21"
sys.path.insert(0, str(STUDY / "src"))
from pour_size_autonomy import SizedCupPerception, sha
from pour_cup_outline import refine_outline
from perception_followup import VerifiedCupTracker
from pour_guard_vision import project, plane_point
from validate_pour_recording import h5py, decode
from diagnose_r16_startup import setup
from fit_wide_mixed_r6 import folder, read, save


def native_trials(study, indices=None):
    for row in read(study / "completion.json")["test_cases"]:
        if indices is not None and row["pair_index"] not in indices:
            continue
        path = folder(row) / "episode.hdf5"
        assert sha(path) == row["recording_sha256"]
        with h5py.File(path) as h:
            assert h.attrs["status"] == "complete"
            frames = h["data/demo_0/obs/cup_detail_image"]
            yield row, lambda i: decode(frames[i], "jpeg95")


def translated(image, dx, dy):
    h,w = image.shape[:2]
    pad = max(abs(dx), abs(dy))
    padded = np.pad(image, ((pad,pad),(pad,pad),(0,0)), mode="edge")
    return padded[pad-dy:pad-dy+h, pad-dx:pad-dx+w]


@torch.inference_mode()
def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    cases = []
    for study in (ROOT / "wide-mixed-r20", PARENT):
        for case, frame in native_trials(study, {12}):
            cfg, geo, cal, z, xy = setup(case)
            tracker = SizedCupPerception(cfg, geo)
            truth = project([*xy,z],cal)
            log = []
            for i in range(41):
                image = frame(i)
                t = tracker.tracker
                row = tracker.track(image)
                item = dict(frame=i, reason=row["reason"],valid=row["valid"],score=row["score"],
                            raw=row.get("raw_uv"), history=len(t.history))
                if i in (0,6,15,30):
                    point, fine = VerifiedCupTracker.detect(t,image)
                    coarse, cs = VerifiedCupTracker.detect(t,image,False)
                    outline = refine_outline(image,point,cal,geo)
                    item.update(fine=fine,coarse=cs,coarse_uv=coarse.tolist(),fine_uv=point.tolist(),
                                presence=t.presence_score(image,point),outline=None if outline is None else outline.tolist())
                    shifts=[]
                    for dx,dy in ((-4,0),(4,0),(0,-4),(0,4),(-4,-4),(4,4),(-8,0),(8,0),(0,-8),(0,8)):
                        u, sc = VerifiedCupTracker.detect(t,translated(image,dx,dy))
                        shifts.append([dx,dy,round(sc,4),round(float(np.linalg.norm(u-[dx,dy]-truth)),3)])
                    item["shifts"]=shifts
                log.append(item)
                if row["valid"]: break
            cases.append(dict(name=case["name"],recording_sha256=case["recording_sha256"],truth_uv=truth.tolist(),
                              threshold=t.artifact["threshold"],consensus=t.artifact["consensus"],frames=log))
    save(STUDY/"startup-diagnosis.json",dict(cases=cases))

if __name__=="__main__":
    main()
