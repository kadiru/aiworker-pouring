"""Trace the three startup timeouts on their original Isaac Sim camera recordings."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "wide-mixed-r24"
STUDY = ROOT / "wide-mixed-r25"
sys.path.insert(0, str(STUDY / "src"))

from pour_wide_motion import MAX_TAIL_DROP_M
from pour_cup_outline import OutlineCupTracker, refine_outline
from perception_followup import VerifiedCupTracker
from pour_guard_vision import locate, project
from pour_cup_locator import locate_cup
from fit_wide_mixed_r20 import state_for
from fit_wide_mixed_r19 import neck_rows, replay
from fit_wide_mixed_r14_stream import sample
from fit_wide_mixed_r6 import h5py, folder, save
from extract_r19_neck import write_startup
from diagnose_r22_startup import native_trials, setup
from pour_size_autonomy import sha

TARGETS = ("12_normal", "12_swap", "14_normal")
MOTION = "09_normal"
ARCHIVE = "native-neck.h5"


def collect():
    with h5py.File(STUDY / "inputs" / ARCHIVE, "x") as archive:
        for revision in (11, 12, 20, 21, 22, 23, 24):
            for pair in (0, 12, 14):
                for side in ("normal", "swap"):
                    row = sample(f"wide-mixed-r{revision}", f"{pair:02}_{side}")
                    path = folder(row) / "episode.hdf5"
                    assert sha(path) == row["recording_sha256"]
                    write_startup(archive, path, row)


@torch.inference_mode()
def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    assert MAX_TAIL_DROP_M == .008
    collect()
    checks = {}
    for r in neck_rows(STUDY, ARCHIVE):
        name = r["name"]
        if "_r24_test_" not in name or not name.endswith(TARGETS):
            continue
        key = name.split("_test_")[1]
        _, state = state_for(key, PARENT)
        uv, method, count = replay(r["frames"], state)
        print(key, "parent", uv, method, count, flush=True)
        measures = []
        for field in ("source_heatmap", "source_heatmap_fallback"):
            bank = state.get(field)
            if not bank:
                continue
            for i in (0, 6, 15, 30):
                image = r["frames"][i]
                point, score = locate_cup(image, bank["proposal"], "cuda")
                fine, confidence = locate(image, bank["refinement"], expected_uv=point, device="cuda")
                measures.append(dict(field=field,frame=i,uv=point.tolist(),score=float(score),fine=fine.tolist(),
                                     confidence=float(confidence),floor=bank["proposal_floor"],refinement_floor=bank["refinement_floor"]))
        checks[key] = dict(recording_sha256=r["recording_sha256"], accepted=uv is not None, frames=count,
                           truth=r["uv"].tolist(), measures=measures)
    cup = []
    for row, frame in native_trials(PARENT, {12}):
        if not row["name"].endswith("12_normal"):
            continue
        cfg, geo, cal, z, xy = setup(row)
        tracker = OutlineCupTracker(cfg["tracker"],geo,"cuda")
        for i in range(41):
            im = frame(i)
            pt, score = tracker.detect(im)
            if i in (0, 6, 15, 24, 30, 40):
                fine, raw = VerifiedCupTracker.detect(tracker,im)
                rim = refine_outline(im,fine,cal,geo)
                alt = tracker.alternate_startup(im)
                cup.append(dict(frame=i,uv=pt.tolist(),score=float(score),threshold=tracker.artifact["threshold"],
                           fine=fine.tolist(),raw=float(raw),presence=tracker.presence_score(im,fine),
                           alt=None if alt is None else [alt[0].tolist(),float(alt[1])],
                           rim=None if rim is None else rim.tolist(),truth=project([*xy,z],cal).tolist()))
    _, state = state_for(MOTION,PARENT)
    options={k:state.get(k) for k in ("tail_window_deg","tail_setback_m","tail_drop_m","late_mouth_shift_m","finish_mouth_pose")}
    save(STUDY / "startup-diagnosis.json",dict(bottles=checks,cup=cup,motion_options=options))


if __name__ == "__main__":
    main()
