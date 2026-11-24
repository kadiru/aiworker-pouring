"""Diagnose the missed cup in normal 07 and refit only the coarse cup heatmap."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "wide-mixed-r25"
STUDY = ROOT / "wide-mixed-r26"
KEY = "07_normal"
sys.path.insert(0, str(STUDY / "src"))

from pour_size_autonomy import SizedCupPerception, sha
from pour_cup_outline import OutlineCupTracker
from perception_followup import VerifiedCupTracker
from pour_guard_vision import plane_point, project
from diagnose_r22_startup import native_trials, setup, translated
from fit_wide_mixed_r25_cup import fit
from fit_wide_mixed_r6 import read, save
from replay_r16_startup import measure


@torch.inference_mode()
def diagnose():
    rows = []
    for case, frame in native_trials(PARENT, {7}):
        cfg, geom, cal, z, xy = setup(case)
        detector = SizedCupPerception(cfg,geom)
        tracker = detector.tracker
        truth = project([*xy,z],cal)
        frames = []
        for i in range(41):
            image = frame(i)
            result = detector.track(image)
            item = dict(frame=i,valid=result["valid"],reason=result["reason"],score=float(result["score"]))
            if i in (0, 6, 15, 30):
                coarse, cs = VerifiedCupTracker.detect(tracker,image,False)
                fine, fs = VerifiedCupTracker.detect(tracker,image)
                item.update(coarse_uv=coarse.tolist(),coarse_error_px=float(np.linalg.norm(coarse-truth)),
                            coarse_score=float(cs),fine_uv=fine.tolist(),fine_score=float(fs),
                            presence_at_proposal=tracker.presence_score(image,coarse),
                            presence_at_label=tracker.presence_score(image,truth))
            frames.append(item)
            if result["valid"]:
                break
        rows.append(dict(name=case["name"],truth=truth.tolist(),threshold=tracker.artifact["threshold"],
                         accepted=frames[-1]["valid"],recording_sha256=case["recording_sha256"],frames=frames))
    assert [r["name"].split("_test_")[1] for r in rows if not r["accepted"]] == [KEY]
    save(STUDY/"cup-diagnosis.json",dict(cases=rows))


def select_blend(cfg, fitted):
    """Choose the smallest weight change that fixes the fitting views."""
    original = cfg["tracker"]["coarse_state"]
    audits = []
    for alpha in (.125, .25, .375, .5, .625, .75, .875):
        blend = {k: torch.lerp(v,fitted[k],alpha) if v.is_floating_point() else v for k,v in original.items()}
        candidate = dict(cfg,tracker=dict(cfg["tracker"],coarse_state=blend))
        rows = []
        for revision in (24,25):
            for case, frame in native_trials(ROOT/f"wide-mixed-r{revision}",{7}):
                _, geom, cal, z, xy = setup(case)
                uv = project([*xy,z],cal)
                for dx,dy in ((0,0),(-5,3),(6,-4)):
                    model = SizedCupPerception(candidate,geom)
                    expected = plane_point(uv+[dx,dy],z,cal)[:2]
                    error = None
                    for i in range(21):
                        image = frame(i)
                        hit = model.track(translated(image,dx,dy) if dx or dy else image)
                        if hit["valid"]:
                            measured = plane_point(np.asarray(hit["uv"]),z,cal)[:2]
                            error = float(1000*np.linalg.norm(measured-expected))
                            break
                    rows.append(dict(name=case["name"],shift=[dx,dy],frame=i,error_mm=error))
        passed = all(r["error_mm"] is not None and r["error_mm"]<=2.5 for r in rows)
        audits.append(dict(alpha=alpha,passed=passed,views=rows))
        if passed:
            save(STUDY/"blend-selection.json",dict(passed=True,selected=alpha,attempts=audits))
            return blend
    save(STUDY/"blend-selection.json",dict(passed=False,attempts=audits))
    raise ValueError("No blend passed the calibration views")


def fit_and_check(calibrate):
    plan = read(PARENT/"plan.json")
    spec = next(r for r in plan["test"] if r["name"].endswith(KEY))
    checkpoint = spec["checkpoint"]
    assert sha(checkpoint) == plan["checkpoints"][checkpoint]
    assert {r["pair_index"] for r in plan["test"] if r["checkpoint"]==checkpoint} == {7}
    cfg = torch.load(checkpoint,map_location="cpu",weights_only=False)["sized_cup_perception"]
    artifact = STUDY/"cup-coarse-07.pt"
    if calibrate:
        initial = read(STUDY/"cup-unblended-replay.json")
        assert initial["checkpoint"]==checkpoint and initial["artifact_sha256"]==sha(artifact)
        weights = select_blend(cfg,torch.load(artifact,map_location="cpu",weights_only=False))
        training = dict(initial["training"],blend=read(STUDY/"blend-selection.json")["selected"])
        artifact = STUDY/"cup-selected-07.pt"
    else:
        weights, training = fit(cfg,revisions=(24,25),pair=7,seed=2611243707)
    torch.save(weights,artifact)
    candidate = dict(cfg,tracker=dict(cfg["tracker"],coarse_state=weights))
    rows = []
    for revision in (11,12,20,21,22,23,24,25):
        for case, frame in native_trials(ROOT/f"wide-mixed-r{revision}",{7}):
            row = measure(case,frame,OutlineCupTracker,STUDY,("07_normal","07_swap"),41,
                          parent_config=cfg,candidate_config=candidate)
            rows.append(row)
    positives = [r[k] for r in rows for k in ("candidate","translated_left","translated_right")]
    negatives = [r[k] for r in rows for k in ("flat","background")]
    target = next(r for r in rows if r["name"]==spec["name"])
    passed = (len(rows)==16 and target["parent"] is None and
              all(v is not None and v["error_mm"]<=2.5 for v in positives) and not any(negatives))
    report = "cup-startup-replay.json" if calibrate else "cup-unblended-replay.json"
    save(STUDY/report,dict(passed=passed,device=torch.cuda.get_device_name(),
         parent_plan_sha256=sha(PARENT/"plan.json"),checkpoint=checkpoint,artifact_sha256=sha(artifact),
         training=training,positives=len(positives),negatives=len(negatives),
         false_acceptances=sum(v is not None for v in negatives),cases=rows))
    print("Cup replay passed",passed,"failed positive views",sum(v is None or v["error_mm"]>2.5 for v in positives),flush=True)
    if calibrate:
        assert passed


if __name__ == "__main__":
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    if sys.argv[1:]==["diagnose"]:
        diagnose()
    else:
        assert sys.argv[1:] in ([],["calibrate"])
        fit_and_check(bool(sys.argv[1:]))
