"""Fit the two shared source detectors and the normal 09 coarse-pour path."""
from itertools import product
import sys

from diagnose_wide_mixed_r25 import ARCHIVE, MOTION, PARENT, ROOT, STUDY, TARGETS
import torch
from fit_wide_mixed_r21 import fit_proposal, score_rebound_paths
from fit_wide_mixed_r20 import fit_neck, state_for
from fit_wide_mixed_r19 import neck_rows
from fit_wide_mixed_r14_stream import sample, trace
from fit_wide_mixed_r12_stream import archived_stream, recorded_stream
from fit_wide_mixed_r6 import read, save
from pour_size_autonomy import sha


def neck_fit():
    diagnosis = read(STUDY / "startup-diagnosis.json")["bottles"]
    assert set(diagnosis) == set(TARGETS) and all(not r["accepted"] for r in diagnosis.values())
    revisions = ("r23", "r24")
    fits = {}
    for key, pairs, targets in (("12_normal", ("12",), TARGETS[:2]), ("14_normal", ("14","00"), TARGETS[2:])):
        rows = [r for r in neck_rows(STUDY, ARCHIVE) if r["name"].split("_test_")[1][:2] in pairs]
        history = ROOT / "wide-mixed-r20"
        rows.extend(r for r in neck_rows(history) if r["name"].split("_test_")[1][:2] in pairs)
        train_rows = [r for r in rows if r["name"].split("_test_")[1][:2] == key[:2]]
        proposal, training = fit_proposal(key, train_rows, parent=PARENT, revisions=revisions, seed=26112400+int(key[:2]))
        fit = fit_neck(key, rows, PARENT, STUDY, revisions, proposal, update_primary=True, template_keys=targets)
        fits[key] = dict(fit, proposal_training=training, targeted_layouts=targets)
    passed = all(f["passed"] for f in fits.values())
    save(STUDY / "neck-replay.json",dict(passed=passed,cases=fits,startup_sha256=sha(STUDY/"inputs"/ARCHIVE)))
    assert passed


def motion_fit():
    case = sample(PARENT.name, MOTION)
    evidence = trace(case)
    assert len(evidence["particles"]) == case["measured_loss_particles"] == 4
    span, focus = (106., 128.), (111., 120.)
    train = [archived_stream(f"wide-mixed-r{r}", MOTION, span, 80, True) for r in (12,20,22)]
    train.append(recorded_stream(case, span, 80, True))
    holdout = [archived_stream(f"wide-mixed-r{r}", MOTION, span, 80, True) for r in (11,21,23)]
    path, state = state_for(MOTION, PARENT)
    on, off = state["tail_window_deg"]
    back, drop, advance = [state[k] for k in ("tail_setback_m","tail_drop_m","late_mouth_shift_m")]
    assert state.get("finish_mouth_pose") is None
    # Rejoin the incumbent terminal position after the two coarse-pour losses.
    finish = (122.,126.,advance-back,drop)
    base = (on,off,back,drop,advance,*finish)
    profiles = [(a,b,s/1000,z/1000,shift/1000,*finish) for (a,b),s,z,shift in
                product(((104.,110.),(106.,112.),(108.,113.),(109.,114.),(110.,115.)),
                        (6,7,8,9,10),(5,6,7,8),(0,2,4,6,8))]
    grid = torch.tensor([base,*profiles],device="cuda",dtype=torch.float64)
    height = train[-1][1]["declared_cup_geometry"]["rim_z_m"]
    events = [next(f for f in reversed(track) if f["xyz"][2] >= height) for track in evidence["particles"].values()]
    assert all(focus[0] <= f["tilt"] <= focus[1] for f in events)
    best, audit, ratios, passed = score_rebound_paths(grid,evidence,train,holdout,focus,events=events)
    start,end,back,down,shift,*terminal = grid[best].tolist()
    options = dict(tail_window_deg=[start,end],tail_setback_m=back,tail_drop_m=down,late_mouth_shift_m=shift,finish_mouth_pose=terminal)
    fit = dict(passed=passed,checkpoint=path,candidates=len(grid),focus_deg=focus,event_frames=events,cases=audit,
               rim_ratios=ratios,mouth_options={k:v for k,v in options.items() if state.get(k)!=v})
    save(STUDY/"spill-trace.json",dict(cases={MOTION:evidence}))
    save(STUDY/"stream-fit.json",dict(passed=passed,device=torch.cuda.get_device_name(),cases={MOTION:fit}))
    print("motion",passed,fit["mouth_options"],ratios,flush=True)
    assert passed


if __name__ == "__main__":
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    if sys.argv[1:] == ["motion"]:
        motion_fit()
    else:
        assert sys.argv[1:] == ["neck"]
        neck_fit()
