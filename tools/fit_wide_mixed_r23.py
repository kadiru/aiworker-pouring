"""Adjust the mid-pour hand trajectory against the paired normal 05 rim rebounds."""
from itertools import product
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "wide-mixed-r22"
STUDY = ROOT / "wide-mixed-r23"
sys.path.insert(0, str(STUDY / "src"))
from pour_wide_motion import MAX_TAIL_DROP_M
from fit_wide_mixed_r21 import score_rebound_paths
from fit_wide_mixed_r20 import state_for
from fit_wide_mixed_r14_stream import TRAIN, HOLDOUT, sample, trace
from fit_wide_mixed_r12_stream import archived_stream, recorded_stream
from fit_wide_mixed_r6 import save

LAYOUT = "05_normal"


def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    case = sample(PARENT.name, LAYOUT)
    observed = trace(case)
    assert case["measured_loss_particles"] == len(observed["particles"]) == 2
    path, state = state_for(LAYOUT, PARENT)
    span, focus = (105., 131.), (115., 124.)
    train = [archived_stream(s, LAYOUT, span, 80, True) for s in (*TRAIN, "wide-mixed-r20")]
    train.append(recorded_stream(case, span, 80, True))
    holdout = [archived_stream(s, LAYOUT, span, 80, True) for s in (*HOLDOUT, "wide-mixed-r21")]
    coarse = (*state["tail_window_deg"], state["tail_setback_m"], state["tail_drop_m"])
    advance = state["late_mouth_shift_m"]
    start, end, final_x, final_drop = state["finish_mouth_pose"]
    assert 0 <= final_drop <= MAX_TAIL_DROP_M and abs(final_x) <= .010
    base = (*coarse, advance, start, end, final_x, final_drop)
    # Retain the early trajectory and terminal pose; delay or reduce the
    # forward/upward motion during the observed 118.72-degree rebound.
    profiles = [(*coarse, shift, onset, finish, final_x, final_drop)
                for shift, onset, finish in product((0., .002, .004, .006, advance),
                                                    (116., 117., 118., 119., 120.),
                                                    (123., 124., 125., 126., 127.))
                if finish-onset >= 4.]
    grid = torch.tensor([base, *(p for p in profiles if p != base)], device="cuda", dtype=torch.float64)
    best, audit, ratios, passed = score_rebound_paths(grid, observed, train, holdout, focus)
    chosen = grid[best].tolist()
    options = {"finish_mouth_pose": chosen[5:], "late_mouth_shift_m": chosen[4]}
    changes = {k: v for k, v in options.items() if v != state[k]}
    fit = dict(passed=passed, checkpoint=path, candidates=len(grid), focus_deg=focus, parent_options={
               k:state[k] for k in options}, mouth_options=changes, rim_ratios=ratios, cases=audit)
    save(STUDY/"spill-trace.json", dict(cases={LAYOUT:observed}))
    save(STUDY/"stream-fit.json", dict(passed=passed, device=torch.cuda.get_device_name(), cases={LAYOUT:fit}))
    print(passed, changes, ratios, flush=True)
    assert passed and changes

if __name__ == "__main__":
    main()
