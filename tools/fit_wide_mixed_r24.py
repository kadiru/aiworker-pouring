"""Fit bottle-neck confirmation and the coarse-pour trajectory from native simulator recordings."""
from itertools import product
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "wide-mixed-r23"
STUDY = ROOT / "wide-mixed-r24"
sys.path.insert(0, str(STUDY / "src"))
from fit_wide_mixed_r21 import score_rebound_paths, fit_proposal
from fit_wide_mixed_r20 import fit_neck, state_for
from fit_wide_mixed_r19 import neck_rows
from fit_wide_mixed_r14_stream import TRAIN, HOLDOUT, sample, trace
from fit_wide_mixed_r12_stream import archived_stream, recorded_stream
from fit_wide_mixed_r6 import folder, h5py, save
from extract_r19_neck import write_startup
from pour_cup_locator import locate_cup
from pour_guard_vision import locate
from pour_size_autonomy import sha

NECK = "04_swap"
MOTION = "13_normal"
STARTUPS = "native-04.h5"


def startup_archive(filename, revisions):
    path = STUDY / "inputs" / filename
    with h5py.File(path, "x") as archive:
        for revision, orientation in product(revisions, ("normal", "swap")):
            row = sample(f"wide-mixed-r{revision}", f"04_{orientation}")
            source = folder(row) / "episode.hdf5"
            assert sha(source) == row["recording_sha256"]
            write_startup(archive, source, row)


def diagnose():
    startup_archive(STARTUPS, (20, 21, 22, 23))
    _, state = state_for(NECK, PARENT)
    row = next(r for r in neck_rows(STUDY, STARTUPS) if r["name"].endswith("r23_test_" + NECK))
    measures = []
    for field in ("source_heatmap", "source_heatmap_fallback"):
        bank = state.get(field)
        if bank is None:
            continue
        for i in (0, 5, 15, 30):
            image = row["frames"][i]
            uv, proposal = locate_cup(image, bank["proposal"], "cuda")
            refined, match = locate(image, bank["refinement"], expected_uv=uv, device="cuda")
            measures.append(dict(field=field, frame=i, target=row["uv"].tolist(), proposal=uv.tolist(),
                                 refinement=refined.tolist(), score=float(proposal), match=float(match),
                                 floor=bank["proposal_floor"], refinement_floor=bank["refinement_floor"],
                                 max_refine_px=bank["max_refine_px"]))
    save(STUDY / "neck-diagnosis.json", dict(measures=measures))
    print(measures, flush=True)


def fit_motion():
    case = sample(PARENT.name, MOTION)
    observed = trace(case)
    assert len(observed["particles"]) == case["measured_loss_particles"] == 2
    path, state = state_for(MOTION, PARENT)
    span, focus = (108., 130.), (116., 124.)
    train = [archived_stream(s, MOTION, span, 80, True) for s in (*TRAIN, "wide-mixed-r20")]
    train.append(recorded_stream(case, span, 80, True))
    holdout = [archived_stream(s, MOTION, span, 80, True) for s in (*HOLDOUT, "wide-mixed-r21", "wide-mixed-r22")]
    window, finish = state["tail_window_deg"], state["finish_mouth_pose"]
    keys = ("tail_setback_m", "tail_drop_m", "late_mouth_shift_m")
    parent = (*window, *(state[k] for k in keys), *finish)
    profiles = [(*window, back / 1000, down / 1000, shift / 1000, *finish)
                for back, down, shift in product(range(4, 11), range(2, 9), (0, 2, 4, 6))]
    grid = torch.tensor([parent, *(p for p in profiles if p != parent)], device="cuda", dtype=torch.float64)
    # Both droplets skimmed the side wall. Project from their final observed
    # above-rim position with measured lateral velocity before they fell away.
    height = train[-1][1]["declared_cup_geometry"]["rim_z_m"]
    events = [next(f for f in reversed(track) if f["xyz"][2] >= height) for track in observed["particles"].values()]
    assert all(focus[0] <= f["tilt"] <= focus[1] for f in events)
    best, audit, ratios, passed = score_rebound_paths(grid, observed, train, holdout, focus, events=events)
    changes = {k:v for k,v in zip(keys, grid[best, 2:5].tolist()) if v != state[k]}
    save(STUDY / "spill-trace.json", dict(cases={MOTION:observed}))
    fit = dict(passed=passed, checkpoint=path, candidates=len(grid), focus_deg=focus, event_frames=events,
               parent_options={k:state[k] for k in keys}, mouth_options=changes, rim_ratios=ratios, cases=audit)
    save(STUDY / "stream-fit.json", dict(passed=passed, device=torch.cuda.get_device_name(), cases={MOTION:fit}))
    print("motion", passed, changes, ratios, flush=True)
    assert passed and changes


def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    if sys.argv[1:] == ["diagnose"]:
        diagnose()
    elif sys.argv[1:] == ["motion"]:
        fit_motion()
    else:
        assert not sys.argv[1:]
        historical = "native-04-historical.h5"
        startup_archive(historical, (11, 12))
        rows = [*neck_rows(STUDY, STARTUPS), *neck_rows(STUDY, historical)]
        revisions = ("r22", "r23")
        proposal, training = fit_proposal(NECK, rows, parent=PARENT, revisions=revisions, seed=26112324)
        result = fit_neck(NECK, rows, PARENT, STUDY, revisions, proposal, update_primary=True)
        result["proposal_training"] = training
        save(STUDY / "neck-replay.json", dict(passed=result["passed"], cases={NECK:result},
             startup_sha256={f:sha(STUDY / "inputs" / f) for f in (STARTUPS, historical)}))
        assert result["passed"]


if __name__ == "__main__":
    main()
