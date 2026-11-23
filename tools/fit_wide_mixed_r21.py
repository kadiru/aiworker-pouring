"""Fit normal 01/10 bottle-neck confirmation and late-pour motion for swapped 08."""
import hashlib
from itertools import product
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "wide-mixed-r20"
STUDY = ROOT / "wide-mixed-r21"
sys.path.insert(0, str(STUDY / "src"))

from pour_wide_motion import MAX_TAIL_DROP_M
from fit_wide_mixed_r20 import fit_neck, state_for
from fit_wide_mixed_r19 import neck_rows
from fit_wide_mixed_r14_stream import TRAIN, HOLDOUT, sample, scores, trace
from fit_wide_mixed_r13_stream import project
from fit_wide_mixed_r12_stream import archived_stream, counterfactual, recorded_stream
from fit_wide_mixed_r7 import dataset, train_epoch
from pour_cup_locator import CupLocator
from fit_wide_mixed_r6 import folder, h5py, read, save
from extract_r19_neck import write_startup
from pour_size_autonomy import sha
from refine_wide_mixed_r13_stream import rebound_at_rim

NECK = ("01_normal", "10_normal")
MOTION = ("08_swap",)


def current_startups():
    filename = "native-r20.h5"
    output = STUDY / "inputs" / filename
    cases = [r for r in read(PARENT / "completion.json")["test_cases"] if r["pair_index"] in (1, 10)]
    assert len(cases) == 4
    with h5py.File(output, "w") as archive:
        for row in cases:
            path = folder(row) / "episode.hdf5"
            assert sha(path) == row["recording_sha256"]
            write_startup(archive, path, row)
    return list(neck_rows(STUDY, filename))


def fit_terminal(key):
    case = sample(PARENT.name, key)
    observed = trace(case)
    span, focus = (114., 132.), (124., 132.)
    train = [archived_stream(s, key, span, 80, True) for s in TRAIN]
    train.append(recorded_stream(case, span, 80, True))
    holdout = [archived_stream(s, key, span, 80, True) for s in HOLDOUT]
    path, state = state_for(key, PARENT)
    window = state["tail_window_deg"]
    setback, drop = state["tail_setback_m"], state["tail_drop_m"]
    advance = state.get("late_mouth_shift_m", 0.)
    assert not state.get("finish_mouth_pose") and window[1] <= 124.
    coarse = (*window, setback, drop, advance)
    parent = (*coarse, 126., 130., advance - setback, drop)
    # Vary only the final approach to full tilt, where the two droplets rebounded.
    candidates = [(*coarse, on, off, offset / 1000., down / 1000.)
                  for (on, off), offset, down in product(((122., 126.), (124., 128.), (126., 129.)),
                                                        (-4, -6, -8, -10), (3, 5, 7, 8))
                  if down / 1000. <= MAX_TAIL_DROP_M]
    grid = torch.tensor([parent, *candidates], device="cuda", dtype=torch.float64)
    best, audit, ratios, passed = score_rebound_paths(grid, observed, train, holdout, focus)
    return dict(passed=passed, checkpoint=path, candidates=len(grid), cases=audit,
                rim_ratios=ratios, mouth_options={"finish_mouth_pose": grid[best, 5:].tolist()}), observed


def score_rebound_paths(grid, observed, train, holdout, focus, *, events=None):
    if events is None:
        events = [next(f for f in track if f["velocity"][2] > .1 and f["xyz"][0] > 0)
                  for track in observed["particles"].values()]
    rebounds = [[f["tilt"], *f["xyz"], *f["velocity"]] for f in events]
    points = grid.new_tensor(rebounds).T
    zero = torch.zeros_like(points[0])
    old_x, old_z = counterfactual(points[0], zero, zero, grid[:1])
    geom = train[-1][1]["declared_cup_geometry"]
    radius = geom["rim_radius_m"]
    ratios = rebound_at_rim((torch.cat((points, -old_x, -old_z)), geom["rim_z_m"], radius), grid)
    training_scores = [project(data, grid, focus) for _, _, data in train]
    costs = torch.stack([result[0] for result in training_scores]).mean(0)
    feasible = torch.ones(len(grid), dtype=torch.bool, device=grid.device)
    for _, impact, rim, radii in training_scores:
        escapes = (rim > radii).sum(1)
        feasible &= (escapes <= escapes[0]) & (torch.quantile(impact / radii, .95, dim=1) < .85)
    worst = ratios.amax(1)
    eligible = feasible & (worst <= max(1. - .0015 / radius, float(worst[feasible].min()) + .0005 / radius))
    best = int(costs.masked_fill(~eligible, torch.inf).argmin())
    audit = []
    for split, batch in (("train", train), ("holdout", holdout)):
        for row, _, data in batch:
            cost, impact, rim, radii = project(data, grid[[0, best]], focus)
            audit.append(dict(scores(row, data, split, cost), outside=(rim > radii).sum(1).tolist(),
                              impact_p95_ratio=torch.quantile(impact / radii, .95, dim=1).tolist()))
    passed = (best > 0 and bool(((ratios[0] - ratios[best]) * radius >= .0015).all()) and
              all(a["outside"][1] <= a["outside"][0] and a["impact_p95_ratio"][1] < .85 for a in audit))
    return best, audit, ratios[[0, best]].tolist(), passed


def fit_proposal(key, rows, *, parent=PARENT, revisions=("r19", "r20"), seed=26112210):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Train only on designated revisions. Other startups are reserved for replay.
    training = [dict(r, validate=False) for r in rows if any(f"_{rev}_test_" in r["name"] for rev in revisions)]
    assert len(training) == 4
    images, uv, present = dataset(training)
    _, state = state_for(key, parent)
    model = CupLocator().cuda()
    base = state["source_heatmap"]["proposal"]
    model.load_state_dict(base["state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    yy, xx = torch.meshgrid(torch.arange(96, device="cuda"), torch.arange(96, device="cuda"), indexing="ij")
    for _ in range(24):
        train_epoch(model, optimizer, images, uv, present, rng, xx, yy)
    weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    identity = hashlib.sha256(b"".join(t.numpy().tobytes() for t in weights.values())).hexdigest()
    return dict(kind="cup_heatmap", state=weights, identity=identity), dict(
        initialization=base["identity"], fitted_identity=identity, seed=seed, epochs=24, images=len(images),
        positive_images=int(present.sum()), learning_rate=2e-4, training_trials=[r["name"] for r in training])


def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    motion, observed = fit_terminal(MOTION[0])
    save(STUDY / "stream-fit.json", dict(passed=motion["passed"], cases={MOTION[0]: motion}, device=torch.cuda.get_device_name()))
    save(STUDY / "spill-trace.json", dict(cases={MOTION[0]: observed}))
    rows = [*neck_rows(STUDY), *current_startups()]
    neck = {}
    for key in NECK:
        relevant = [r for r in rows if r["name"].split("_test_")[1][:2] == key[:2]]
        proposal, training = fit_proposal(key, relevant) if key == "10_normal" else (None, None)
        neck[key] = fit_neck(key, relevant, PARENT, STUDY, ("r19", "r20"), proposal)
        if training:
            neck[key]["proposal_training"] = training
    save(STUDY / "neck-replay.json", dict(passed=all(r["passed"] for r in neck.values()), cases=neck,
            startup_sha256={p.name: sha(p) for p in (STUDY / "inputs").glob("*.h5")}))
    assert motion["passed"] and all(r["passed"] for r in neck.values())


if __name__ == "__main__":
    main()
