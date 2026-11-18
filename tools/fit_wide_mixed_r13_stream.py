"""Fit four coarse-pour mouth paths from archived droplet trajectories."""
from itertools import product
from pathlib import Path

import torch

from fit_wide_mixed_r12_stream import archived_stream, evaluate
from fit_wide_mixed_r6 import save

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "wide-mixed-r13"
TRAIN = ("wide-mixed-r7b", "wide-mixed-r9b", "wide-mixed-r12")
HOLDOUT = ("wide-mixed-r8", "wide-mixed-r10b", "wide-mixed-r11")
# Use samples around the observed spill, with enough lead-in to fit a smooth ramp.
SCOPES = {
    "05_normal": ((106., 126.), (115., 122.), (106., 109., 112.), (113., 116., 119.)),
    "08_swap": ((114., 132.), (124., 132.), (112., 115., 118., 121.), (120., 124., 127.)),
    "10_swap": ((106., 128.), (114., 121.), (106., 109., 112.), (113., 116., 119.)),
    "11_normal": ((108., 128.), (116., 124.), (108., 111., 114.), (115., 118., 121.)),
}


def project(data, grid, focus):
    # Include the mean around the rebound angle separately from the overall
    # shape of the stream so late, sparse drops influence candidate selection.
    overall = evaluate(data, grid, focus)[0]
    near_event = (data[:, 0] >= focus[0]) & (data[:, 0] <= focus[1])
    assert int(near_event.sum()) >= 8
    local, impacts, rim = evaluate(data[near_event], grid, focus)
    return .45 * overall + .55 * local, impacts, rim, data[near_event, 9]


def fit(source, scope):
    span, focus, starts, ends = scope
    training = [archived_stream(study, source, span, 100) for study in TRAIN]
    holdout = [archived_stream(study, source, span, 100) for study in HOLDOUT]
    target, spec, _ = training[-1]
    assert target["measured_loss_particles"] == 2
    incumbent = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    base = (*incumbent.get("tail_window_deg", (123.,126.)),
            incumbent.get("tail_setback_m", 0.), incumbent.get("tail_drop_m", 0.))
    options = [(start, end, setback, drop)
               for start, end, setback, drop in product(
                   starts, ends, (0., .002, .004, .006, .008, .010), (.003, .004, .005, .006))
               if end - start >= 4. and (start, end, setback, drop) != base]
    grid = torch.tensor([base, *options], device="cuda", dtype=torch.float64)
    training_loss = torch.stack([project(data, grid, focus)[0] for _, _, data in training]).mean(0)
    best = int(training_loss.argmin())
    audits = []
    for split, entries in (("train", training), ("holdout", holdout)):
        for case, _, data in entries:
            objective, impacts, rim, radius = project(data, grid[[0, best]], focus)
            before, after = objective.tolist()
            audits.append(dict(
                name=case["name"], split=split, recording_sha256=case["recording_sha256"],
                incoming_particles=len(data), focus_particles=len(radius),
                loss_before=before, loss_after=after,
                impact_p95_mm_before=float(torch.quantile(impacts[0], .95)*1000.),
                impact_p95_mm_after=float(torch.quantile(impacts[1], .95)*1000.),
                predicted_beyond_rim_before=int((rim[0] > radius).sum()),
                predicted_beyond_rim_after=int((rim[1] > radius).sum())))
    validation = [row for row in audits if row["split"] == "holdout"]
    passed = (best > 0
              and sum(row["loss_after"] for row in validation) < sum(row["loss_before"] for row in validation)
              and all(row["predicted_beyond_rim_after"] <= row["predicted_beyond_rim_before"] for row in audits))
    start, end, setback, drop = grid[best].tolist()
    return dict(passed=passed, checkpoint=spec["checkpoint"], candidate_count=len(options),
                focus_deg=focus, sample_range_deg=span,
                mouth_options=dict(tail_window_deg=[start, end], tail_setback_m=setback, tail_drop_m=drop),
                selected_training_objective=float(training_loss[best]),
                incumbent_training_objective=float(training_loss[0]), cases=audits)


def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    fits = {source: fit(source, scope) for source, scope in SCOPES.items()}
    passed = all(fit["passed"] for fit in fits.values())
    save(STUDY / "stream-fit.json",
         dict(passed=passed, model="gravity-only incoming stream projection",
              device=torch.cuda.get_device_name(0), cases=fits))
    for source, result in fits.items():
        print(source, result["passed"], result["mouth_options"], flush=True)
    assert passed


if __name__ == "__main__":
    main()
