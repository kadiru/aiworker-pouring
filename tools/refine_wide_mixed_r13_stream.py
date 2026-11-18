"""Refit normal 11 after its two-particle development rebound."""
from itertools import product
from pathlib import Path

import torch

from fit_wide_mixed_r12_stream import archived_stream, counterfactual, recorded_stream
from fit_wide_mixed_r13_stream import project
from fit_wide_mixed_r6 import folder, read, save
from prepare_wide_mixed_r8 import measure_spill
from pour_wide_motion import late_mouth_offset, tail_progress

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "wide-mixed-r13"
STUDY = ROOT / "wide-mixed-r13b"
SOURCE = "11_normal"
SPAN = (108., 128.)
FOCUS = (117., 124.)


def rebound(record, base, geometry):
    """Project the traced outgoing pair to the rim as a local risk proxy."""
    traces = record["observations"]
    first = next(frame for frame in traces if any(
        p["velocity_m_s"][2] > 0 and p["xyz_cup_frame_m"][0] > 0 for p in frame["particles"]))
    rows = []
    for particle in first["particles"]:
        x, y, z = particle["xyz_cup_frame_m"]
        vx, vy, vz = particle["velocity_m_s"]
        assert vx > .2 and vz > 0
        tilt = first["tilt_deg"]
        old_x = -late_mouth_offset(tilt, base[4], tail_setback_m=base[2], tail_window_deg=base[:2])
        old_z = base[3] * tail_progress(tilt, base[:2])
        rows.append((tilt, x, y, z, vx, vy, vz, old_x, old_z))
    return torch.tensor(rows, device="cuda", dtype=torch.float64).T, geometry["rim_z_m"], geometry["rim_radius_m"]


def rebound_at_rim(event, grid):
    rows, height, rim_radius = event
    tilt, x, y, z, vx, vy, vz, old_x, old_z = rows
    dx, dz = counterfactual(tilt, old_x, old_z, grid)
    flight = (vz + torch.sqrt((vz.square() + 2 * 9.81 * (z + dz - height)).clamp(min=0.))) / 9.81
    return torch.hypot(x + dx + vx * flight, y + vy * flight) / rim_radius


def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    development = read(PARENT / "development-results.json")["cases"]
    failed, = [case for case in development if not case["strict_pass"]]
    assert failed["name"].endswith(SOURCE) and failed["measured_loss_particles"] == 2
    target = recorded_stream(failed, SPAN, 100, True)
    geometry = read(folder(failed) / "parameters.json")["declared_cup_geometry"]
    previous = PARENT / "stream-fit.json"
    incumbent = torch.load(target[1]["checkpoint"], map_location="cpu", weights_only=False)
    base = (*incumbent["tail_window_deg"], incumbent["tail_setback_m"], incumbent["tail_drop_m"],
            incumbent["late_mouth_shift_m"])
    assert base == (108., 121., .010, .005, .008)
    trace = measure_spill(failed, {})
    event = rebound(trace, base, geometry)
    fit = [archived_stream(study, SOURCE, SPAN, 100, True) for study in ("wide-mixed-r9b", "wide-mixed-r12")]
    fit.append(target)
    holdout = [archived_stream(study, SOURCE, SPAN, 100, True)
               for study in ("wide-mixed-r8", "wide-mixed-r10b", "wide-mixed-r11")]
    options = [(108., 121., setback, drop, advance)
               for setback, drop, advance in product((.009, .010), (.005, .006, .007, .008),
                                                     (0., .001, .002, .003, .004))]
    grid = torch.tensor([base, *options], device="cuda", dtype=torch.float64)
    danger = rebound_at_rim(event, grid).max(dim=1).values
    tilt, _, _, _, _, _, _, old_x, old_z = event[0]
    shift, _ = counterfactual(tilt, old_x, old_z, grid)
    # Require lateral movement as well as projected clearance. A drop alone
    # can move the model's rebound below the rim even if the real bounce persists.
    feasible = (danger <= .95) & (shift.max(dim=1).values <= -.0055)
    costs = torch.stack([project(data, grid, FOCUS)[0] for _, _, data in fit]).mean(0)
    best = int(costs.masked_fill(~feasible, torch.inf).argmin())
    assert best > 0 and bool(feasible[best])
    audits = []
    for split, entries in (("train", fit), ("holdout", holdout)):
        for case, _, data in entries:
            scores, impacts, rim, radius = project(data, grid[[0, best]], FOCUS)
            audits.append(dict(name=case["name"], split=split, recording_sha256=case["recording_sha256"],
                               particles=len(data), objective=scores.tolist(),
                               impact_p95_mm=(1000 * torch.quantile(impacts, .95, dim=1)).tolist(),
                               predicted_beyond_rim=(rim > radius[None]).sum(dim=1).tolist()))
    validation = [row for row in audits if row["split"] == "holdout"]
    passed = all(row["predicted_beyond_rim"][1] <= row["predicted_beyond_rim"][0] for row in audits)
    passed = passed and all(row["impact_p95_mm"][1] <= .35 * geometry["rim_radius_m"] * 1000.
                            for row in validation)
    start, end, setback, drop, advance = grid[best].tolist()
    mouth = dict(tail_window_deg=[start, end], tail_setback_m=setback, tail_drop_m=drop, late_mouth_shift_m=advance)
    save(STUDY / "stream-fit.json", dict(passed=passed, device=torch.cuda.get_device_name(0),
         candidate_count=len(options), checkpoint=target[1]["checkpoint"], source=SOURCE, mouth_options=mouth,
         initial_fit=str(previous), rebound_rim_ratio=danger[[0,best]].tolist(),
         rebound_horizontal_shift_mm=(shift[best]*1000.).tolist(), cases=audits))
    save(STUDY / "spill-trace.json", dict(cases=[dict(trace, mouth_options=mouth)]))
    print(mouth, danger[[0,best]].tolist(), passed, flush=True)
    assert passed


if __name__ == "__main__":
    main()
