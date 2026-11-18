"""Fit terminal braking and coarse-pour trajectories using archived Isaac Sim particles."""
from itertools import product
from pathlib import Path

import torch

from fit_wide_mixed_r6 import folder, read, save
from fit_wide_mixed_r12_stream import archived_stream, recorded_stream, smooth_ramp
from fit_wide_mixed_r13_stream import project
from refine_wide_mixed_r13_stream import rebound_at_rim
from pour_size_autonomy import sha
from pour_wide_motion import late_mouth_offset, tail_progress

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "wide-mixed-r13b"
STUDY = ROOT / "wide-mixed-r14"
TRAIN = ("wide-mixed-r9b", "wide-mixed-r12")
HOLDOUT = ("wide-mixed-r8", "wide-mixed-r10b", "wide-mixed-r11")


def sample(study, layout):
    return next(row for row in read(ROOT / study / "completion.json")["test_cases"]
                if row["name"].endswith(layout))


def trace(case):
    folder_path = folder(case)
    frames = read(folder_path / "samples.json")
    escaped = {p["index"]: [] for p in frames[-1]["diagnostic_outside_particles"]}
    assert len(escaped) == case["measured_loss_particles"] > 0
    for frame in frames:
        for particle in frame["diagnostic_outside_particles"]:
            key = particle["index"]
            if key in escaped and len(escaped[key]) < 5:
                assert particle["origin"] == "bottle"
                escaped[key].append(dict(time=frame["t"], tilt=frame["bottle_tilt_deg"], phase=frame["phase"],
                                         xyz=particle["xyz_cup_frame_m"], velocity=particle["velocity_m_s"]))
    return dict(name=case["name"], recording_sha256=case["recording_sha256"],
                samples_sha256=sha(folder_path / "samples.json"),
                parameters_sha256=sha(folder_path / "parameters.json"), particles=escaped)


def scores(case, dataset, split, before_after):
    return dict(name=case["name"], split=split, observations=len(dataset),
                samples_sha256=sha(folder(case) / "samples.json"),
                recording_sha256=case["recording_sha256"], objective=before_after.tolist())


def coarse(target, evidence, archive):
    layout = "01_normal"
    span, focus = (108., 124.), (112., 121.)
    training = [archived_stream(s, layout, span, 100, True) for s in TRAIN]
    training.append(recorded_stream(target, span, 100, True, verified_archive=archive))
    testing = [archived_stream(s, layout, span, 100, True) for s in HOLDOUT]
    spec = training[-1][1]
    saved = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    base = (*saved["tail_window_deg"], saved["tail_setback_m"], saved["tail_drop_m"], saved["late_mouth_shift_m"])
    geometry = spec["declared_cup_geometry"]
    events = []
    for observations in evidence["particles"].values():
        bounce = next(p for p in observations if p["velocity"][2] > .2)
        tilt = bounce["tilt"]
        old_x = -late_mouth_offset(tilt, base[4], tail_setback_m=base[2], tail_window_deg=base[:2])
        events.append([tilt, *bounce["xyz"], *bounce["velocity"], old_x, base[3]*tail_progress(tilt, base[:2])])
    assert min(row[0] for row in events) < 115. and max(row[0] for row in events) > 119.
    grid = torch.tensor([base, *[(base[0], base[1], back, down, advance)
                         for back, down, advance in product((.007, .008, .009, .010),
                                                           (.005, .006, .007, .008), (0., .002, .004, .006))]],
                        device="cuda", dtype=torch.float64)
    event = (torch.tensor(events, device="cuda", dtype=torch.float64).T,
             geometry["rim_z_m"], geometry["rim_radius_m"])
    ratios = rebound_at_rim(event, grid)
    cost = torch.stack([project(d, grid, focus)[0] for _, _, d in training]).mean(0)
    # A ballistic miss outside the rim remains a risk. Optimize each recorded bounce
    # as well as the incoming stream without interpreting this proxy as a live success.
    objective = cost + .4 * (ratios - .90).clamp(min=0.).square().mean(1)
    best = int(objective.argmin())
    audits = []
    for split, batch in (("train", training), ("holdout", testing)):
        for case, _, data in batch:
            value, impacts, rim, radius = project(data, grid[[0, best]], focus)
            audits.append(dict(scores(case, data, split, value),
                               predicted_beyond_rim=(rim > radius).sum(1).tolist(),
                               impact_p95_ratio=torch.quantile(impacts / radius, .95, dim=1).tolist()))
    passed = (best > 0 and bool((ratios[best] < ratios[0]).all())
              and all(a["predicted_beyond_rim"][1] <= a["predicted_beyond_rim"][0] for a in audits)
              and all(a["impact_p95_ratio"][1] < .60 for a in audits if a["split"] == "holdout"))
    start, end, back, drop, advance = grid[best].tolist()
    return dict(passed=passed, checkpoint=spec["checkpoint"], candidates=len(grid),
                rebound_rim_ratio=ratios[[0, best]].tolist(), cases=audits,
                mouth_options=dict(tail_window_deg=[start, end], tail_setback_m=back,
                                   tail_drop_m=drop, late_mouth_shift_m=advance))


def brake_data(case):
    path = folder(case)
    spec = read(path / "parameters.json")
    geometry = spec["declared_cup_geometry"]
    rim, bottom, radius = (geometry[k] for k in ("rim_z_m", "bottom_z_m", "rim_radius_m"))
    seen, rows = set(), []
    for f in read(path / "samples.json"):
        tilt = f["bottle_tilt_deg"]
        if not 79. <= tilt <= 90. or not f["phase"].endswith(("brake", "settle")):
            continue
        surface = bottom + (rim-bottom)*f["cup_fill_fraction"]
        for p in f["diagnostic_outside_particles"]:
            idx = p["index"]
            x, y, z = p["xyz_cup_frame_m"]
            vx, vy, vz = p["velocity_m_s"]
            if (idx not in seen and p["origin"] == "bottle" and vz < -.04
                    and abs(y) < .012 and rim < z < rim+.030 and abs(x) < radius):
                seen.add(idx)
                rows.append([tilt, x, y, z, vx, vy, vz, surface, rim, radius])
    assert len(rows) >= 15, (case["name"], len(rows))
    return torch.tensor(rows, device="cuda", dtype=torch.float64)


def brake_cost(data, grid, lip):
    tilt, x, y, z, vx, vy, vz, surface, rim, radius = data.T
    rate, drop = grid.T[:, :, None]
    taper = smooth_ramp(tilt, 82., 90.)
    speed = rate + (45.-rate)*taper
    lowering = drop * smooth_ramp(tilt, 70., 80.) * (1.-smooth_ramp(tilt, 81., 84.))
    fall = (z-lowering-surface).clamp(min=0.)
    flight = (vz + (vz.square()+2*9.81*fall).sqrt())/9.81
    impacts = torch.hypot(x+vx*flight, y+vy*flight) / radius
    # Gravity and wrist tangential velocity are proxies for impact energy.
    # Penalize slower passage through the last ten degrees where fluid can still drain.
    energy = vz.square() + 2*9.81*fall + (torch.deg2rad(speed)*lip).square()
    extra_time = 10.*(1./rate-1./45.)
    objective = (energy + .10*impacts.square()).mean(1) + .11*extra_time[:, 0].square()
    return objective, impacts


def brake(target):
    layout = "00_normal"
    spec = read(folder(target) / "parameters.json")
    saved = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    assert saved["thresholds"]["brake_rate_deg_s"] == 45. and not saved.get("low_tilt_drop_m")
    grid = torch.tensor([(45., 0.), *product((24., 28., 32., 36., 40.), (.002, .003, .004, .005))],
                        device="cuda", dtype=torch.float64)
    lip = spec["declared_bottle_geometry"]["lip_height_m"]
    datasets = []
    for split, studies in (("train", (*TRAIN, PARENT.name)), ("holdout", HOLDOUT)):
        for study in studies:
            case = sample(study, layout)
            datasets.append((split, case, brake_data(case)))
    loss = torch.stack([brake_cost(d, grid, lip)[0] for split, _, d in datasets if split == "train"]).mean(0)
    best = int(loss.argmin())
    audits = []
    for split, case, data in datasets:
        obj, impacts = brake_cost(data, grid[[0, best]], lip)
        audits.append(dict(scores(case, data, split, obj), impact_p95_ratio=torch.quantile(impacts, .95, dim=1).tolist()))
    rate, drop = grid[best].tolist()
    return dict(passed=best > 0 and all(a["objective"][1] < a["objective"][0] and a["impact_p95_ratio"][1] < .85
                                        for a in audits), checkpoint=spec["checkpoint"], candidates=len(grid),
                threshold_options=dict(brake_finish_rate_deg_s=rate), mouth_options=dict(low_tilt_drop_m=drop),
                cases=audits)


def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    targets = {r["name"].rsplit("_test_", 1)[1]: r for r in read(PARENT/"completion.json")["test_cases"]
               if not r["strict_pass"]}
    assert {k:r["measured_loss_particles"] for k,r in targets.items()} == {"00_normal": 2, "01_normal": 4}
    receipt = read(PARENT/"test-01-recording-archives.json")
    assert receipt["complete"] and receipt["report_sha256"] == sha(PARENT/"test-01-results.json")
    traces = {key:trace(case) for key,case in targets.items()}
    save(STUDY/"spill-trace.json", dict(cases=traces))
    target = targets["01_normal"]
    archive = next(r for r in receipt["records"] if r["source"] == str(Path(target["path"])/"episode.hdf5"))
    fits = {"00_normal": brake(targets["00_normal"]), "01_normal": coarse(target, traces["01_normal"], archive)}
    passed = all(f["passed"] for f in fits.values())
    save(STUDY/"stream-fit.json", dict(passed=passed, device=torch.cuda.get_device_name(), cases=fits,
         model="Phase-conditioned trajectory parameter fit with ballistic and wrist-energy proxies"))
    for key, fit in fits.items():
        print(key, fit["passed"], fit["mouth_options"], fit.get("threshold_options"), flush=True)
    assert passed


if __name__ == "__main__":
    main()
