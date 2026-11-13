"""Fit bounded cup calibration and seal the fresh mixed-layout study."""
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "wide-mixed-r7"
PARENT = ROOT / "wide-mixed-r6"
OLDER = ROOT / "wide-mixed-r5"
sys.path.insert(0, str(STUDY / "src"))
from pour_size_autonomy import (MAX_SETTLED_RGB_BIAS, calibrated_rgb_height,
                                settled_pair, settled_rgb_weight, sha, validate_rgb_calibration)
from fit_wide_mixed_r6 import folder, read, save

TARGETS = ("07_normal", "09_swap", "14_swap")
DEV = ("12_swap", "14_normal", *TARGETS, "09_normal", "11_normal")
WINDOWS = {
    ((.9,.9),"settling"): (.739,.787,.850,.9),
    ((.9,.9),"draining"): (.739,.787,.850,.9),
    ((1.1,1.),"draining"): (.565,.620,.675,.755),
}


def observations(case):
    path = folder(case)
    spec = read(path/"parameters.json")
    decision = read(path/"policy-decisions.json")
    frames = {round(s["t"],6):s for s in read(path/"samples.json")}
    data = []
    for f in decision:
        s = frames.get(round(f["t"],6))
        level = f.get("sized_cup_perception") or {}
        if s is not None and level.get("valid"):
            levels = level.get("guard_filtered", level["filtered"])
            phase = ("settling" if f["fill_mode"] == "settle" else "draining"
                     if f["fill_mode"] == "coarse" and f["own_pour_phase_deg"] >= 115. else None)
            data.append((phase,f["t"],levels["rgb"],levels["color"],s["cup_fill_fraction"],f["stop_latched"]))
        if f["stop_latched"]:
            break
    return spec, data


def fit_levels():
    variants = {}
    audit = {}
    for size, phase in WINDOWS:
        cohorts = []
        for index,study in enumerate((PARENT, OLDER)):
            for case in read(study/"completion.json")["test_cases"]:
                if tuple(case["cup_size"]) != size:
                    continue
                spec, data = observations(case)
                rows = [r for r in data if r[0] == phase]
                if len(rows) < 12:
                    continue
                # Interleave blocks so both splits span the observed fill
                # range. Keep the last six decision frames out of the fit.
                fit = [r for i,r in enumerate(rows[:-6]) if i%12<6]
                val = [r for i,r in enumerate(rows) if i>=len(rows)-6 or i%12>=6]
                cohorts.append((case,spec,fit,val,index))
        key = f"{size[0]:.1f}x{size[1]:.1f}-{phase}"
        knobs = WINDOWS[size,phase]
        training = []
        for case,_,rows,_,_ in cohorts:
            is_target = case["name"].endswith(TARGETS)
            weight = (3. if is_target else 1.)/len(rows)
            training.extend((rgb,color,height,weight) for _,_,rgb,color,height,_ in rows)
        rows = torch.tensor(training,dtype=torch.float64,device="cuda")
        rgb,color,height,weight = rows.T
        basis = torch.as_tensor([settled_rgb_weight(v,knobs) for v in rgb.cpu().numpy()],device="cuda")
        bias = float(torch.clamp((weight*basis*(rgb-height)).sum()/(weight*basis.square()).sum(),
                                 0., MAX_SETTLED_RGB_BIAS))
        geometry = cohorts[0][1]["declared_cup_geometry"]
        cal = dict(bias=bias,knots=list(knobs),geometry=geometry)
        validate_rgb_calibration(cal,geometry)
        variants.setdefault(size,{})[phase] = cal
        checks = []
        for case,_,_,rows,index in cohorts:
            data = np.asarray([(rgb,color,h) for _,_,rgb,color,h,_ in rows])
            rgb,color,height = data.T
            corrected = np.asarray([calibrated_rgb_height(v,cal) for v in rgb])
            target = (index == 0 and case["name"].endswith(TARGETS)
                      and case["stop_reason"] in ("bottle_empty","level_unconfirmed")
                      and (phase == "settling") == (size == (.9,.9)))
            settle_replay = None
            if target and phase == "settling":
                observed,stable,full = settled_pair(list(zip(corrected,color)),.925,.035,.012)
                settle_replay = dict(stable=stable,full=full,level=observed)
            checks.append(dict(name=case["name"],source_sha256=case["recording_sha256"],
                               target=target,frames=len(rows),
                               last_six_gap_before=float(max(abs(rgb-color)[-6:])),
                               last_six_gap_after=float(max(abs(corrected-color)[-6:])),
                               rgb_mae_before=float(abs(rgb-height).mean()),
                               rgb_mae_after=float(abs(corrected-height).mean()),
                               settle_replay=settle_replay))
        targets = [r for r in checks if r["target"]]
        assert len(targets) == (2 if phase=="settling" else 1 if size==(1.1,1.) else 0),checks
        passed = all(r["last_six_gap_after"] < .035 and r["rgb_mae_after"] < r["rgb_mae_before"]
                     and (r["settle_replay"] is None
                          or (r["settle_replay"]["stable"] and not r["settle_replay"]["full"]))
                     for r in targets)
        audit[key] = dict(passed=passed,calibration=cal,train_frames=len(training),checks=checks)
        print(key, bias, [(r["name"],r["last_six_gap_after"]) for r in targets],flush=True)
    save(STUDY/"level-fit.json",dict(passed=all(x["passed"] for x in audit.values()), fits=audit))
    assert all(x["passed"] for x in audit.values()), audit
    return variants


def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name(0)
    torch.set_num_threads(2)
    parent = PARENT/"plan.json"
    parent_result = PARENT/"completion.json"
    plan = read(parent)
    prior = read(parent_result)
    assert prior["plan_sha256"] == sha(parent) and prior["test"]["trials"] == 30
    neck_path = STUDY/"neck-fusion.json"
    fit = read(neck_path)
    weight = STUDY/"neck-heatmap.pt"
    assert fit["passed"] and sha(weight) == fit["checkpoint_sha256"]
    neck = torch.load(weight,map_location="cpu",weights_only=False)
    variants = fit_levels()
    specs = {s["name"].split("_test_")[-1]:s for s in plan["test"]}
    spill_combinations = {(tuple(c["cup_size"]),tuple(c["bottle_size"])) for c in prior["test_cases"]
                          if c["measured_loss_particles"]}
    assert len(spill_combinations) == 2
    checkpoints = plan["checkpoints"]
    plan["checkpoints"] = {}
    outputs = {}

    def updated(spec):
        path = spec["checkpoint"]
        if path in outputs:
            return outputs[path]
        assert sha(path) == checkpoints[path]
        state = torch.load(path,map_location="cpu",weights_only=False)
        cup = (spec["cup_height_scale"],spec["cup_diameter_scale"])
        bottle = (spec["bottle_height_scale"],spec["bottle_diameter_scale"])
        state["source_heatmap"] = neck
        if cup in variants:
            state["sized_cup_perception"]["rgb_height_calibration"] = variants[cup]
        if (cup,bottle) in spill_combinations:
            state.update(tail_setback_m=.009, tail_window_deg=(115.,119.), tail_drop_m=.004)
        state["model_version"] = "visual_guard_wide_mixed_r7"
        dest = STUDY/Path(path).name
        assert not dest.exists()
        torch.save(state,dest)
        outputs[path] = str(dest)
        plan["checkpoints"][str(dest)] = sha(dest)
        return str(dest)

    rng = np.random.default_rng(2611130700)
    delta = rng.uniform(.0011,.0025,(15,2,2))*rng.choice((-1.,1.),(15,2,2))
    test = []
    for i,previous in enumerate(plan["test"]):
        spec = copy.deepcopy(previous)
        a,b = delta[spec["pair_index"]][::(-1 if spec["swapped"] else 1)]
        spec["cup_world"][:2] = (np.asarray(spec["cup_world"][:2])+a).tolist()
        difference = b-a
        spec["source_xy"] = (np.asarray(spec["source_xy"])+[difference[1],-difference[0]]).tolist()
        spec.update(name=previous["name"].replace("_r6_test_","_r7_test_"),
                    seed=2611131700+i,checkpoint=updated(previous))
        test.append(spec)
    development = []
    for i,suffix in enumerate(DEV):
        original = specs[suffix]
        spec = copy.deepcopy(original)
        spec.update(name=original["name"].replace("_r6_test_","_r7_development_"),
                    study_split="development",seed=2611131800+i,checkpoint=updated(original))
        development.append(spec)
    reports = (parent,parent_result,neck_path,STUDY/"level-fit.json",PARENT/"wrist-fit.json")
    plan["prerequisite_reports"].update({str(p):sha(p) for p in reports})
    replay = dict(parent_empty_replay_sha256=sha(PARENT/"empty-replay.json"),
                  unchanged_exact_empty_fit_sha256=sha(PARENT/"wrist-fit.json"),
                  level_fit_sha256=sha(STUDY/"level-fit.json"),
                  neck_fit_sha256=sha(neck_path),
                  spill_combinations=sorted(spill_combinations),
                  tail_setback_m=.009, tail_window_deg=[115.,119.], tail_drop_m=.004)
    save(STUDY/"empty-replay.json",replay)
    plan.update(development=development,test=test,
                test_shards=[[s["name"] for s in test[i:i+10]] for i in (0,10,20)],
                controller_scope="Fine-tuned neck and calibrated RGB levels on archived mixed-size trials",
                development_empty_lag_limit_s=2.,
                development_empty_lag_cases=[s["name"] for s in development
                                             if s["expected_stop"]=="bottle_empty" and s["bottle_height_scale"]>1.],
                empty_confirmation_replay_sha256=sha(STUDY/"empty-replay.json"))
    shutil.copy2(PARENT/"calibration.json",STUDY/"calibration.json")
    save(STUDY/"plan.json",plan)
    print(dict(development=len(development),test=len(test)),flush=True)


if __name__ == "__main__":
    main()
