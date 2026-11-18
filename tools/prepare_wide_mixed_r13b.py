"""Freeze the refined normal 11 trajectory before any reserved runs."""
from pathlib import Path
import shutil

import torch

from fit_wide_mixed_r6 import read, save
from pour_size_autonomy import sha
from refine_wide_mixed_r13_stream import PARENT, SOURCE, STUDY


def main():
    torch.set_num_threads(2)
    parent_plan = PARENT / "plan.json"
    parent_development = PARENT / "development-results.json"
    plan, previous = read(parent_plan), read(parent_development)
    assert previous["plan_sha256"] == sha(parent_plan)
    assert previous["completed"] == len(plan["development"]) == 4
    failed, = [case for case in previous["cases"] if not case["strict_pass"]]
    assert failed["name"].endswith(SOURCE) and failed["failure_reasons"] == ["measured_spill"]
    assert not any(PARENT.glob("test-*-results.json"))

    fit_path = STUDY / "stream-fit.json"
    fit = read(fit_path)
    assert fit["passed"] and fit["device"] == "NVIDIA GeForce RTX 4070"
    original = fit["checkpoint"]
    assert sha(original) == plan["checkpoints"].pop(original)
    checkpoint = torch.load(original, map_location="cpu", weights_only=False)
    checkpoint.update(fit["mouth_options"], model_version="visual_guard_wide_mixed_r13b")
    dest = STUDY / Path(original).name
    assert not dest.exists()
    torch.save(checkpoint, dest)
    plan["checkpoints"][str(dest)] = sha(dest)

    def update(spec):
        revised = dict(spec, name=spec["name"].replace("_r13_", "_r13b_"))
        if spec["checkpoint"] == original:
            revised["checkpoint"] = str(dest)
        return revised

    development = [update(spec) for spec in plan["development"]]
    fresh = [update(spec) for spec in plan["test"]]
    plan.update(development=development, test=fresh,
                test_shards=[[s["name"] for s in fresh[i:i+10]] for i in (0, 10, 20)],
                shared_recording_shards=[2],
                development_empty_lag_cases=[s["name"] for s in development
                                             if s["expected_stop"] == "bottle_empty" and s["bottle_height_scale"] > 1.],
                controller_scope="Four fitted paths with additional terminal rebound clearance on normal 11")
    plan["prerequisite_reports"].update({str(p): sha(p) for p in
        (parent_plan, parent_development, fit_path, STUDY / "spill-trace.json")})
    for filename in ("calibration.json", "empty-replay.json"):
        shutil.copy2(PARENT / filename, STUDY / filename)
    assert sha(STUDY / "empty-replay.json") == plan["empty_confirmation_replay_sha256"]
    save(STUDY / "plan.json", plan)
    print(dict(development=len(development), fresh=len(fresh), shared_recording_shards=plan["shared_recording_shards"]),
          flush=True)


if __name__ == "__main__":
    main()
