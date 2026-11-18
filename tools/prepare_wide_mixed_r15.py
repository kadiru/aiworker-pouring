"""Freeze the two fitted trajectories and the near-full confirmation threshold and 15 new matched layout pairs."""
import copy
from pathlib import Path
import shutil

import numpy as np
import torch

from fit_wide_mixed_r15 import PARENT, STUDY
from fit_wide_mixed_r6 import read, save
from pour_size_autonomy import sha
from prepare_wide_mixed_r8 import suffix
from prepare_wide_mixed_r10 import shift


def main():
    torch.set_num_threads(2)
    parent_plan, parent_completion = PARENT / "plan.json", PARENT / "completion.json"
    plan, previous = read(parent_plan), read(parent_completion)
    cases = previous["test_cases"]
    assert previous["plan_sha256"] == sha(parent_plan) and len(cases) == 30
    failed = {suffix(case): case for case in cases if not case["strict_pass"]}
    assert {k: r["measured_loss_particles"] for k, r in failed.items()} == {"00_swap": 2, "10_normal": 2, "04_normal": 0}
    assert failed["04_normal"]["stop_reason"] == "fill_unconfirmed"

    fit_path = STUDY / "stream-fit.json"
    fit = read(fit_path)
    assert fit["passed"] and fit["device"] == "NVIDIA GeForce RTX 4070" and fit["cases"].keys() == failed.keys()
    spill_path = STUDY / "spill-trace.json"
    assert set(read(spill_path)["cases"]) == {"00_swap", "10_normal"}

    parent_specs = {suffix(spec): spec for spec in plan["test"]}
    replacements = {}
    for key, result in fit["cases"].items():
        original = result["checkpoint"]
        row = parent_specs[key]
        assert original == row["checkpoint"] and original not in replacements
        assert sha(original) == plan["checkpoints"].pop(original)
        state = torch.load(original, map_location="cpu", weights_only=False)
        state.update(result["mouth_options"])
        state["thresholds"].update(result.get("threshold_options", {}))
        state["model_version"] = "visual_guard_wide_mixed_r15"
        destination = STUDY / Path(original).name
        assert not destination.exists()
        torch.save(state, destination)
        replacements[original] = str(destination)
        plan["checkpoints"][str(destination)] = sha(destination)

    def candidate(parent):
        spec = copy.deepcopy(parent)
        spec["checkpoint"] = replacements.get(parent["checkpoint"], parent["checkpoint"])
        return spec

    rng = np.random.default_rng(2611181500)
    delta = rng.uniform(.0011, .0025, (15, 2, 2)) * rng.choice((-1., 1.), (15, 2, 2))
    fresh = []
    for index, (key, parent) in enumerate(parent_specs.items()):
        spec = candidate(parent)
        offsets = delta[spec["pair_index"]]
        shift(spec, *(offsets[::-1] if spec["swapped"] else offsets))
        spec.update(name="SA_A_wide_mixed_r15_test_" + key, seed=2611185700 + index)
        fresh.append(spec)

    development = []
    for index, key in enumerate(("00_swap", "10_normal", "04_normal", "01_normal")):
        spec = candidate(parent_specs[key])
        spec.update(name="SA_A_wide_mixed_r15_development_" + key,
                    seed=2611185800 + index, study_split="development")
        development.append(spec)
    assert set(plan["checkpoints"]) == {spec["checkpoint"] for spec in fresh + development}

    largest = max(case["recording_bytes"] for case in cases + previous["development_cases"])
    plan.update(development=development, test=fresh,
                test_shards=[[s["name"] for s in fresh[i:i+3]] for i in range(0, 30, 3)],
                storage_recording_upper_bytes=max(384*1024**2, int(1.35*largest)),
                study_local_recordings=True, shared_recording_shards=list(range(1, 11)),
                development_empty_lag_cases=[s["name"] for s in development if s["expected_stop"] == "bottle_empty" and s["bottle_height_scale"] > 1.],
                development_requires_strict_zero=True,
                controller_scope="Geometry-specific coarse stream, braking taper, and a fitted final-pause confirmation threshold")
    plan["prerequisite_reports"].update({str(p): sha(p) for p in (parent_plan, parent_completion, fit_path, spill_path, STUDY / "near-full-fit.json")})
    for filename in ("calibration.json", "empty-replay.json"):
        shutil.copy2(PARENT / filename, STUDY / filename)
    plan["empty_confirmation_replay_sha256"] = sha(STUDY / "empty-replay.json")
    save(STUDY / "plan.json", plan)
    print(dict(development=len(development), fresh=len(fresh), recording_budget_bytes=plan["storage_recording_upper_bytes"]),
          flush=True)


if __name__ == "__main__":
    main()
