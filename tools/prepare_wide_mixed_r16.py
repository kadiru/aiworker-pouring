"""Seal the startup locator and 15 fresh matched position pairs with unchanged checkpoints."""
import copy
from pathlib import Path
import shutil

import numpy as np
import torch

from replay_r16_startup import PARENT, STUDY
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
    failed, = [case for case in cases if not case["strict_pass"]]
    assert suffix(failed) == "02_normal" and failed["stop_reason"] == "localization_unconfirmed"
    replay_path = STUDY / "cup-startup-replay.json"
    replay = read(replay_path)
    assert replay["passed"] and replay["parent_plan_sha256"] == sha(parent_plan)
    assert replay["frozen_checkpoints"] == plan["checkpoints"]
    assert all(sha(path) == digest for path, digest in plan["checkpoints"].items())
    assert replay["locator_sha256"] == sha(STUDY / "src/pour_cup_outline.py")
    parent_specs = {suffix(spec): spec for spec in plan["test"]}

    rng = np.random.default_rng(2611191600)
    delta = rng.uniform(.0011, .0025, (15, 2, 2)) * rng.choice((-1., 1.), (15, 2, 2))
    fresh = []
    for index, (key, parent) in enumerate(parent_specs.items()):
        spec = copy.deepcopy(parent)
        offsets = delta[spec["pair_index"]]
        shift(spec, *(offsets[::-1] if spec["swapped"] else offsets))
        spec.update(name="SA_A_wide_mixed_r16_test_" + key, seed=2611196700 + index)
        fresh.append(spec)

    development = []
    for index, key in enumerate(("02_normal",)):
        spec = copy.deepcopy(parent_specs[key])
        spec.update(name="SA_A_wide_mixed_r16_development_" + key,
                    seed=2611196800 + index, study_split="development")
        development.append(spec)
    assert set(plan["checkpoints"]) == {spec["checkpoint"] for spec in fresh + development}

    largest = max(case["recording_bytes"] for case in cases + previous["development_cases"])
    # The targeted fixture is near-full. The fresh empty trials retain their latency audit.
    plan.pop("development_empty_lag_limit_s", None)
    plan.update(development=development, test=fresh,
                test_shards=[[s["name"] for s in fresh[i:i+2]] for i in range(0, 30, 2)],
                storage_recording_upper_bytes=max(384*1024**2, int(1.35*largest)),
                study_local_recordings=True, shared_recording_shards=list(range(1, 16)),
                development_empty_lag_cases=[s["name"] for s in development if s["expected_stop"] == "bottle_empty" and s["bottle_height_scale"] > 1.],
                development_requires_strict_zero=True,
                controller_scope="Revision 15 checkpoints and pouring control frozen; startup cup localization only")
    plan["prerequisite_reports"].update({str(p): sha(p) for p in (parent_plan, parent_completion, replay_path, STUDY / "startup-diagnosis.json")})
    for filename in ("calibration.json", "empty-replay.json"):
        shutil.copy2(PARENT / filename, STUDY / filename)
    plan["empty_confirmation_replay_sha256"] = sha(STUDY / "empty-replay.json")
    save(STUDY / "plan.json", plan)
    print(dict(development=len(development), fresh=len(fresh), recording_budget_bytes=plan["storage_recording_upper_bytes"]),
          flush=True)


if __name__ == "__main__":
    main()
