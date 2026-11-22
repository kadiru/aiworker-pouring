"""Replay measured cup startups and negative controls using immutable r21 checkpoints."""
import importlib.util
import sys

import torch

from diagnose_r22_startup import PARENT, ROOT, STUDY, native_trials, read, save, sha
from replay_r16_startup import measure
from diagnose_r16_startup import trials as archived_trials

sys.path.insert(0, str(STUDY / "src"))


def controls():
    # Verify every most-recent layout, other normal/swapped 12 placements,
    # and the original normal 02 failure from revision 15.
    yield from native_trials(PARENT)
    for revision in (11, 12, 20):
        yield from native_trials(ROOT / f"wide-mixed-r{revision}", {12})
    for case, frame in archived_trials():
        if case["name"].endswith("r15_test_02_normal"):
            yield case, frame
            break


def main():
    assert torch.cuda.device_count() == 1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    before = read(PARENT / "geometry.json")["sources"]
    after = {p.name: sha(p) for p in (STUDY / "src").glob("*.py")}
    assert before.keys() == after.keys()
    changed = [k for k in before if before[k] != after[k]]
    assert changed == ["pour_cup_outline.py"]
    plan = read(PARENT / "plan.json")
    assert all(sha(p) == digest for p, digest in plan["checkpoints"].items())
    loader = importlib.util.spec_from_file_location("parent_outline", PARENT/"src/pour_cup_outline.py")
    parent = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(parent)
    rows = []
    for case, frames in controls():
        row = measure(case, frames, parent.OutlineCupTracker, STUDY, "12_normal", 41)
        rows.append(row)
        print(row["name"],row["candidate"],flush=True)
    positives = [r[k] for r in rows for k in ("candidate","translated_left","translated_right") if k in r]
    missing = sum(r[k] is not None for r in rows for k in ("flat","background"))
    target, = [r for r in rows if r["name"].endswith("r21_test_12_normal")]
    passed = (len(rows) == 37 and target["parent"] is None and missing == 0 and
              all(v is not None and v["error_mm"] <= 2.5 for v in positives))
    result = dict(passed=passed, device=torch.cuda.get_device_name(), parent_plan_sha256=sha(PARENT/"plan.json"),
                  locator_sha256=after[changed[0]], changed_runtime_sources=changed,
                  unchanged_runtime_sources=len(before)-1, frozen_checkpoints=plan["checkpoints"],
                  positive_sequences=len(positives), negative_sequences=2*len(rows), false_acceptances=missing,
                  changed_previously_confirmed=[r["name"] for r in rows if r["parent"] and r["candidate"]!=r["parent"]],
                  cases=rows)
    save(STUDY/"cup-startup-replay.json",result)
    print("replay", passed, missing,flush=True)
    assert passed


if __name__ == "__main__":
    main()
