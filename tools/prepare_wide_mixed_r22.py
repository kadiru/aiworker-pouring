"""Seal a targeted cup-localization pour and a fresh 15-pair benchmark after replay."""
import shutil

from diagnose_r22_startup import PARENT, STUDY, read, save, sha
from prepare_wide_mixed_r8 import suffix
from prepare_wide_mixed_r19 import seal_layouts


def main():
    plan_path = PARENT / "plan.json"
    plan = read(plan_path)
    previous = read(PARENT / "completion.json")
    assert previous["plan_sha256"] == sha(plan_path)
    failed, = [r for r in previous["test_cases"] if not r["strict_pass"]]
    assert suffix(failed) == "12_normal" and failed["stop_reason"] == "localization_unconfirmed"
    assert failed["delivered_to_cup_ml"] == 0 and previous["test"]["lost_particles"] == 0
    replay_path = STUDY / "cup-startup-replay.json"
    replay = read(replay_path)
    assert replay["passed"] and replay["parent_plan_sha256"] == sha(plan_path)
    assert replay["frozen_checkpoints"] == plan["checkpoints"]
    assert all(sha(p) == h for p, h in plan["checkpoints"].items())
    original = read(PARENT / "geometry.json")["sources"]
    sources = {p.name: sha(p) for p in (STUDY / "src").glob("*.py")}
    assert original.keys() == sources.keys()
    assert [k for k in original if original[k] != sources[k]] == ["pour_cup_outline.py"]
    assert sources["pour_cup_outline.py"] == replay["locator_sha256"]
    parents = {suffix(s): s for s in plan["test"]}
    seal_layouts(plan, parents, {}, ("12_normal",), "SA_A_wide_mixed_r22", 2611224100, 2611224700, 2,
                 "Alternate startup cup confirmation only; revision 21 perception weights and pour control frozen")
    for p in (plan_path, PARENT/"completion.json", replay_path, STUDY/"startup-diagnosis.json"):
        plan["prerequisite_reports"][str(p)] = sha(p)
    for name in ("calibration.json", "empty-replay.json"):
        shutil.copy2(PARENT/name, STUDY/name)
    save(STUDY/"plan.json", plan)

if __name__ == "__main__":
    main()
