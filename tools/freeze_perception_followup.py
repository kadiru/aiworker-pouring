"""Authorize the fresh cases only after recorded development and geometry pass."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def freeze():
    protocol_name = "followup-protocol.json"
    protocol = read(protocol_name)
    original = read("protocol.json")
    assert protocol["previous_protocol_sha256"] == sha(ROOT/"protocol.json")
    assert protocol["gates"] == original["gates"]
    names = {s["name"] for s in protocol["holdout"]}
    assert len(names) == 18 and names.isdisjoint(s["name"] for s in original["holdout"])
    development = read("followup-development.json")
    assert development["passed"] and development["passed_cases"] == development["total_cases"] == 297
    assert development["missing_color_rejected"]
    geometry = read("followup-protocol-holdout-geometry.json")
    assert geometry["complete"] and geometry["passed"] and all(r["passed"] for r in geometry["cases"])
    assert names == {r["name"] for r in geometry["cases"]}
    assert geometry["protocol_sha256"] == sha(ROOT / protocol_name)
    frozen = development["frozen_files"]
    assert all(sha(ROOT/n) == h for n,h in frozen.items())
    assert frozen[protocol_name] == sha(ROOT/protocol_name)
    fit = read("followup-presence-fit.json")
    assert fit["checkpoint_sha256"] == frozen["followup-tracker.pt"]
    assert fit["protocol_sha256"] == frozen[protocol_name]
    assert fit["helper_sha256"] == frozen["fit_followup_presence.py"]
    for n in ("followup-development.json", "followup-presence-fit.json",
              "followup-protocol-holdout-geometry.json", "followup-development-raw.json",
              "dynamic_cup_tracker.py", "ensemble_level_model.py", "evaluate_landmark_holdout.py",
              "watch_dynamic_stage.py", "check_dynamic_stage.py", "run_followup_collection.py",
              "freeze_perception_followup.py"):
        frozen[n] = sha(ROOT/n)
    outputs = ROOT.parents[1]/"outputs/fountain"
    assert not any(json.loads(p.read_text())["name"] in names for p in outputs.glob("*/*SD_A_*/parameters.json"))
    selection = dict(all_required_development_gates_passed=True, holdout_names=sorted(names),
                     protocol_sha256=sha(ROOT/protocol_name), frozen_files=frozen,
                     locator="followup-tracker.pt", filters=development["filters"],
                     autonomy_accepted=False)
    with (ROOT/"followup-selection.json").open("x") as f:
        json.dump(selection,f,indent=2)
        f.write("\n")
    print("Frozen", len(names), "fresh fixtures", flush=True)


if __name__ == "__main__":
    freeze()
