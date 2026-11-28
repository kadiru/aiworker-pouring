"""Extract the camera evidence at the actual pair 09 stop in each simulation split."""
import json
from pathlib import Path

from close_wide_mixed_autonomy import read, sha


def main():
    study = Path(__file__).resolve().parent/"wide-mixed-r27b"
    report = read(study/"completion.json")
    model = read(study/"surface-replay.json")
    cases = []
    for part in ("development_cases","test_cases"):
        for row in report[part]:
            if not row["name"].endswith(("09_normal","09_swap")):
                continue
            path = Path(row["path"])/"policy-decisions.json"
            stop = next(frame for frame in read(path) if frame["stop_latched"])
            view = stop["sized_cup_perception"]
            lower = view.get("settled_surface_lower")
            cases.append(dict(name=row["name"], strict_pass=row["strict_pass"],
                              stop=stop["stop_latched"], pauses=stop["pause_count"],
                              rgb=view["filtered"]["rgb"], color=view["filtered"]["color"],
                              camera_agreement=view["agreement"], surface_lower=lower,
                              surface_minimum=model["minimum"], final_height=row["cup_surface_height_fraction"],
                              empty_lag_s=row.get("empty_confirmation_lag_s"),
                              spill_particles=row["measured_loss_particles"], recording_sha256=row["recording_sha256"],
                              decisions_sha256=sha(path)))
    assert len(cases)==4
    with (study/"pair-09-stops.json").open("x") as out:
        json.dump(dict(completion_sha256=sha(study/"completion.json"),cases=cases),out,indent=2,allow_nan=False)
        out.write("\n")


if __name__=="__main__":
    main()
