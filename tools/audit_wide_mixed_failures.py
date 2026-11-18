"""Summarize each strict failure from the native sidecars after a frozen benchmark closes."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def inspect_failure(case):
    folder = Path(case["path"])
    samples = read(folder / "samples.json")
    decisions = read(folder / "policy-decisions.json")
    stop = next(row for row in decisions if row["stop_latched"])
    assert stop["stop_latched"] == case["stop_reason"]
    traces = {p["index"]: [] for p in samples[-1]["diagnostic_outside_particles"]}
    assert len(traces) == case["measured_loss_particles"]
    if traces:
        for row in samples:
            for p in row["diagnostic_outside_particles"]:
                observations = traces.get(p["index"])
                if observations is not None and len(observations) < 5:
                    observations.append(dict(t=row["t"], phase=row["phase"], tilt_deg=row["bottle_tilt_deg"],
                                             origin=p["origin"], xyz=p["xyz_cup_frame_m"], velocity=p["velocity_m_s"]))
    return dict(name=case["name"], reasons=case["failure_reasons"], stop=stop["stop_latched"],
                stop_time_s=stop["t"], pause_count=stop["pause_count"], visual_at_stop=stop["sized_cup_perception"],
                measured_final_height_fraction=samples[-1]["cup_fill_fraction"], escaped_particles=traces,
                recording_sha256=case["recording_sha256"],
                sidecar_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                                for name in ("samples.json", "policy-decisions.json", "parameters.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    closure = args.study / "completion.json"
    failures = [inspect_failure(case) for case in read(closure)["test_cases"] if not case["strict_pass"]]
    with (args.study / "fresh-failures.json").open("x") as output:
        json.dump(dict(completion_sha256=hashlib.sha256(closure.read_bytes()).hexdigest(), cases=failures),
                  output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
