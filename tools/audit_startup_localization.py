"""Measure startup cup confirmation independently of grasp and pouring outcomes."""
import argparse
import hashlib
import json
from math import dist
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurement(case):
    path = Path(case["path"]) / "policy-decisions.json"
    rows = read(path)
    first = next((row for row in rows if (row["localization_consensus"] or {}).get("cup")
                  and (row["localization_consensus"] or {}).get("source") and row["cup_xy"] is not None), None)
    record = dict(name=case["name"], recording_sha256=case["recording_sha256"],
                  decisions_sha256=digest(path), both_confirmed=first is not None)
    if first is not None:
        xy = first["cup_xy"][:2]
        record.update(localization_time_s=first["t"], consensus_elapsed_s=first["localization_consensus"]["elapsed"],
                      cup_xy=xy, declared_fixture_error_mm=1000.*dist(xy,case["cup_world"][:2]))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    study = parser.parse_args().study
    closure = study/"completion.json"
    report = read(closure)
    rows = [measurement(case) for case in report["test_cases"]]
    confirmed = [r for r in rows if r["both_confirmed"]]
    summary = dict(closure_sha256=digest(closure), trials=len(rows), both_confirmed=len(confirmed),
                   maximum_declared_fixture_error_mm=max((r["declared_fixture_error_mm"] for r in confirmed), default=None),
                   development=[measurement(c) for c in report["development_cases"]], cases=rows)
    with (study/"fresh-startup.json").open("x") as output:
        json.dump(summary,output,indent=2,allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
