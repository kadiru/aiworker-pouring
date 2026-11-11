"""Declare fresh moving scenes before fitting. Keep the original study intact."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = ROOT / "protocol.json"
    protocol = json.loads(prior.read_text())
    assert protocol["version"] == 6 and len(protocol["holdout"]) == 18
    fresh = copy.deepcopy(protocol)
    for index, scene in enumerate(fresh["holdout"]):
        size = index // 2
        direction = 1 if size % 2 == 0 else -1
        scene["name"] = scene["name"].replace("_r0_", "_r1_")
        scene["seed"] += 901 + index
        scene["cup_world"][0] += direction * (0.004 + 0.001 * (size % 3))
        scene["cup_world"][1] += direction * 0.005
        scene["source_xy"][0] += direction * 0.003
        scene["calibration_motion"]["lateral_amplitude_m"] += direction * 0.002
        scene["calibration_motion"]["maximum_angle_deg"] += direction
    fresh.update(version=7, previous_protocol_sha256=sha(prior))
    destination = ROOT / "followup-protocol.json"
    with destination.open("x") as f:
        json.dump(fresh, f, indent=2)
        f.write("\n")
    print(sha(destination), len(fresh["holdout"]), "fresh scenes")


if __name__ == "__main__":
    main()
