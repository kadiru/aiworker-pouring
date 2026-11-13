"""Check outline localization on archived native images and erased-cup controls."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "wide-mixed-r3"
sys.path[:0] = [str(STUDY/"src"), "/workspace"]
from pour_guard_vision import plane_point, project
from pour_size_autonomy import SizedCupPerception, sha
from validate_pour_recording import decode, h5py


def read(path):
    return json.loads(path.read_text())


def archives():
    base = Path("/workspace/outputs")
    for row in read(ROOT/"followup-fresh-raw.json")["rows"]:
        paths = list((base/"fountain").glob(f"*/*-{row['name']}-*/episode.hdf5"))
        assert len(paths) == 1
        yield paths[0], row["recording_sha256"], "prior_perception"
    for study,part,category in (("layout-repair-r3","test","prior_autonomy"),
                                ("wide-mixed-r2","development","wide_development")):
        for row in read(ROOT/study/f"{part}-results.json")["cases"]:
            relative = Path(row["path"]).relative_to("/home/kadir/aiworker-pouring-4070")
            yield Path("/workspace")/relative/"episode.hdf5", row["recording_sha256"], category


def measure(path, expected_sha, category, config, device):
    spec = read(path.parent/"parameters.json")
    assert sha(path) == expected_sha
    geometry = spec["declared_cup_geometry"]
    base = SizedCupPerception(config, geometry, device)
    revised = SizedCupPerception(dict(config,outline_cup_localization=True), geometry, device)
    missing = SizedCupPerception(dict(config,outline_cup_localization=True), geometry, device)
    cal = config["tracker"]["calibration"]
    truth = np.asarray(spec["cup_world"])
    rim_z = cal["table_z"]+geometry["rim_z_m"]
    x,y = np.rint(project(np.r_[truth[:2],rim_z],cal)).astype(int)
    erased_accepts = 0
    with h5py.File(path) as file:
        frames = file["data/demo_0/obs/cup_detail_image"]
        for i in range(min(15,len(frames))):
            image = decode(frames[i],spec["image_encoding"])
            base.track(image)
            revised.track(image)
            blank = image.copy()
            blank[max(0,y-75):y+160,max(0,x-140):x+140] = 180
            erased_accepts += int(missing.track(blank)["valid"])
    errors = {}
    for key, perception in (("baseline",base),("outline",revised)):
        uv = perception.tracker.anchor
        errors[key] = float(np.linalg.norm(plane_point(uv,rim_z,cal)[:2]-truth[:2])*1000) if uv is not None else None
    return dict(name=spec["name"], archive=category, recording_sha256=expected_sha,
                size=[spec["cup_height_scale"],spec["cup_diameter_scale"]], swapped=spec.get("swapped",False),
                error_mm=errors, missing_cup_acceptances=erased_accepts)


def main():
    torch.set_num_threads(1)
    scene = read(ROOT/"wide-mixed-r2/plan.json")["development"][0]
    saved = torch.load(scene["checkpoint"],map_location="cpu",weights_only=False)
    config = saved["sized_cup_perception"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cases=[]
    for archive in archives():
        row=measure(*archive, config, device)
        cases.append(row)
        print(json.dumps(row),flush=True)
    assert len(cases)==28 and len({tuple(r["size"]) for r in cases})==9
    assert all(r["missing_cup_acceptances"]==0 and r["error_mm"]["outline"] is not None
               and r["error_mm"]["outline"]<=2.5 for r in cases)
    with (STUDY/"cup-outline-replay.json").open("x") as file:
        json.dump(dict(cases=cases,max_outline_error_mm=max(r["error_mm"]["outline"] for r in cases),
                       missing_cup_acceptances=0, device=device),file,indent=2,allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
