"""Extract losslessly recorded startup task views for the affected geometry on the Mac."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import h5py

from compress_sized_recordings import sha

PROJECT=Path(__file__).resolve().parents[1]


def write_startup(out, raw, case):
    with h5py.File(raw) as h:
        g=out.create_group(case["name"])
        g.attrs["recording_sha256"]=case["recording_sha256"]
        obs=h["data/demo_0/obs"]
        g.create_dataset("K",data=h["calibration/task/intrinsics"][:])
        g.create_dataset("T",data=obs["task_world_T_camera"][0])
        for i in range(min(42,len(obs["task_image"]))):
            g.create_dataset(str(i),data=obs["task_image"][i])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--study",choices=("wide-mixed-r19","wide-mixed-r20","wide-mixed-r21"),default="wide-mixed-r19")
    study_id=parser.parse_args().study
    latest=18 if study_id=="wide-mixed-r19" else 19
    pairs={0,12,14} if study_id=="wide-mixed-r20" else {1}
    if study_id=="wide-mixed-r21":
        pairs={1,10}
    dest=PROJECT/"data"/study_id
    dest.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as work,h5py.File(dest/"neck-startup.partial","w") as out:
        raw=Path(work)/"episode.hdf5"
        for revision in range(latest-3,latest+1):
            study=f"wide-mixed-r{revision}"
            report=PROJECT/"evidence/bottle-size-A"/(study+"-completion.json")
            cases=json.loads(report.read_text())["test_cases"]
            for case in cases:
                if case["pair_index"] not in pairs:
                    continue
                archive=PROJECT/"data"/study/(case["name"]+".hdf5.zst")
                subprocess.run(["zstd","-qd","-f","-o",str(raw),"--",str(archive)],check=True)
                assert sha(raw)==case["recording_sha256"]
                write_startup(out, raw, case)
    (dest/"neck-startup.partial").replace(dest/"neck-startup.h5")


if __name__=="__main__":
    main()
