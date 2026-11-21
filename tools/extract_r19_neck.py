"""Extract losslessly recorded startup task views for the affected geometry on the Mac."""
import json
from pathlib import Path
import subprocess
import tempfile

import h5py

from compress_sized_recordings import sha

PROJECT=Path(__file__).resolve().parents[1]


def main():
    dest=PROJECT/"data/wide-mixed-r19"
    dest.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as work,h5py.File(dest/"neck-startup.partial","w") as out:
        raw=Path(work)/"episode.hdf5"
        for study in ("wide-mixed-r15","wide-mixed-r16","wide-mixed-r17","wide-mixed-r18"):
            report=PROJECT/"evidence/bottle-size-A"/(study+"-completion.json")
            cases=json.loads(report.read_text())["test_cases"]
            for case in cases:
                if case["pair_index"]!=1:
                    continue
                archive=PROJECT/"data"/study/(case["name"]+".hdf5.zst")
                subprocess.run(["zstd","-qd","-f","-o",str(raw),"--",str(archive)],check=True)
                assert sha(raw)==case["recording_sha256"]
                with h5py.File(raw) as h:
                    g=out.create_group(case["name"])
                    g.attrs["recording_sha256"]=case["recording_sha256"]
                    obs=h["data/demo_0/obs"]
                    g.create_dataset("K",data=h["calibration/task/intrinsics"][:])
                    g.create_dataset("T",data=obs["task_world_T_camera"][0])
                    for i in range(min(42,len(obs["task_image"]))):
                        g.create_dataset(str(i),data=obs["task_image"][i])
    (dest/"neck-startup.partial").replace(dest/"neck-startup.h5")


if __name__=="__main__":
    main()
