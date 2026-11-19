"""Extract byte-identical startup camera observations from the verified Mac archives."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PARENT = "wide-mixed-r15"


def main():
    report = json.loads((PROJECT / "evidence/bottle-size-A" / (PARENT+"-completion.json")).read_text())
    dest = PROJECT / "data/wide-mixed-r16"
    dest.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as work, h5py.File(dest/"startup.h5.partial","w") as out:
        raw = Path(work)/"episode.hdf5"
        for case in report["test_cases"]:
            archive = PROJECT/"data"/PARENT/(case["name"]+".hdf5.zst")
            subprocess.run(["zstd","-qd","-f","-o",str(raw),"--",str(archive)],check=True)
            with raw.open("rb") as f:
                assert hashlib.file_digest(f,"sha256").hexdigest() == case["recording_sha256"]
            with h5py.File(raw) as file:
                assert file.attrs["status"] == "complete"
                obs=file["data/demo_0/obs/cup_detail_image"]
                group=out.create_group(case["name"])
                group.attrs["recording_sha256"]=case["recording_sha256"]
                for i in range(42):
                    group.create_dataset(str(i),data=np.asarray(obs[i]))
    (dest/"startup.h5.partial").replace(dest/"startup.h5")


if __name__ == "__main__":
    main()
