"""Pair recorded cup-detail frames and simulated surface heights at the same control tick."""
from fit_wide_mixed_r27_neck import PARENT, ROOT, STUDY, state_for, sample, folder
from fit_wide_mixed_r6 import read, save, h5py
from validate_pour_recording import decode
from pour_guard_vision import cup_crop
from pour_size_autonomy import sha
import numpy as np

DEST = ROOT/"wide-mixed-r27b"


def main():
    _,state=state_for("09_swap",PARENT)
    cfg=state["sized_cup_perception"]
    cal = dict(cfg["tracker"]["calibration"],cup_geometry=cfg["rgb_height_calibration"]["draining"]["geometry"])
    entries = [sample(f"wide-mixed-r{i}",key) for i in (11,12,20,21,22,23,24,25,26)
               for key in ("09_swap","14_normal","00_normal")]
    entries.extend(r for r in read(STUDY/"development-results.json")["cases"] if r["name"].endswith(("09_swap","14_normal")))
    meta,images,heights=[],[],[]
    for row in entries:
        path=folder(row)
        decision=read(path/"policy-decisions.json")
        samples=read(path/"samples.json")
        clock=np.array([s["t"] for s in samples])
        frames=[]
        for d in decision:
            if d["fill_mode"]=="settle" and d.get("sized_cup_perception",{}).get("valid"):
                frames.append(d)
            if d["stop_latched"]:break
        if not frames:continue
        with h5py.File(path/"episode.hdf5") as h:
            demo=h["data/demo_0"];times=demo["time"][:].reshape(-1)
            for n in sorted({f["pause_count"] for f in frames}):
                pause=[d for d in frames if d["pause_count"]==n]
                if len(pause)<6:continue
                for f in pause[-8:]:
                    j=int(np.abs(times-f["t"]).argmin());i=int(np.abs(clock-f["t"]).argmin())
                    assert abs(times[j]-f["t"])<.026 and abs(clock[i]-f["t"])<.026,(times[j],clock[i],f["t"])
                    image=decode(demo["obs/cup_detail_image"][j],"jpeg95")
                    crop=cup_crop(image,f["cup_xy"],cal)
                    v=f["sized_cup_perception"]["filtered"]
                    meta.append(dict(name=row["name"],pause=n,t=f["t"],rgb=v["rgb"],color=v["color"],
                                     recording_sha256=row["recording_sha256"]))
                    images.append(crop);heights.append(samples[i]["cup_fill_fraction"])
    with (DEST/"settled-crops.npz").open("xb") as f:
        np.savez_compressed(f,images=np.stack(images),heights=np.array(heights))
    save(DEST/"settled-crops.json",meta)
    print(len(meta),"aligned settling views",flush=True)


if __name__=="__main__":
    main()
