"""Replay the missed startups and fit only the two shared bottle detectors."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "wide-mixed-r26"
STUDY = ROOT / "wide-mixed-r27"
sys.path.insert(0,str(STUDY/"src"))
from pour_wide_motion import MAX_TAIL_DROP_M
from fit_wide_mixed_r21 import fit_proposal
from fit_wide_mixed_r20 import fit_neck, state_for
from fit_wide_mixed_r19 import neck_rows, replay
from fit_wide_mixed_r14_stream import sample
from fit_wide_mixed_r6 import h5py, folder, save
from extract_r19_neck import write_startup
from pour_size_autonomy import sha

TARGETS = ("04_swap","14_normal")


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    archive = STUDY/"inputs"/"recent-startups.h5"
    with h5py.File(archive,"x") as h:
        for revision in (11,12,20,21,22,23,24,25,26):
            for index in (0,4,14):
                for side in ("normal","swap"):
                    row = sample(f"wide-mixed-r{revision}",f"{index:02}_{side}")
                    path = folder(row)/"episode.hdf5"
                    assert sha(path)==row["recording_sha256"]
                    write_startup(h,path,row)
    cases = {}
    for key in TARGETS:
        pairs = ("04",) if key=="04_swap" else ("00","14")
        rows = [r for r in neck_rows(STUDY,archive.name) if r["name"].split("_test_")[1][:2] in pairs]
        if key=="14_normal":
            rows.extend(r for r in neck_rows(ROOT/"wide-mixed-r20") if r["name"].split("_test_")[1][:2] in pairs)
        relevant = [r for r in rows if r["name"].split("_test_")[1][:2]==key[:2]]
        revisions = ("r25","r26")
        proposal,training = fit_proposal(key,relevant,parent=PARENT,revisions=revisions,seed=26112700+int(key[:2]))
        result = fit_neck(key,rows,PARENT,STUDY,revisions,proposal,update_primary=True)
        cases[key] = dict(result,proposal_training=training)
    passed = all(c["passed"] for c in cases.values())
    save(STUDY/"neck-replay.json",dict(passed=passed,cases=cases,archive_sha256=sha(archive)))
    assert passed


if __name__=="__main__":
    main()
