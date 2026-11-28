"""Refit the three failed bottle startups and replay both orientations for each detector."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
STUDY = ROOT/"wide-mixed-r28"
PARENT = ROOT/"wide-mixed-r27b"
sys.path.insert(0,str(STUDY/"src"))
from pour_size_autonomy import sha
from pour_wide_motion import MAX_TAIL_DROP_M
from fit_wide_mixed_r21 import fit_proposal
from fit_wide_mixed_r20 import fit_neck
from fit_wide_mixed_r19 import neck_rows
from fit_wide_mixed_r14_stream import sample
from fit_wide_mixed_r6 import folder, h5py, read, save
from extract_r19_neck import write_startup

BOTTLES = ("04_swap","07_swap","12_normal")
CUP = "11_swap"
HISTORY = (11,12,20,21,22,23,24,25,26,"27b")
FITTING = ("r26","r27b")


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    archive = STUDY/"inputs"/"neck-startup.h5"
    with h5py.File(archive,"x") as h:
        for revision in HISTORY:
            for key in BOTTLES:
                for side in ("normal","swap"):
                    case = sample(f"wide-mixed-r{revision}",key[:2]+"_"+side)
                    path = folder(case)/"episode.hdf5"
                    assert sha(path)==case["recording_sha256"]
                    write_startup(h,path,case)
    cases = {}
    for key in BOTTLES:
        rows = [r for r in neck_rows(STUDY) if r["name"].split("_test_")[1][:2]==key[:2]]
        assert len(rows)==2*len(HISTORY)
        proposal,training = fit_proposal(key,rows,parent=PARENT,revisions=FITTING,seed=26112830+int(key[:2]))
        verified = fit_neck(key,rows,PARENT,STUDY,FITTING,proposal,update_primary=True)
        cases[key] = dict(verified,training=training)
    passed = all(c["passed"] for c in cases.values())
    save(STUDY/"neck-replay.json",dict(passed=passed,cases=cases,archive_sha256=sha(archive),
         parent_plan_sha256=sha(PARENT/"plan.json")))
    assert passed


if __name__=="__main__":
    main()
