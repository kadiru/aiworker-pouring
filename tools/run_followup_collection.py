"""Run the normal owned simulator collection against the new frozen scene list."""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from freeze_perception_followup import ROOT, read, sha

STAGE = ROOT / "followup-collection"


def audit():
    import check_dynamic_stage
    check_dynamic_stage.ROOT = STAGE
    sys.argv = [check_dynamic_stage.__file__, "--part", "holdout"]
    check_dynamic_stage.main()
    source = STAGE/"holdout-r0-audit.json"
    with (ROOT/"holdout-r1-audit.json").open("xb") as f:
        f.write(source.read_bytes())


def replace(destination, source):
    temporary = destination.with_name(destination.name + ".followup-tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def collect():
    import start_owned_perception_sim as simulator
    import watch_dynamic_stage as observer

    files = {"protocol.json": "followup-protocol.json",
             "component-selection.json": "followup-selection.json",
             "holdout-geometry.json": "followup-protocol-holdout-geometry.json"}
    selection = read("followup-selection.json")
    assert selection["all_required_development_gates_passed"]
    assert all(sha(ROOT/n) == h for n,h in selection["frozen_files"].items())
    assert selection["protocol_sha256"] == sha(ROOT/files["protocol.json"])
    assert read(files["protocol.json"])["previous_protocol_sha256"] == sha(ROOT/"protocol.json")
    assert not json.loads(simulator.out("docker","inspect",simulator.CID))[0]["State"]["Running"]

    STAGE.mkdir()
    backup = STAGE/"original"
    backup.mkdir()
    prior_hashes = {}
    for name, incoming in files.items():
        shutil.copy2(ROOT/name, backup/name)
        shutil.copy2(ROOT/incoming, STAGE/name)
        prior_hashes[name] = sha(backup/name)
    for name in ("best.pt", "plan.json", "candidate.json"):
        (STAGE/name).symlink_to(Path("..")/name)

    def terminate(signum, _frame):
        raise InterruptedError(f"Collection interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    launched = False
    try:
        for name in files:
            replace(ROOT/name, STAGE/name)
        launched = True
        simulator.main()
        observer.ROOT = STAGE
        sys.argv = [observer.__file__, "--part", "holdout"]
        observer.main()
        command = ["docker","exec","-e","CUDA_VISIBLE_DEVICES=","-e","PYTHONDONTWRITEBYTECODE=1",
                   simulator.CID,"nice","-n","10","/isaac-sim/python.sh",
                   "/workspace/learning/A-scale-dynamic-20261109-v1/run_followup_collection.py","--audit"]
        subprocess.run(command, check=True, timeout=240)
    finally:
        try:
            if launched:
                state = json.loads(simulator.out("docker","inspect",simulator.CID))[0]["State"]
                if state["Running"]:
                    subprocess.run(["docker","stop","--timeout","30",simulator.CID],check=True,timeout=45)
        finally:
            for name in files:
                if sha(ROOT/name) != prior_hashes[name]:
                    assert sha(ROOT/name) == sha(STAGE/name)
                    replace(ROOT/name, backup/name)
            assert all(sha(ROOT/n) == h for n,h in prior_hashes.items())
            state = json.loads(simulator.out("docker","inspect",simulator.CID))[0]["State"]
            after = simulator.out("nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv,noheader")
            with (STAGE/"restoration.json").open("x") as f:
                json.dump(dict(restored=prior_hashes, state=state, gpu_after=after),f,indent=2)
                f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit",action="store_true")
    if parser.parse_args().audit:
        audit()
    else:
        with (ROOT/"isolated-job.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            collect()


if __name__ == "__main__":
    main()
