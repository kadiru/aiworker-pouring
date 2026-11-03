"""Yield the project container if another compute job uses the RTX 4070.

Runs on the host, bound to one immutable container ID. Never signals other jobs.
"""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

GPU = "GPU-cde23916-c5fa-f18e-a343-127f732cc2f6"

def output(*args):
    return subprocess.check_output(args, text=True, timeout=10).strip()

def log(event, **fields):
    print(json.dumps({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "event": event, **fields}), flush=True)

def main(container_id):
    info = json.loads(output("docker", "inspect", container_id))[0]
    if (info["Id"] != container_id or info["Name"] != "/aiworker_pouring_4070"
            or info["Config"]["Labels"].get("owner") != "aiworker-pouring"):
        raise SystemExit("Refusing to guard an unowned container")
    log("guard_started", container_id=container_id, gpu=GPU)
    failures = 0
    while True:
        try:
            running = output("docker", "inspect", "--format", "{{.State.Running}}", container_id)
            if running != "true":
                log("container_finished"); return
            # Query GPU first so a newly created container child cannot look foreign.
            rows = output("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader")
            own_pids = {int(s.strip()) for s in output("docker", "top", container_id, "-eo", "pid").splitlines()[1:]}
            foreign = [int(row.split(",")[1].strip()) for row in rows.splitlines()
                       if row.startswith(GPU) and int(row.split(",")[1].strip()) not in own_pids
                       and Path(f"/proc/{int(row.split(',')[1].strip())}").exists()]
            if foreign:
                log("yielding_gpu", foreign_pids=foreign)
                subprocess.run(["docker", "stop", "--timeout", "5", container_id], check=True, timeout=20)
                return
            failures = 0
        except (subprocess.SubprocessError, ValueError, IndexError) as error:
            if isinstance(error, subprocess.CalledProcessError):
                try:
                    if container_id not in output("docker", "ps", "-aq", "--no-trunc").splitlines():
                        log("container_removed"); return
                except subprocess.SubprocessError:
                    pass
            failures += 1
            log("inspection_error", error=str(error), consecutive=failures)
            if failures >= 3:
                log("yielding_after_inspection_failures")
                subprocess.run(["docker", "stop", "--timeout", "5", container_id], check=True, timeout=20)
                return
        time.sleep(2)

if __name__ == "__main__":
    main(sys.argv[1])
