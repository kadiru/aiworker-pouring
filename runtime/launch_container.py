"""Run on Ubuntu to launch only the project-owned RTX 4070 container."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/home/kadir/aiworker-pouring-4070")
GPU = "GPU-cde23916-c5fa-f18e-a343-127f732cc2f6"
NAME = "aiworker_pouring_4070"

def run(*args):
    return subprocess.check_output(args, text=True).strip()

parser = argparse.ArgumentParser()
parser.add_argument("--script", choices=("inspect_robot.py", "run_aiworker.py"), default="run_aiworker.py")
parser.add_argument("--profile", choices=("baseline", "quest", "fountain"), default="baseline")
parser.add_argument("--fluid", action="store_true", help="Enable the bounded PBD fill in the Quest profile")
args = parser.parse_args()
if NAME in run("docker", "ps", "-a", "--format", "{{.Names}}").splitlines():
    raise SystemExit(f"{NAME} already exists. Inspect it before restarting or replacing it.")
processes = run("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader")
if any(line.startswith(GPU) for line in processes.splitlines()):
    raise SystemExit("RTX 4070 has a compute workload; refusing to compete with it.")
for folder in ("assets", "src", "outputs", "logs", "cache", "kit-cache"):
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
snapshot = {
    "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "gpu_processes": processes,
    "gpu_inventory": run("nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader"),
}
(ROOT / "outputs/prelaunch.json").write_text(json.dumps(snapshot, indent=2) + "\n")
command = [
    "docker", "run", "-d", "--name", NAME,
    "--label", "owner=aiworker-pouring", "--restart", "no", "--network", "host", "--no-healthcheck",
    "--gpus", f"device={GPU}", "--user", "0:0",
    "--cpuset-cpus", "16-23", "--cpus", "4", "--cpu-shares", "256",
    "--memory", "16g", "--memory-swap", "16g", "--shm-size", "1g", "--pids-limit", "1024",
    "--log-opt", "max-size=10m", "--log-opt", "max-file=2",
    "-e", "ACCEPT_EULA=Y", "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
    "-e", "NVIDIA_VISIBLE_DEVICES=" + GPU,
    "-e", "OMP_NUM_THREADS=4", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=4",
    "-v", f"{ROOT}:/workspace:rw",
    "-v", f"{ROOT}/cache:/root/.cache:rw",
    "-v", f"{ROOT}/cache:/isaac-sim/.cache:rw",
    "-v", f"{ROOT}/kit-cache:/isaac-sim/kit/cache:rw",
    "-v", f"{ROOT}/logs:/root/.nvidia-omniverse/logs:rw",
    "--entrypoint", "/usr/bin/nice", "nvcr.io/nvidia/isaac-sim:6.0.1",
    "-n", "10", "/isaac-sim/python.sh", "-u", f"/workspace/src/{args.script}", "--allow-root",
]
if args.profile in ("quest", "fountain"):
    for folder in ("cache51", "kit-cache51", "kit-data51", "logs51", "config51", "data51"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    # The installed 5.1 runtime and its bundled Python are read-only. All writes
    # and dependencies belong to this project, including Kit and CloudXR caches.
    extra = [
        "-e", "AIWORKER_FLUID=" + ("1" if args.fluid else "0"),
        "-v", "/home/kadir/isaac-sim:/isaac-sim:ro",
        "-e", "XDG_CACHE_HOME=/workspace/cache51",
        "-v", f"{ROOT}/kit-cache51:/isaac-sim/kit/cache:rw",
        "-v", f"{ROOT}/kit-data51:/isaac-sim/kit/data:rw",
        "-e", "XDG_CONFIG_HOME=/workspace/config51",
        "-e", "XDG_DATA_HOME=/workspace/data51",
    ]
    # Remove baseline cache mounts before adding the 5.1 read-only base mount.
    for destination in ("/isaac-sim/.cache:rw", "/isaac-sim/kit/cache:rw"):
        index = next(i for i, value in enumerate(command) if value.endswith(destination))
        del command[index-1:index+1]
    index = command.index("--entrypoint")
    command[index:index] = extra
    index = command.index("-n", command.index("--entrypoint"))
    command[index:] = ["-n", "10", "/bin/bash", "/workspace/src/start_quest.sh"]
    if args.profile == "fountain":
        index_env=command.index("--entrypoint")
        command[index_env:index_env]=["-e","FOUNTAIN_RESUME=1"]
        index = command.index("-n", command.index("--entrypoint"))
        command[index:] = ["-n", "10", "/isaac-sim/python.sh", "-u", "/workspace/src/run_fountain.py", "--allow-root"]
(ROOT / "outputs/container-command.json").write_text(json.dumps(command, indent=2) + "\n")
container_id = run(*command)
with (ROOT / "outputs/gpu-priority-guard.log").open("a") as guard_log:
    subprocess.Popen([sys.executable, str(ROOT / "gpu_priority_guard.py"), container_id],
                     stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True)
print(container_id)
