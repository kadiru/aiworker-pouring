"""Explicitly replace one owned container, then run the guarded launcher."""
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--expected-id", required=True)
parser.add_argument("--profile", choices=("baseline", "quest", "fountain"), required=True)
parser.add_argument("--fluid", action="store_true")
args = parser.parse_args()
info = json.loads(subprocess.check_output(["docker", "inspect", "aiworker_pouring_4070"]))[0]
if info["Id"] != args.expected_id or info["Config"]["Labels"].get("owner") != "aiworker-pouring":
    raise SystemExit("Container ownership or identity changed; refusing replacement.")
if info["State"]["Running"]:
    subprocess.run(["docker", "stop", "--timeout", "15", info["Id"]], check=True, timeout=30)
subprocess.run(["docker", "rm", info["Id"]], check=True, timeout=15)
subprocess.run(["python3", str(Path(__file__).with_name("launch_container.py")), "--profile", args.profile] + (["--fluid"] if args.fluid else []), check=True)
