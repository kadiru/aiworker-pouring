"""Run from the Mac project checkout after r15 development. Keep durable recordings on the Mac."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from compress_sized_recordings import archive_identity, sha

PROJECT = Path(__file__).resolve().parents[1]
REMOTE = "/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1"
STUDY = "wide-mixed-r15"
HOST = "kfuUbuntu"
DATA = PROJECT / "data" / STUDY
STAGING = "/dev/shm/aiworker-pouring-r15"


def remote(*args, capture=False):
    command = "cd " + shlex.quote(REMOTE) + " && " + shlex.join(list(args))
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, command], check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def report(filename):
    return json.loads(remote("cat", STUDY + "/" + filename, capture=True))


def backup(part, arguments):
    command = ["nice", "-n", "15", "taskset", "-c", "23", "python3", "compress_sized_recordings.py",
               "--study", STUDY, *arguments]
    remote(*command)
    manifest = report(part + "-recording-archives.json")
    assert manifest["complete"]
    checked = []
    for record in manifest["records"]:
        archive = Path(record["archive"])
        assert str(archive.parent) == STAGING and archive.name.startswith("SA_A_" + STUDY.replace("-", "_") + "_")
        dest = DATA / archive.name
        if not dest.exists():
            temporary = dest.with_suffix(".partial")
            subprocess.run(["scp", "-q", HOST + ":" + str(archive), str(temporary)], check=True)
            assert sha(temporary) == record["archive_sha256"]
            assert archive_identity(temporary) == (record["original_bytes"], record["original_sha256"])
            temporary.replace(dest)
        assert dest.stat().st_size == record["archive_bytes"] and sha(dest) == record["archive_sha256"]
        assert archive_identity(dest) == (record["original_bytes"], record["original_sha256"])
        name = dest.name.removesuffix(".hdf5.zst")
        checked.append(dict(name=name, durable_mac_archive=str(dest),
                            **{k: record[k] for k in ("original_bytes", "original_sha256", "archive_bytes", "archive_sha256")}))
    receipt = DATA / (part + "-mac-backup.json")
    receipt.write_text(json.dumps(dict(report_sha256=manifest["report_sha256"], records=checked), indent=2)+"\n")
    subprocess.run(["scp", "-q", str(receipt), HOST + ":" + REMOTE + "/" + STUDY + "/" + receipt.name], check=True)
    remote(*command, "--release-verified-originals")
    final = report(part + "-recording-archives.json")
    assert final["complete"] and all(r["original_released"] and r["durable_mac_archive"] for r in final["records"])
    print(part, "durably verified", len(checked), flush=True)


def release_staging():
    closure = report("completion.json")
    for batch in [closure["development_runtime"], *closure["test_shards"]]:
        manifest = batch["recording_archives"]
        name = Path(manifest["manifest"]).name
        raw = remote("cat", STUDY + "/" + name, capture=True)
        assert hashlib.sha256(raw.encode()).hexdigest() == manifest["sha256"]
        records = json.loads(raw)["records"]
        for item in records:
            durable = Path(item["durable_mac_archive"])
            assert durable.parent == DATA and sha(durable) == item["archive_sha256"]
            assert archive_identity(durable) == (item["original_bytes"], item["original_sha256"])
            # Evict only the verified RAM copy of this experiment's recording.
            program = """from pathlib import Path
import sys
from close_wide_mixed_autonomy import sha
p = Path(sys.argv[1])
study = sys.argv[3]
assert study in ('wide-mixed-r15', 'wide-mixed-r16')
assert p.parent == Path('/dev/shm/aiworker-pouring-' + study.rsplit('-', 1)[1])
assert p.name.startswith('SA_A_' + study.replace('-', '_') + '_')
assert p.resolve() == p and p.name.endswith('.hdf5.zst')
if p.exists():
    assert sha(p) == sys.argv[2]
    p.unlink()
"""
            remote("python3", "-c", program, item["archive"], item["archive_sha256"], STUDY)


def main():
    global STUDY, DATA, STAGING
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("wide-mixed-r15", "wide-mixed-r16"), default=STUDY)
    STUDY = parser.parse_args().study
    revision = STUDY.rsplit("-", 1)[1]
    DATA = PROJECT / "data" / STUDY
    STAGING = "/dev/shm/aiworker-pouring-" + revision
    DATA.mkdir(parents=True, exist_ok=True)
    dev = report("development-results.json")
    assert dev["planned"] == dev["completed"] and all(r["strict_pass"] for r in dev["cases"])
    backup("development", ["--part", "development"])
    plan = report("plan.json")
    for i, names in enumerate(plan["test_shards"], 1):
        part = f"test-{i:02d}"
        exists = remote("sh", "-c", 'test -f "$1" && echo yes || echo no', "check",
                        STUDY + "/" + part + "-results.json", capture=True).strip()
        if exists == "no":
            remote("python3", "run_sized_autonomy.py", "--study", STUDY, "--part", "test", "--shard", str(i),
                   "--label", f"wide-{revision}-test-{i}", "--seconds", "7200")
        batch = report(part + "-results.json")
        assert [row["name"] for row in batch["cases"]] == names
        backup(part, ["--part", "test", "--shard", str(i)])
    remote("python3", "close_wide_mixed_autonomy.py", "--study", STUDY)
    remote("python3", "audit_wide_mixed_failures.py", STUDY)
    release_staging()


if __name__ == "__main__":
    main()
