"""Losslessly archive this repair's closed recordings and verify original hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil

from close_wide_mixed_autonomy import ROOT, HOST, ORIGINAL, out, read, sha


def archive_identity(path):
    process = subprocess.Popen(["zstd", "-q", "-d", "-c", "--", str(path)], stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    size = 0
    try:
        for block in iter(lambda: process.stdout.read(4*1024**2), b""):
            size += len(block)
            digest.update(block)
        process.stdout.close()
        assert process.wait() == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return size, digest.hexdigest()


def verify_recording_archive(row, record):
    original = Path(row["path"]) / "episode.hdf5"
    assert record["source"] == str(original)
    assert (record["original_bytes"], record["original_sha256"]) == (row["recording_bytes"], row["recording_sha256"])
    archive = Path(record["archive"])
    external = Path("/data/aiworker-pouring/recording-archives-r14")
    dedicated = (archive.parent == external and archive.name == row["name"] + ".hdf5.zst"
                 and row["name"].startswith("SA_A_wide_mixed_r14_"))
    staged = (archive.parent == Path("/dev/shm/aiworker-pouring-r15") and
              row["name"].startswith("SA_A_wide_mixed_r15_") and archive.name == row["name"] + ".hdf5.zst")
    assert (archive.is_relative_to(HOST) or dedicated or staged) and archive.resolve() == archive and archive.is_file()
    assert sha(archive) == record["archive_sha256"]
    assert archive_identity(archive) == (row["recording_bytes"], row["recording_sha256"])


def save(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("wide-mixed-r13", "wide-mixed-r13b", "wide-mixed-r14", "wide-mixed-r15"), required=True)
    parser.add_argument("--part", choices=("development", "test"), required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--release-verified-originals", action="store_true")
    args = parser.parse_args()
    assert (args.part == "test") == (args.shard is not None)
    study = ROOT / args.study
    part = "development" if args.part == "development" else f"test-{args.shard:02d}"
    report_path = study / (part + "-results.json")
    report = read(report_path)
    runtime = report["runtime"]
    assert report["planned"] == report["completed"] == len(report["cases"])
    assert runtime["stop_reason"] is None and runtime["state"]["ExitCode"] == 0
    container = json.loads(out("docker", "inspect", runtime["container"]))[0]
    assert not container["State"]["Running"] and container["Config"]["Labels"]["owner"] == "aiworker-pouring"
    assert all(row["name"].startswith("SA_A_" + args.study.replace("-", "_") + "_")
               for row in report["cases"])
    image = json.loads(out("docker", "inspect", ORIGINAL))[0]["Image"]
    manifest_path = study / (part + "-recording-archives.json")
    manifest = dict(report_sha256=sha(report_path), records=[], complete=False)
    backup = {}
    if args.study == "wide-mixed-r15" and args.release_verified_originals:
        receipt = read(study / (part + "-mac-backup.json"))
        assert receipt["report_sha256"] == manifest["report_sha256"]
        backup = {item["name"]: item for item in receipt["records"]}
        assert set(backup) == {row["name"] for row in report["cases"]}
    for row in report["cases"]:
        source = Path(row["path"]) / "episode.hdf5"
        assert source.is_relative_to(HOST) and source.parent.resolve() == source.parent
        assert not source.is_symlink()
        output_root = Path(runtime["storage"]["output_root"])
        assert source.is_relative_to(output_root)
        archive_root = study / "recording-archives"
        if output_root.stat().st_dev != study.stat().st_dev:
            archive_root = HOST / "outputs/size-autonomy/recording-archives-r13b"
        if args.study == "wide-mixed-r14":
            if args.part == "development" or args.shard % 2 == 0:
                archive_root = Path("/data/aiworker-pouring/recording-archives-r14")
                assert Path("/data").is_mount() and archive_root.resolve() == archive_root
                if not (archive_root / (row["name"] + ".hdf5.zst")).exists():
                    assert shutil.disk_usage("/data").free >= 3 * 1024**3 + row["recording_bytes"]
            elif output_root.stat().st_dev != study.stat().st_dev:
                archive_root = HOST / "outputs/size-autonomy/recording-archives-r14"
        if args.study == "wide-mixed-r15":
            archive_root = Path("/dev/shm/aiworker-pouring-r15")
            assert Path("/dev/shm").is_mount() and archive_root.resolve() == archive_root
            memory_kib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                                  if line.startswith("MemAvailable:")))
            assert memory_kib * 1024 > 16 * 1024**3 + row["recording_bytes"], "Insufficient staging memory"
        archive_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        archive = archive_root / (row["name"] + ".hdf5.zst")
        if not archive.exists():
            assert source.stat().st_size == row["recording_bytes"] and sha(source) == row["recording_sha256"]
            temporary = archive.with_suffix(".partial.zst")
            assert not temporary.exists()
            subprocess.run(["nice", "-n", "19", "taskset", "-c", "23", "zstd", "-q", "-T1", "-3", "--long=27", "--check",
                            "-o", str(temporary), "--", str(source)], check=True)
            assert archive_identity(temporary) == (row["recording_bytes"], row["recording_sha256"])
            temporary.replace(archive)
        record = dict(source=str(source), original_bytes=row["recording_bytes"],
                      original_sha256=row["recording_sha256"], archive=str(archive),
                      archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
                      original_released=False)
        verify_recording_archive(row, record)
        if backup:
            copy = backup[row["name"]]
            for key in ("original_bytes", "original_sha256", "archive_bytes", "archive_sha256"):
                assert copy[key] == record[key]
            assert Path(copy["durable_mac_archive"]).name == archive.name
            record["durable_mac_archive"] = copy["durable_mac_archive"]
        manifest["records"].append(record)
        save(manifest_path, manifest)
        if args.release_verified_originals and source.exists():
            # Only this checked closed trial directory is writable in the helper.
            program = """set -eu
bytes=$(stat -c %s /recording/episode.hdf5)
test "$bytes" = "$1"
digest=$(sha256sum /recording/episode.hdf5)
test "${digest%% *}" = "$2"
rm -- /recording/episode.hdf5
"""
            subprocess.run(["docker", "run", "--rm", "--network", "none", "--cpus", "0.25",
                            "--memory", "256m", "--pids-limit", "32", "--user", "0:0",
                            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                            "-v", str(source.parent) + ":/recording:rw",
                            "--entrypoint", "/bin/sh", image, "-c", program, "archive",
                            str(row["recording_bytes"]), row["recording_sha256"]], check=True)
        record["original_released"] = not source.exists()
        save(manifest_path, manifest)
        print(json.dumps(dict(name=row["name"], original_bytes=record["original_bytes"],
                              archive_bytes=record["archive_bytes"], released=record["original_released"])), flush=True)
    manifest.update(complete=True, bytes_saved=sum(
        r["original_bytes"] - r["archive_bytes"] for r in manifest["records"] if r["original_released"]))
    save(manifest_path, manifest)


if __name__ == "__main__":
    main()
