"""Audit the native recordings and merge the three frozen wide test shards."""
from collections import defaultdict
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess

ROOT = Path(__file__).resolve().parent
HOST = ROOT.parents[1]
STUDY = ROOT / "wide-mixed-r5"
LATENCY = ROOT / "tall-aligned-r0"
GPU = "GPU-cde23916-c5fa-f18e-a343-127f732cc2f6"
ORIGINAL = "0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host(path):
    return HOST / Path(path).relative_to("/workspace")


def out(*args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def verify(path, plan_sha, names):
    data = read(path)
    assert data["plan_sha256"] == plan_sha and data["planned"] == data["completed"] == len(names)
    assert [r["name"] for r in data["cases"]] == names
    runtime = data["runtime"]
    limits = runtime["limits"]
    assert runtime["state"]["ExitCode"] == 0 and not runtime["state"]["Running"] and runtime["stop_reason"] is None
    assert limits["DeviceRequests"][0]["DeviceIDs"] == [GPU] and limits["NetworkMode"] == "none"
    assert limits["NanoCpus"] == 4_000_000_000 and limits["CpusetCpus"] == "16-23"
    assert limits["Memory"] == limits["MemorySwap"] == 16*1024**3
    assert not any(line.startswith(GPU) for key in ("gpu_before","gpu_after") for line in runtime[key].splitlines())
    archive_manifest=path.with_name(path.name.replace("-results.json","-recording-archives.json"))
    archived={}
    if archive_manifest.exists():
        from compress_sized_recordings import verify_recording_archive
        receipt=read(archive_manifest)
        assert receipt["complete"] and receipt["report_sha256"]==sha(path)
        archived={r["source"]:r for r in receipt["records"]}
        assert len(archived)==len(data["cases"])
    for row in data["cases"]:
        recording = Path(row["path"]) / "episode.hdf5"
        if recording.is_file():
            assert sha(recording) == row["recording_sha256"] and recording.stat().st_size == row["recording_bytes"]
        else:
            assert str(recording) in archived, "Missing native recording and verified archive"
            verify_recording_archive(row,archived[str(recording)])
    result=dict(report_sha256=sha(path), container=runtime["container"],
                wall_seconds=runtime["elapsed"], gpu_before=runtime["gpu_before"],
                gpu_after=runtime["gpu_after"])
    if "storage" in runtime:
        storage=runtime["storage"]
        assert all(Path(row["path"]).is_relative_to(storage["output_root"]) for row in data["cases"])
        assert storage["free_after_bytes"] >= storage["minimum_free_bytes"]
        result["storage"]=storage
    if archived:
        result["recording_archives"]=dict(manifest=str(archive_manifest),sha256=sha(archive_manifest),recordings=len(archived))
    if "interrupted_runtime" in data:
        interrupted=data["interrupted_runtime"]
        previous=interrupted["limits"]
        assert interrupted["plan_sha256"]==plan_sha
        assert interrupted["state"]["ExitCode"] in (137,143) and not interrupted["state"]["Running"]
        assert previous["DeviceRequests"]==limits["DeviceRequests"] and previous["NetworkMode"]=="none"
        assert all(previous[k]==limits[k] for k in ("NanoCpus","CpusetCpus","Memory","MemorySwap"))
        assert not any(line.startswith(GPU) for line in interrupted["gpu_before"].splitlines())
        preserved=interrupted["completed"]
        assert preserved==names[:len(preserved)]
        assert all(Path(row["path"]).is_relative_to(interrupted["output_folder"])
                   for row in data["cases"] if row["name"] in preserved)
        result["interrupted_runtime"]=interrupted
        result["wall_seconds"]+=interrupted["wall_seconds"]
    return data["cases"],result


def totals(rows):
    return dict(trials=len(rows), strict_passes=sum(r["strict_pass"] for r in rows),
                zero_measured_loss=sum(r["measured_loss_particles"]==0 for r in rows),
                lost_particles=sum(r["measured_loss_particles"] for r in rows),
                within_particle_tolerance=sum(r["within_particle_tolerance"] for r in rows),
                verified_grasps=sum(r["grasp_verified"] for r in rows),
                ordered_returns=sum(r["ordered_return"] for r in rows))


def group_totals(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name:totals(group) for name,group in groups.items()}


def main():
    global STUDY
    parser=argparse.ArgumentParser()
    parser.add_argument("--study",choices=("wide-mixed-r5","wide-mixed-r6","wide-mixed-r7","wide-mixed-r7b",
                                          "wide-mixed-r8","wide-mixed-r9","wide-mixed-r9b","wide-mixed-r10",
                                          "wide-mixed-r10b","wide-mixed-r11","wide-mixed-r12","wide-mixed-r13",
                                          "wide-mixed-r13b", "wide-mixed-r14"),
                        default="wide-mixed-r5")
    STUDY=ROOT/parser.parse_args().study
    plan_path, geometry_path = STUDY/"plan.json", STUDY/"geometry.json"
    plan, geometry = read(plan_path), read(geometry_path)
    digest = sha(plan_path)
    assert geometry["plan_sha256"] == digest and geometry["passed"]
    assert all(sha(STUDY/"src"/name) == expected for name,expected in geometry["sources"].items())
    assert all(sha(host(path)) == expected for path,expected in plan["checkpoints"].items())
    dev, dev_runtime = verify(STUDY/"development-results.json", digest,
                              [s["name"] for s in plan["development"]])
    latency_path = LATENCY/"development-results.json"
    latency, latency_runtime = verify(latency_path, sha(LATENCY/"plan.json"),
                                      ["SA_A_tall_aligned_r0_development"])
    assert latency[0]["within_particle_tolerance"] and 0. <= latency[0]["empty_confirmation_lag_s"] <= 2.
    test, runtimes = [], []
    for i,names in enumerate(plan["test_shards"],1):
        rows, runtime = verify(STUDY/f"test-{i:02d}-results.json", digest, names)
        test.extend(rows)
        runtimes.append(runtime)
    assert [r["name"] for r in test] == [s["name"] for s in plan["test"]]
    assert len({r["name"] for r in test}) == 30
    assert sum(r["swapped"] for r in test) == 15
    candidate = read(ROOT/"candidate.json")
    assert sha(ROOT/"best.pt") == candidate["checkpoint_sha256"]
    assert all(sha(HOST/"src"/n)==d for n,d in candidate["runtime_sources"].items())
    assert all(sha(host(n))==d for n,d in candidate["protected_parent_files"].items())
    assert not json.loads(out("docker","inspect",ORIGINAL))[0]["State"]["Running"]
    statuses = out("nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv,noheader")
    tall_empty = [r for r in test if r["bottle_size"][0]>1. and r["expected_stop"]=="bottle_empty"]
    lags = [r["empty_confirmation_lag_s"] for r in tall_empty if r["empty_confirmation_lag_s"] is not None]
    report = dict(closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), plan_sha256=digest,
                  geometry_sha256=sha(geometry_path), geometry_cases=len(geometry["cases"]),
                  empty_replay_sha256=sha(STUDY/"empty-replay.json"), test=totals(test),
                  by_stop=group_totals(test,lambda r:r["expected_stop"]),
                  by_orientation=group_totals(test,lambda r:"swapped" if r["swapped"] else "normal"),
                  by_size=group_totals(test,lambda r:f'{r["cup_size"][0]:.1f} × {r["cup_size"][1]:.1f}'),
                  tall_empty=dict(trials=len(tall_empty),confirmed=len(lags),
                                  at_most_two_seconds=sum(0.<=v<=2. for v in lags),
                                  median_seconds=median(lags) if lags else None,
                                  max_seconds=max(lags,default=None)),
                  test_cases=test, test_shards=runtimes, development=totals(dev), development_runtime=dev_runtime,
                  latency_case=latency[0], latency_runtime=latency_runtime,
                  original_runtime_sources_verified=len(candidate["runtime_sources"]),
                  protected_parent_files_verified=len(candidate["protected_parent_files"]),
                  original_checkpoint_sha256=candidate["checkpoint_sha256"], gpu_at_closure=statuses)
    if STUDY.name=="wide-mixed-r6":
        report["development_cases"]=dev
        report["controller_test_log_sha256"]=sha(STUDY/"controller-tests.log")
        fit_study=ROOT/"wide-mixed-r6"
        for file in ("neck-fusion.json","wrist-fit.json"):
            assert read(fit_study/file)["passed"]
        report["neck_fit_sha256"]=sha(fit_study/"neck-fusion.json")
        report["wrist_fit_sha256"]=sha(fit_study/"wrist-fit.json")
    if STUDY.name in ("wide-mixed-r7","wide-mixed-r7b"):
        report["development_cases"]=dev
        report["controller_test_log_sha256"]=sha(STUDY/"controller-tests.log")
        for file in ("neck-fusion.json","level-fit.json"):
            path=STUDY/file
            assert read(path)["passed"] and plan["prerequisite_reports"]["/workspace/"+str(path.relative_to(HOST))]==sha(path)
            report[file.removesuffix(".json").replace("-","_")+"_sha256"]=sha(path)
        wrist=ROOT/"wide-mixed-r6"/"wrist-fit.json"
        assert read(wrist)["passed"] and plan["prerequisite_reports"]["/workspace/"+str(wrist.relative_to(HOST))]==sha(wrist)
        report["wrist_fit_sha256"]=sha(wrist)
        if STUDY.name=="wide-mixed-r7b":
            for file in ("cup-anchor-replay.json","high-fill-replay.json"):
                path=STUDY/file
                assert read(path)["passed"] and plan["prerequisite_reports"]["/workspace/"+str(path.relative_to(HOST))]==sha(path)
                report[file.removesuffix(".json").replace("-","_")+"_sha256"]=sha(path)
    if STUDY.name in ("wide-mixed-r8","wide-mixed-r9","wide-mixed-r9b","wide-mixed-r10","wide-mixed-r10b",
                      "wide-mixed-r11","wide-mixed-r12","wide-mixed-r13","wide-mixed-r13b", "wide-mixed-r14"):
        report["development_cases"]=dev
        report["controller_test_log_sha256"]=sha(STUDY/"controller-tests.log")
        if STUDY.name in ("wide-mixed-r12","wide-mixed-r13","wide-mixed-r13b", "wide-mixed-r14"):
            fit = "stream-fit.json"
            assert read(STUDY/fit)["passed"]
        else:
            fit = "near-full-fit.json" if STUDY.name in ("wide-mixed-r9","wide-mixed-r9b") else "neck-fusion.json"
        reports = (fit,"spill-trace.json")
        if STUDY.name=="wide-mixed-r9b":
            reports += ("draining-color-fit.json",)
        if STUDY.name=="wide-mixed-r11":
            reports += ("neck-cascade-replay.json",)
        for filename in reports:
            path=STUDY/filename
            digest=sha(path)
            assert plan["prerequisite_reports"]["/workspace/"+str(path.relative_to(HOST))]==digest
            report[filename.removesuffix(".json").replace("-","_")+"_sha256"]=digest
    with (STUDY/"completion.json").open("x") as file:
        json.dump(report,file,indent=2,allow_nan=False)
        file.write("\n")
    print(json.dumps({k:report[k] for k in ("test","by_stop","by_orientation","tall_empty")}),flush=True)


if __name__ == "__main__":
    main()
