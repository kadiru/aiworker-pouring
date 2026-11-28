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
                                          "wide-mixed-r13b", "wide-mixed-r14", "wide-mixed-r15", "wide-mixed-r16", "wide-mixed-r17", "wide-mixed-r18", "wide-mixed-r19", "wide-mixed-r20", "wide-mixed-r21", "wide-mixed-r22", "wide-mixed-r23", "wide-mixed-r24", "wide-mixed-r25", "wide-mixed-r26", "wide-mixed-r27", "wide-mixed-r27b"),
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
                      "wide-mixed-r11","wide-mixed-r12","wide-mixed-r13","wide-mixed-r13b", "wide-mixed-r14", "wide-mixed-r15"):
        report["development_cases"]=dev
        report["controller_test_log_sha256"]=sha(STUDY/"controller-tests.log")
        if STUDY.name in ("wide-mixed-r12","wide-mixed-r13","wide-mixed-r13b", "wide-mixed-r14", "wide-mixed-r15"):
            fit = "stream-fit.json"
            assert read(STUDY/fit)["passed"]
        else:
            fit = "near-full-fit.json" if STUDY.name in ("wide-mixed-r9","wide-mixed-r9b") else "neck-fusion.json"
        reports = (fit,"spill-trace.json")
        if STUDY.name == "wide-mixed-r15":
            reports += ("near-full-fit.json",)
        if STUDY.name=="wide-mixed-r9b":
            reports += ("draining-color-fit.json",)
        if STUDY.name=="wide-mixed-r11":
            reports += ("neck-cascade-replay.json",)
        for filename in reports:
            path=STUDY/filename
            digest=sha(path)
            assert plan["prerequisite_reports"]["/workspace/"+str(path.relative_to(HOST))]==digest
            report[filename.removesuffix(".json").replace("-","_")+"_sha256"]=digest
    if STUDY.name in ("wide-mixed-r16", "wide-mixed-r22"):
        replay = STUDY/"cup-startup-replay.json"
        proof = read(replay)
        assert proof["passed"] and proof["frozen_checkpoints"] == plan["checkpoints"]
        assert proof["locator_sha256"] == geometry["sources"]["pour_cup_outline.py"]
        assert proof["changed_runtime_sources"] == ["pour_cup_outline.py"]
        parent_name = "wide-mixed-r15" if STUDY.name == "wide-mixed-r16" else "wide-mixed-r21"
        parent = read(ROOT/parent_name/"geometry.json")["sources"]
        assert parent.keys() == geometry["sources"].keys()
        assert [n for n in parent if parent[n] != geometry["sources"][n]] == ["pour_cup_outline.py"]
        report.update(cup_startup_replay_sha256=sha(replay), development_cases=dev,
                      frozen_checkpoints=len(plan["checkpoints"]),
                      controller_test_log_sha256=sha(STUDY/"controller-tests.log"))
    if STUDY.name in ("wide-mixed-r17", "wide-mixed-r18", "wide-mixed-r19", "wide-mixed-r20", "wide-mixed-r21", "wide-mixed-r23", "wide-mixed-r24", "wide-mixed-r25", "wide-mixed-r26"):
        parent_name = "wide-mixed-r"+str(int(STUDY.name.rsplit("r",1)[1])-1)
        parent = read(ROOT/parent_name/"geometry.json")
        assert parent["sources"] == geometry["sources"]
        reports = ("stream-fit.json", "frozen-localization.json")
        if STUDY.name in ("wide-mixed-r19", "wide-mixed-r20", "wide-mixed-r21", "wide-mixed-r24", "wide-mixed-r25"):
            reports = ("stream-fit.json", "neck-replay.json", "frozen-controller.json")
        if STUDY.name == "wide-mixed-r23":
            reports = ("stream-fit.json", "frozen-controller.json")
        if STUDY.name == "wide-mixed-r26":
            reports = ("cup-startup-replay.json", "frozen-controller.json")
        if STUDY.name == "wide-mixed-r25":
            reports += ("cup-startup-replay.json",)
        for filename in reports:
            path = STUDY/filename
            assert read(path)["passed"]
            digest = sha(path)
            assert plan["prerequisite_reports"]["/workspace/"+str(path.relative_to(HOST))] == digest
            report[filename.split(".")[0].replace("-","_")+"_sha256"] = digest
        report.update(development_cases=dev, controller_test_log_sha256=sha(STUDY/"controller-tests.log"))
    if STUDY.name in ("wide-mixed-r27","wide-mixed-r27b"):
        surface = STUDY.name=="wide-mixed-r27b"
        parent = read(ROOT/("wide-mixed-r27" if surface else "wide-mixed-r26")/"geometry.json")["sources"]
        assert parent.keys()==geometry["sources"].keys()
        expected = {"pour_size_autonomy.py","pour_guard_actor.py"} if surface else {"pour_size_autonomy.py"}
        assert {k for k in parent if parent[k]!=geometry["sources"][k]}==expected
        reports = ("surface-fit.json","surface-replay.json") if surface else ("neck-replay.json","stop-fit.json")
        for filename in (*reports,"frozen-controller.json"):
            path = STUDY/filename
            assert read(path)["passed"] and plan["prerequisite_reports"]["/workspace/"+str(path.relative_to(HOST))]==sha(path)
            report[filename.split(".")[0].replace("-","_")+"_sha256"] = sha(path)
        assert len(dev)==4 and all(r["strict_pass"] and r["measured_loss_particles"]==0 for r in dev)
        report.update(development_cases=dev,controller_test_log_sha256=sha(STUDY/"controller-tests.log"))
    with (STUDY/"completion.json").open("x") as file:
        json.dump(report,file,indent=2,allow_nan=False)
        file.write("\n")
    print(json.dumps({k:report[k] for k in ("test","by_stop","by_orientation","tall_empty")}),flush=True)


if __name__ == "__main__":
    main()
