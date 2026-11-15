"""Launch only the owned non-networked 4070 simulator batch and observe it to completion."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
HOST = ROOT.parents[1]
OUTPUT = HOST / "outputs/size-autonomy"
ORIGINAL = "0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1"
sys.path.insert(0, str(HOST))
from gpu_priority_guard import GPU, foreign_gpu_pids

sys.path.insert(0, str(HOST/"src"))
from pour_robust_report import completed_row
from pour_wide_report import reasons, strict_success


def out(*args, timeout=20):
    return subprocess.check_output(args, text=True, timeout=timeout).strip()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native(path):
    return HOST / Path(path).relative_to("/workspace")


def sandbox(path):
    return str(Path("/workspace")/path.relative_to(HOST))


def save(path, value):
    temp = path.with_suffix(".partial.json")
    temp.write_text(json.dumps(value,indent=2)+"\n")
    temp.replace(path)


def rows_from_disk(output, cases, plan):
    rows = {}
    for path in output.glob("*/*SA_A_*/parameters.json"):
        scene = read(path)
        name = scene["name"]
        assert name in cases and name not in rows
        assert all(scene[k]==v for k,v in cases[name].items())
        folder = path.parent
        rows[name] = dict(name=name,status="incomplete")
        if not (folder/"metrics.json").exists():
            continue
        index=folder.parent/"results.json"
        if not index.exists() or sandbox(folder) not in {x["path"] for x in read(index)}:
            continue
        expected = plan["checkpoints"][scene["checkpoint"]]
        row = completed_row(folder, expected)
        samples = read(folder/"samples.json")
        decisions = read(folder/"policy-decisions.json")
        stop = next((x for x in decisions if x["stop_latched"]),None)
        perception = stop.get("sized_cup_perception") if stop else None
        peak = max(x["cup_fill_fraction"] for x in samples)
        quantum = scene["spacing"]**3*1e6
        particles = round(max(row["final_outside_ml"],row["ever_below_worktop_ml"])/quantum)
        mode_order=("Returning upright","Setting bottle down","Releasing bottle","Retreating","Holding completed pose")
        mode_times=[next((x["t"] for x in decisions if x["stop_latched"] and x["mode"]==mode),None) for mode in mode_order]
        ordered=all(t is not None for t in mode_times) and all(a<b for a,b in zip(mode_times,mode_times[1:]))
        side=next((x["side_return_plan"] for x in decisions if x.get("side_return_plan")),None)
        record=folder/"episode.hdf5"
        assert record.is_file() and not (folder/"episode.partial.hdf5").exists()
        row.update(cup_size=[scene["cup_height_scale"],scene["cup_diameter_scale"]],
                   peak_height=peak, measured_loss_particles=particles,
                   ordered_return=ordered,side_return=side,perception_at_stop=perception,
                   recording_bytes=record.stat().st_size,recording_sha256=sha(record))
        if "test_shards" in plan:
            row.update(layout_family=scene["layout_family"], pair_index=scene["pair_index"],
                       swapped=scene["swapped"], bottle_size=[scene["bottle_height_scale"],
                                                            scene["bottle_diameter_scale"]])
        if plan.get("layout_repair"):
            grasp = next((x for x in decisions if x.get("cup_xy") is not None), None)
            row["source_localization_method"] = grasp.get("source_localization_method") if grasp else None
            row["neck_fallback_frames"] = sum(x.get("source_localization_method") == "neck_fallback"
                                               for x in decisions if x.get("cup_xy") is None)
            row["empty_confirmation_method"] = stop.get("empty_confirmation_method") if stop else None
        calibrated = [x["sized_cup_perception"] for x in decisions
                      if not x["stop_latched"]
                      and (x.get("sized_cup_perception") or {}).get("rgb_settle_correction", 0.) > 0.]
        if calibrated:
            gate = plan["settled_rgb_calibration"]["agreement_limit"]
            row["settled_rgb_adjustment"] = dict(
                frames=len(calibrated),
                restored_agreement_frames=sum(
                    abs(x["guard_filtered"]["rgb"] - x["guard_filtered"]["color"]) >= gate
                    and x["agreement"] for x in calibrated),
                max_correction=max(x["rgb_settle_correction"] for x in calibrated))
        extra=[]
        if scene["expected_stop"]=="bottle_empty" and plan.get("empty_confirmation_replay_sha256"):
            zero_time=next((x["t"] for x in samples if x["bottle_ml"]==0.),None)
            stop_sample=min(samples,key=lambda x:abs(x["t"]-stop["t"])) if stop else None
            row["first_zero_source_s"]=zero_time
            row["empty_confirmation_lag_s"]=(round(stop["t"]-zero_time,6)
                if stop and zero_time is not None and row["stop_reason"]=="bottle_empty" else None)
            row["source_at_stop_ml"]=stop_sample["bottle_ml"] if stop_sample else None
            row["tall_empty_votes_at_stop"]=stop.get("tall_empty_votes") if stop else None
            row["tall_empty_wrist_votes_at_stop"]=stop.get("tall_empty_wrist_votes") if stop else None
            row["tall_empty_view_at_stop"]=stop.get("tall_empty_view") if stop else None
            if row["source_at_stop_ml"]!=0. or row["source_remaining_ml"]!=0.:
                extra.append("source_not_depleted_at_stop")
        if scene.get("bottle_size_autonomy"):
            recorded=read(folder/"metrics.json")["scene"]
            row["bottle_geometry"]=recorded["bottle_geometry"]
            row["bottle_authored_mass_properties"]=recorded["bottle_authored_mass_properties"]
            if recorded["bottle_geometry"] != scene["declared_bottle_geometry"]:
                extra.append("recorded_bottle_geometry_mismatch")
            if recorded["bottle_kinematic"] or not recorded["bottle_gravity_enabled"] or recorded["bottle_attachment_joints"]:
                extra.append("bottle_was_not_free")
        if peak>=plan["maximum_sample_height"]:extra.append("headroom_exceeded")
        if not ordered or not side:extra.append("side_return_incomplete")
        if not perception or not perception["agreement"]:extra.append("sized_perception_stop_unconfirmed")
        row["failure_reasons"]=list(dict.fromkeys(reasons(row)+extra))
        row["strict_pass"]=strict_success(row) and not extra
        row["within_particle_tolerance"]=particles<=plan["provisional_particle_limit"] and all(
            r=="measured_spill" for r in row["failure_reasons"])
        rows[name]=row
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--part",choices=("development","test"),required=True)
    parser.add_argument("--label",required=True)
    parser.add_argument("--seconds",type=int,default=6000)
    parser.add_argument("--shard",type=int)
    parser.add_argument("--study",choices=("sized-autonomy","sized-autonomy-r1","bottle-autonomy",
                                          "bottle-empty-autonomy","mixed-vessel-autonomy","tall-empty-r1",
                                          "layout-repair-r2","layout-repair-r3","tall-aligned-r0",
                                          "wide-mixed-r0","wide-mixed-r1","wide-mixed-r2","wide-mixed-r3",
                                          "wide-mixed-r4","wide-mixed-r5","wide-mixed-r6","wide-mixed-r7",
                                          "wide-mixed-r7b","wide-mixed-r8","wide-mixed-r9"),
                        default="sized-autonomy")
    args=parser.parse_args()
    assert args.part=="test" or args.shard is None
    deadline_limit = 7200 if args.part == "development" else 14400
    assert args.label.replace("-","").isalnum() and 0 < args.seconds <= deadline_limit
    with (ROOT/"isolated-job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        minimum_free_gib = 57.
        if args.study in ("wide-mixed-r8","wide-mixed-r9"):
            plan = read(ROOT/args.study/"plan.json")
            # Bound storage by the actual batch and measured prior recording
            # sizes. Keep the 10 GiB retained margin from the storage plan.
            largest = plan["storage_recording_upper_bytes"]
            if args.part == "development":
                count = len(plan["development"])
            else:
                assert args.shard is not None and 1<=args.shard<=len(plan["test_shards"])
                count = sum(len(s) for s in plan["test_shards"][args.shard-1:])
            minimum_free_gib = 10. + count*largest/1024**3
        subprocess.run([sys.executable,str(HOST/"tools/verify_robot_training_storage.py"),
                        "--minimum-free-gib",str(minimum_free_gib)],check=True)
        original=json.loads(out("docker","inspect",ORIGINAL))[0]
        assert original["Id"]==ORIGINAL and not original["State"]["Running"]
        assert original["Config"]["Labels"]["owner"]=="aiworker-pouring"
        gpu_query=("nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv,noheader")
        before=out(*gpu_query)
        assert not any(r.startswith(GPU) for r in before.splitlines()),"4070 occupied"
        run(args,original,gpu_query,before)


def run(args,original,gpu_query,before):
    study=ROOT/args.study
    plan=read(study/"plan.json")
    geometry=read(study/"geometry.json")
    assert geometry["passed"] and geometry["plan_sha256"]==sha(study/"plan.json")
    assert all(sha(study/"src"/n)==h for n,h in geometry["sources"].items())
    assert sha(ROOT/"followup-selection.json")==plan["perception_selection_sha256"]
    assert sha(ROOT/"followup-fresh.json")==plan["perception_test_sha256"]
    candidates={s["name"]:s for s in plan[args.part]}
    result_name=args.part
    if args.part=="test" and "test_shards" in plan:
        shards=plan["test_shards"]
        names=[name for shard in shards for name in shard]
        assert len(names)==len(set(names)) and set(names)==candidates.keys()
        assert args.shard is not None and 1<=args.shard<=len(shards)
        candidates={name:candidates[name] for name in shards[args.shard-1]}
        result_name=f"test-{args.shard:02d}"
    else:
        assert args.shard is None
    assert candidates.keys() <= {r["name"] for r in geometry["cases"] if r["passed"]}
    if args.part=="test":
        dev=read(study/"development-results.json")
        assert dev["plan_sha256"]==sha(study/"plan.json") and dev["completed"]==len(plan["development"])
        required = "strict_pass" if plan.get("development_requires_strict_zero") else "within_particle_tolerance"
        assert all(r[required] for r in dev["cases"]), "Development verification failed"
        if "development_empty_lag_limit_s" in plan:
            empty_cases={r["name"]:r for r in dev["cases"] if r["expected_stop"]=="bottle_empty"}
            timing_names=plan.get("development_empty_lag_cases",list(empty_cases))
            assert timing_names and all(name in empty_cases for name in timing_names)
            assert all(empty_cases[name]["empty_confirmation_lag_s"] is not None and
                       0. <= empty_cases[name]["empty_confirmation_lag_s"] <= plan["development_empty_lag_limit_s"]
                       for name in timing_names), "Tall-bottle latency verification failed"
    assert all(sha(native(p))==h for p,h in plan["checkpoints"].items())
    for old in OUTPUT.glob("*/*/*SA_A_*/parameters.json"):
        assert read(old)["name"] not in candidates,"Trial ID already used"
    output=OUTPUT/args.label
    output.mkdir(parents=True,exist_ok=False)
    batch=study/(args.label+"-batch.json")
    files={sandbox(study/"plan.json"):sha(study/"plan.json"),
           sandbox(study/"geometry.json"):sha(study/"geometry.json"),**plan["checkpoints"]}
    if "settled_rgb_calibration" in plan:
        receipt=study/"calibration.json"
        digest=plan["settled_rgb_calibration"]["receipt_sha256"]
        assert sha(receipt)==digest
        files[sandbox(receipt)]=digest
    for name,digest in plan.get("prerequisite_reports",{}).items():
        assert sha(native(name))==digest
        files[name]=digest
    if "empty_confirmation_replay_sha256" in plan:
        replay=study/"empty-replay.json"
        digest=plan["empty_confirmation_replay_sha256"]
        assert sha(replay)==digest
        files[sandbox(replay)]=digest
    files.update({"/workspace/src/"+name:h for name,h in geometry["sources"].items()})
    for selection_name in ("followup-selection.json","component-selection.json"):
        for name,digest in read(ROOT/selection_name)["frozen_files"].items():
            if name.endswith(".py"):
                path=ROOT/name
                assert sha(path)==digest
                files[sandbox(path)]=digest
    with batch.open("x") as f:
        json.dump(dict(run_authorized=True,cases=list(candidates.values()),files=files),f,indent=2)
    cmd=["docker","create","--name","aiworker_sized_4070_"+args.label,
         "--label","owner=aiworker-pouring","--label","purpose=sized-autonomous-pour","--restart","no",
         "--network","none","--gpus","device="+GPU,"--cpuset-cpus","16-23","--cpus","4","--cpu-shares","256",
         "--memory","16g","--memory-swap","16g","--shm-size","1g","--pids-limit","512",
         "--security-opt","no-new-privileges","--user",original["Config"]["User"],"--workdir","/isaac-sim",
         "-v",str(HOST)+":/workspace:ro",
         "-v",str(study/"src")+":/workspace/src:ro","-v",str(output)+":"+sandbox(output)+":rw"]
    for mount in original["Mounts"]:
        if mount["Destination"] not in ("/workspace","/root/.nvidia-omniverse/logs"):
            cmd+=["-v",mount["Source"]+":"+mount["Destination"]+":"+("rw" if mount["RW"] else "ro")]
    env={k:v for k,v in (entry.split("=",1) for entry in original["Config"]["Env"])
         if k in ("ACCEPT_EULA","VK_DRIVER_FILES","NVIDIA_DRIVER_CAPABILITIES","OMNICLIENT_HUB_EXE","MIN_DRIVER_VERSION")}
    env.update(NVIDIA_VISIBLE_DEVICES=GPU, PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1",PYTHONPATH="/workspace/src:/workspace:"+sandbox(ROOT),
               AIWORKER_AUTONOMY_BATCH=sandbox(batch),AIWORKER_AUTONOMY_OUTPUT=sandbox(output))
    for key,value in env.items():
        cmd+=["-e",key+"="+value]
    cmd+=["--entrypoint","/usr/bin/nice",original["Image"],"-n","10",
          "/isaac-sim/python.sh","-u","/workspace/src/run_fountain.py","--allow-root"]
    cid=out(*cmd)
    info=json.loads(out("docker","inspect",cid))[0]
    witness=dict(container=cid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 gpu_before=before,limits=info["HostConfig"],batch_sha256=sha(batch),plan_sha256=sha(study/"plan.json"))
    save(study/(args.label+"-before.json"),witness)
    started=time.monotonic()
    stop=None
    failures=0
    last_progress=0.
    logged=set()
    try:
        out("docker","start",cid)
        while json.loads(out("docker","inspect",cid))[0]["State"]["Running"]:
            try:
                gpu=out(*gpu_query)
                own={int(x) for x in out("docker","top",cid,"-eo","pid").splitlines()[1:]}
                if foreign_gpu_pids(gpu,own,cid):
                    stop="foreign_4070_job"
                    break
                failures=0
            except (OSError,ValueError,IndexError,subprocess.SubprocessError):
                failures+=1
                if failures>=3:
                    stop="gpu_monitoring_failed"
                    break
            elapsed=time.monotonic()-started
            if elapsed>args.seconds:
                stop="deadline"
                break
            if elapsed-last_progress>=25:
                last_progress=elapsed
                for path in output.glob("*/*SA_A_*/metrics.json"):
                    if path in logged:
                        continue
                    metrics=read(path)
                    observed=metrics["bottle_result"]
                    print(json.dumps(dict(name=read(path.parent/"parameters.json")["name"],
                                     stop=observed["stop_reason"],level=observed["cup_fill_fraction"],
                                     spill=metrics["spill"])),flush=True)
                    logged.add(path)
            time.sleep(2)
    finally:
        state=json.loads(out("docker","inspect",cid))[0]["State"]
        if state["Running"]:
            out("docker","stop","--timeout","8",cid,timeout=30)
            state=json.loads(out("docker","inspect",cid))[0]["State"]
        log=subprocess.run(["docker","logs",cid],capture_output=True,text=True,timeout=30)
        (study/(args.label+".log")).write_text(log.stdout+log.stderr)
        witness.update(state=state,stop_reason=stop,gpu_after=out(*gpu_query),elapsed=time.monotonic()-started)
        save(study/(args.label+"-isolation.json"),witness)
    rows=rows_from_disk(output,candidates,plan)
    completed=[r for r in rows.values() if r["status"]=="complete"]
    report=dict(part=args.part,plan_sha256=sha(study/"plan.json"),planned=len(candidates),completed=len(completed),
                strict_passes=sum(r["strict_pass"] for r in completed),
                tolerance_passes=sum(r["within_particle_tolerance"] for r in completed),
                cases=[rows.get(n,dict(name=n,status="unstarted")) for n in candidates],runtime=witness)
    save(study/(result_name+"-results.json"),report)
    print(json.dumps({k:report[k] for k in ("part","planned","completed","strict_passes","tolerance_passes")}),flush=True)
    assert stop is None and len(completed)==len(candidates) and "AUTONOMY_BATCH_COMPLETE" in log.stdout


if __name__=="__main__":
    main()
