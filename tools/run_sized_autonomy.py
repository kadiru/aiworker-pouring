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
STUDY = ROOT / "sized-autonomy"
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
        extra=[]
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
    args=parser.parse_args()
    assert args.label.replace("-","").isalnum() and 0<args.seconds<=7200
    with (ROOT/"isolated-job.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run([sys.executable,str(HOST/"tools/verify_robot_training_storage.py"),"--minimum-free-gib","57"],check=True)
        original=json.loads(out("docker","inspect",ORIGINAL))[0]
        assert original["Id"]==ORIGINAL and not original["State"]["Running"]
        assert original["Config"]["Labels"]["owner"]=="aiworker-pouring"
        gpu_query=("nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv,noheader")
        before=out(*gpu_query)
        assert not any(r.startswith(GPU) for r in before.splitlines()),"4070 occupied"
        run(args,original,gpu_query,before)


def run(args,original,gpu_query,before):
    plan=read(STUDY/"plan.json")
    geometry=read(STUDY/"geometry.json")
    assert geometry["passed"] and geometry["plan_sha256"]==sha(STUDY/"plan.json")
    assert all(sha(STUDY/"src"/n)==h for n,h in geometry["sources"].items())
    assert sha(ROOT/"followup-selection.json")==plan["perception_selection_sha256"]
    assert sha(ROOT/"followup-fresh.json")==plan["perception_test_sha256"]
    candidates={s["name"]:s for s in plan[args.part]}
    assert candidates.keys() <= {r["name"] for r in geometry["cases"] if r["passed"]}
    if args.part=="test":
        dev=read(STUDY/"development-results.json")
        assert dev["plan_sha256"]==sha(STUDY/"plan.json") and dev["completed"]==len(plan["development"])
        assert all(r["within_particle_tolerance"] for r in dev["cases"])
    assert all(sha(native(p))==h for p,h in plan["checkpoints"].items())
    for old in OUTPUT.glob("*/*/*SA_A_*/parameters.json"):
        assert read(old)["name"] not in candidates,"Trial ID already used"
    output=OUTPUT/args.label
    output.mkdir(parents=True,exist_ok=False)
    batch=STUDY/(args.label+"-batch.json")
    files={sandbox(STUDY/"plan.json"):sha(STUDY/"plan.json"),
           sandbox(STUDY/"geometry.json"):sha(STUDY/"geometry.json"),**plan["checkpoints"]}
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
         "-v",str(STUDY/"src")+":/workspace/src:ro","-v",str(output)+":"+sandbox(output)+":rw"]
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
                 gpu_before=before,limits=info["HostConfig"],batch_sha256=sha(batch),plan_sha256=sha(STUDY/"plan.json"))
    save(STUDY/(args.label+"-before.json"),witness)
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
        (STUDY/(args.label+".log")).write_text(log.stdout+log.stderr)
        witness.update(state=state,stop_reason=stop,gpu_after=out(*gpu_query),elapsed=time.monotonic()-started)
        save(STUDY/(args.label+"-isolation.json"),witness)
    rows=rows_from_disk(output,candidates,plan)
    completed=[r for r in rows.values() if r["status"]=="complete"]
    report=dict(part=args.part,plan_sha256=sha(STUDY/"plan.json"),planned=len(candidates),completed=len(completed),
                strict_passes=sum(r["strict_pass"] for r in completed),
                tolerance_passes=sum(r["within_particle_tolerance"] for r in completed),
                cases=[rows.get(n,dict(name=n,status="unstarted")) for n in candidates],runtime=witness)
    save(STUDY/(args.part+"-results.json"),report)
    print(json.dumps({k:report[k] for k in ("part","planned","completed","strict_passes","tolerance_passes")}),flush=True)
    assert stop is None and len(completed)==len(candidates) and "AUTONOMY_BATCH_COMPLETE" in log.stdout


if __name__=="__main__":
    main()
