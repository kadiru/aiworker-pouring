"""Bounded host launcher: 4070-only fitting without starting the simulator.

Serializes this project's jobs and stops only its immutable child container
on competing GPU use, monitoring failure, or deadline. No automatic restart.
"""
import argparse
import datetime
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/kadir/aiworker-pouring-4070')
STUDY = ROOT/'learning/A-scale-dynamic-20261109-v1'
SIM = '0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'
GPU = 'GPU-cde23916-c5fa-f18e-a343-127f732cc2f6'
sys.path.insert(0, str(ROOT))
from gpu_priority_guard import foreign_gpu_pids

def out(*args):
    return subprocess.check_output(args, text=True, timeout=20).strip()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--helper', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--seconds', type=int, default=3600)
    p.add_argument('--minimum-free-gib', type=float, default=57)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a=p.parse_args()
    assert Path(a.helper).name==a.helper and a.helper.endswith('.py')
    assert a.label.replace('-','').isalnum() and 0<a.seconds<=7200
    assert a.minimum_free_gib>=10
    assert (STUDY/a.helper).is_file()
    with (STUDY/'isolated-job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run([sys.executable,str(ROOT/'tools/verify_robot_training_storage.py'),
                        '--minimum-free-gib',str(a.minimum_free_gib)],check=True)
        sim=json.loads(out('docker','inspect',SIM))[0]
        assert sim['Id']==SIM and not sim['State']['Running']
        assert sim['Config']['Labels']['owner']=='aiworker-pouring'
        before=out('nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader')
        assert not any(r.startswith(GPU) for r in before.splitlines()),'4070 occupied; refusing launch'
        dest=STUDY/a.label;dest.mkdir()
        args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
        cmd=['docker','create','--name','aiworker_perception_4070_'+a.label,'--label','owner=aiworker-pouring',
             '--label','purpose=perception-repair','--restart','no','--network','none',
             '--gpus','device='+GPU,'--cpuset-cpus','16-23','--cpus','4','--cpu-shares','256',
             '--memory','16g','--memory-swap','16g','--shm-size','1g','--pids-limit','512',
             '--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges',
             '-e','NVIDIA_VISIBLE_DEVICES='+GPU,'-e','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
             '-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4','-e','OPENBLAS_NUM_THREADS=1',
             '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/workspace/src:/workspace',
             '-v',str(ROOT)+':/workspace:ro','-v',str(STUDY)+':/workspace/learning/A-scale-dynamic-20261109-v1:rw',
             '-v','/home/kadir/isaac-sim:/isaac-sim:ro','--entrypoint','/usr/bin/nice',sim['Image'],
             '-n','10','/isaac-sim/python.sh','-u','/workspace/learning/A-scale-dynamic-20261109-v1/'+a.helper,*args]
        cid=out(*cmd);start=time.monotonic();reason=None;inspection_failures=0;snapshots=[]
        info=json.loads(out('docker','inspect',cid))[0]
        report=dict(container=cid,helper=a.helper,arguments=args,limits=info['HostConfig'],gpu_before=before,
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),simulator_started=False)
        (dest/'before.json').write_text(json.dumps(report,indent=2)+'\n')
        try:
            out('docker','start',cid)
            while json.loads(out('docker','inspect',cid))[0]['State']['Running']:
                try:
                    rows=out('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader')
                    own={int(s.strip()) for s in out('docker','top',cid,'-eo','pid').splitlines()[1:]}
                    foreign=foreign_gpu_pids(rows,own,cid)
                    if foreign:reason='foreign_4070_job';break
                    if time.monotonic()-start>a.seconds:reason='wall_deadline';break
                    if not snapshots or time.monotonic()-start-snapshots[-1]['elapsed']>=60:
                        snapshots.append(dict(elapsed=time.monotonic()-start,gpu_processes=rows))
                    inspection_failures=0
                except (subprocess.SubprocessError,OSError,ValueError,IndexError):
                    inspection_failures+=1
                    if inspection_failures>=3:reason='monitoring_failure';break
                time.sleep(2)
        finally:
            state=json.loads(out('docker','inspect',cid))[0]['State']
            if state['Running']:out('docker','stop','--timeout','5',cid)
            state=json.loads(out('docker','inspect',cid))[0]['State']
            log=subprocess.run(['docker','logs',cid],capture_output=True,text=True,timeout=20)
            (dest/'run.log').write_text(log.stdout+log.stderr)
            report.update(state=state,stop_reason=reason,elapsed=time.monotonic()-start,snapshots=snapshots,
                          gpu_after=out('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'))
            (dest/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(container=cid,exit_code=state['ExitCode'],stop_reason=reason,elapsed=report['elapsed'])),flush=True)
        assert state['ExitCode']==0 and reason is None

if __name__=='__main__':main()
