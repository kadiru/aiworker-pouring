"""Resume only the existing owned simulator after exclusive 4070 preflight."""
import datetime,json,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/home/kadir/aiworker-pouring-4070')
STUDY=ROOT/'learning/A-scale-dynamic-20261109-v1'
CID='0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'
GPU='GPU-cde23916-c5fa-f18e-a343-127f732cc2f6'
def out(*a):return subprocess.check_output(a,text=True,timeout=25).strip()
def main():
    subprocess.run([sys.executable,str(ROOT/'tools/verify_robot_training_storage.py'),'--minimum-free-gib','57'],check=True)
    i=json.loads(out('docker','inspect',CID))[0];c=i['HostConfig']
    assert i['Id']==CID and i['Name']=='/aiworker_pouring_4070' and i['Config']['Labels']['owner']=='aiworker-pouring'
    assert not i['State']['Running'] and c['DeviceRequests'][0]['DeviceIDs']==[GPU]
    assert c['NanoCpus']==4000000000 and c['CpusetCpus']=='16-23' and c['CpuShares']==256
    assert c['Memory']==c['MemorySwap']==17179869184
    before=out('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader')
    assert not any(r.startswith(GPU) for r in before.splitlines()),'4070 occupied; simulator stays stopped'
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out('docker','start',CID)
    with (STUDY/('sim-priority-guard-'+stamp+'.log')).open('x') as f:
        guard=subprocess.Popen([sys.executable,str(ROOT/'gpu_priority_guard.py'),CID],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        assert guard.poll() is None,'Guard exited; inspect before retrying'
        try:
            health=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=5))
            if health.get('error') or health.get('learning_error'):raise RuntimeError(str(health))
            if (health.get('ready') or health.get('phase')=='Ready') and health.get('run') and not health.get('queue') and not (health.get('recording') or {}).get('active'):
                report=dict(container=CID,guard_pid=guard.pid,health=health,gpu_before=before,limits={k:c[k] for k in ('DeviceRequests','NanoCpus','CpusetCpus','CpuShares','Memory','MemorySwap')})
                (STUDY/('sim-start-'+stamp+'.json')).write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(dict(ready=True,run=health['run'],guard_pid=guard.pid)),flush=True);return
        except (OSError,ValueError):pass
        time.sleep(2)
    raise TimeoutError('Simulator did not become ready; inspect preserved logs')
if __name__=='__main__':main()
