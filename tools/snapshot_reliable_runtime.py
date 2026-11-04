"""Run on Ubuntu; read-only isolation/provenance snapshot, no job control."""
import datetime,hashlib,json,subprocess,sys,urllib.request
from pathlib import Path
ROOT=Path('/home/kadir/aiworker-pouring-4070')
sys.path.insert(0,str(ROOT/'src'))
from pour_reliable_study import RUNTIME_SOURCES

def out(*args):return subprocess.check_output(args,text=True,timeout=15).strip()
info=json.loads(out('docker','inspect','aiworker_pouring_4070'))[0]
assert info['Config']['Labels']['owner']=='aiworker-pouring'
h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
model=ROOT/'learning/A-reliable-20261103-v1/best.pt'
record=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),container=info['Id'],
    limits={k:info['HostConfig'][k] for k in ('NanoCpus','CpuShares','CpusetCpus','Memory','MemorySwap','DeviceRequests')},
    sources={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in RUNTIME_SOURCES},
    selected_checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest() if model.exists() else None,
    asset_sha256=hashlib.sha256((ROOT/'assets/FFW_BG2_without_mimic.usd').read_bytes()).hexdigest(),
    guard_processes=[x for x in out('pgrep','-af','gpu_priority_guard.py').splitlines() if info['Id'] in x],
    gpu_compute_processes=out('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'),
    gpu_utilization=out('nvidia-smi','--query-gpu=name,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'),memory=out('free','-m'),
    health={k:h.get(k) for k in ('run','phase','sim_time','queue','recording','error','learning_error','learning')})
assert record['guard_processes']
print(json.dumps(record,indent=2))
