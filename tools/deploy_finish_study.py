"""Host-side guarded deployment to the idle, immutable AIWorker container only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path('/home/kadir/aiworker-pouring-4070')
STUDY = ROOT / 'learning/A-finish-20261106-v1'
GPU = 'GPU-cde23916-c5fa-f18e-a343-127f732cc2f6'


def out(*args):
    return subprocess.check_output(args, text=True, timeout=35).strip()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--expected-container', required=True)
    p.add_argument('--phase', required=True)
    p.add_argument('files', nargs='+')
    a=p.parse_args()
    info=json.loads(out('docker','inspect',a.expected_container))[0]
    assert info['Id']==a.expected_container and info['Name']=='/aiworker_pouring_4070'
    assert info['Config']['Labels']['owner']=='aiworker-pouring'
    caps=info['HostConfig']
    assert caps['DeviceRequests'][0]['DeviceIDs']==[GPU]
    assert caps['NanoCpus']==4000000000 and caps['CpusetCpus']=='16-23' and caps['CpuShares']==256
    assert caps['Memory']==caps['MemorySwap']==17179869184
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not h['queue'] and not h.get('recording',{}).get('active') and not h.get('error')
    assert not (STUDY/'final-selection.json').exists(), 'Never mutate a frozen reserved evaluation'
    own={int(s.strip()) for s in out('docker','top',a.expected_container,'-eo','pid').splitlines()[1:]}
    gpu=out('nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader')
    assert all(int(r.split(',')[1]) in own for r in gpu.splitlines() if r.startswith(GPU))
    assert all('/' not in n and n.endswith('.py') for n in a.files)
    payload=dict(phase=a.phase,files=a.files,container=info['Id'],
        limits={k:caps[k] for k in ('DeviceRequests','NanoCpus','CpusetCpus','CpuShares','Memory','MemorySwap')},
        health={k:h.get(k) for k in ('run','phase','queue','recording','error')}, gpu_processes=gpu)
    code="""
import ast,datetime,hashlib,json,shutil,sys
from pathlib import Path
p=json.load(sys.stdin);root=Path('/workspace');study=root/'learning/A-finish-20261106-v1'
archive=study/('before-'+p['phase']);archive.mkdir(exist_ok=False)
for name in p['files']:ast.parse((study/'staged'/name).read_text())
sources={}
for path in (root/'src').glob('*.py'):
 shutil.copy2(path,archive/path.name);sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
p.update(sources=sources,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(archive/'snapshot.json').write_text(json.dumps(p,indent=2)+'\\n')
for name in p['files']:shutil.copy2(study/'staged'/name,root/'src'/name)
(study/('deployment-'+p['phase']+'.json')).write_text(json.dumps(dict(**p,new_sources={n:hashlib.sha256((root/'src'/n).read_bytes()).hexdigest() for n in p['files']}),indent=2)+'\\n')
"""
    subprocess.run(['docker','exec','-i',a.expected_container,'/isaac-sim/python.sh','-c',code],
                   input=json.dumps(payload),text=True,check=True,timeout=40)
    out('docker','stop','--timeout','15',a.expected_container)
    assert not any(r.startswith(GPU) for r in out('nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader').splitlines()), '4070 acquired by another job; ours stays stopped'
    out('docker','start',a.expected_container)
    guards=subprocess.run(['pgrep','-af','gpu_priority_guard.py '+a.expected_container],capture_output=True,text=True)
    if guards.returncode:
        with Path('/tmp/aiworker-wide-gpu-priority-guard.log').open('a') as log:
            subprocess.Popen([sys.executable,str(ROOT/'gpu_priority_guard.py'),a.expected_container],
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(deployed=a.files,phase=a.phase,owned_container=a.expected_container)),flush=True)


if __name__=='__main__':
    main()
