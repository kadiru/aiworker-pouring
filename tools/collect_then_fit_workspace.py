"""Finite continuation: finish references, fit once, queue development once.

Run on Ubuntu. Never restarts containers, changes limits, touches other jobs,
freezes a model, or starts the reserved test batch. Stops for failed references.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path('/home/kadir/aiworker-pouring-4070')
STUDY=ROOT/'learning/A-workspace-20261104-v1'
EXPECTED='697166fd9ca8f803f339097137f1d47fc80ff560d43bf71f0ab6d77869474cbd'
URL='http://192.168.68.60:8212'


def health():
    h=json.load(urllib.request.urlopen(URL+'/health',timeout=10))
    assert not h.get('error') and not h.get('learning_error'),h.get('error') or h.get('learning_error')
    return h


def isolation():
    info=json.loads(subprocess.check_output(['docker','inspect','aiworker_pouring_4070']))[0]
    assert info['Id']==EXPECTED and info['State']['Running'] and info['Config']['Labels']['owner']=='aiworker-pouring'
    c=info['HostConfig']
    assert c['NanoCpus']==4_000_000_000 and c['Memory']==c['MemorySwap']==16*1024**3 and c['CpusetCpus']=='16-23' and c['CpuShares']==256
    assert c['DeviceRequests'][0]['DeviceIDs']==['GPU-cde23916-c5fa-f18e-a343-127f732cc2f6']
    assert EXPECTED in subprocess.check_output(['pgrep','-af','gpu_priority_guard.py'],text=True)
    return info


def command(name):
    isolation();h=health()
    assert not h['queue'] and not (h.get('recording') or {}).get('active')
    req=urllib.request.Request(URL+'/control',data=json.dumps(dict(command=name)).encode(),headers={'Content-Type':'application/json'})
    assert urllib.request.urlopen(req,timeout=10).status==202


def record(stage,**data):
    value=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**data)
    (ROOT/'workspace-pipeline-progress.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


def main():
    plan=json.loads((STUDY/'frozen-plan.json').read_text())
    collection=json.loads((STUDY/'collection-selection.json').read_text())
    expected={s['name'] for s in plan['demonstrations']}
    assert not (STUDY/'best.pt').exists() and not (STUDY/'final-selection.json').exists()
    for _ in range(540):
        isolation();h=health();rows={}
        for n,sha in collection['runtime_sources'].items():assert hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest()==sha,n
        for p in (ROOT/'outputs/fountain').glob('*/*W30_A_demo_*/metrics.json'):
            name=json.loads((p.parent/'parameters.json').read_text())['name']
            assert name not in rows;rows[name]=json.loads(p.read_text())
        failed=[n for n,m in rows.items() if not (m['grasp_verified'] and m['zero_spill'] and not m['gates'])]
        record('waiting_for_reference_collection',completed=len(rows),planned=len(expected),failed=failed)
        assert not failed,'Retain and review failed reference demonstrations before fitting'
        if set(rows)==expected and not h['queue'] and not h['recording']['active']:break
        time.sleep(40)
    else:raise RuntimeError('Reference collection time limit; inspect owned job without restarting others')
    command('workspace_train');record('fit_requested')
    for _ in range(90):
        time.sleep(10);isolation();h=health()
        if (STUDY/'model-summary.json').exists() and (h.get('learning') or {}).get('stage')=='fit_complete':break
    else:raise RuntimeError('Fit did not complete in fifteen minutes; inspect native state')
    m=json.loads((STUDY/'model-summary.json').read_text())
    assert hashlib.sha256((STUDY/'best.pt').read_bytes()).hexdigest()==m['checkpoint_sha256']
    assert m['device']['visible_cuda_devices']==1 and '4070' in m['device']['name']
    for n,sha in m['sources'].items():assert hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest()==sha,n
    record('fit_complete',checkpoint_sha256=m['checkpoint_sha256'],training=m['training'],metrics=m['chosen']['metrics'])
    assert not list((ROOT/'outputs/fountain').glob('*/*W30_A_dev_*/parameters.json'))
    command('workspace_development')
    record('development_queued',cases=len(plan['development']),round=0,checkpoint_sha256=m['checkpoint_sha256'],reserved_tests_started=False)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        record('stopped_for_inspection',error=repr(exc));raise
