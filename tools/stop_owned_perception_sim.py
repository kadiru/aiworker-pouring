"""Stop the idle owned simulator after completed reserved recording/audit."""
import datetime,json,subprocess,urllib.request
from pathlib import Path
ROOT=Path('/home/kadir/aiworker-pouring-4070/learning/A-scale-dynamic-20261109-v1')
CID='0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'


def out(*args):return subprocess.check_output(args,text=True,timeout=45).strip()


def main():
    receipt=ROOT/'sim-stop-after-holdout.json';assert not receipt.exists()
    audit=json.loads((ROOT/'holdout-r0-audit.json').read_text())
    assert audit['complete'] and len(audit['records'])==18
    health=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not health['queue'] and not (health.get('recording') or {}).get('active')
    info=json.loads(out('docker','inspect',CID))[0]
    assert info['Id']==CID and info['Name']=='/aiworker_pouring_4070'
    assert info['Config']['Labels']['owner']=='aiworker-pouring' and info['State']['Running']
    processes=out('docker','top',CID,'-eo','pid,args')
    assert not any('/workspace/learning/A-scale-dynamic-20261109-v1/' in line and '/isaac-sim' in line for line in processes.splitlines()),'Owned audit/evaluation must finish first'
    out('docker','stop','--timeout','30',CID)
    state=json.loads(out('docker','inspect',CID))[0]['State'];assert not state['Running']
    report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),container=CID,health_before=health,state_after=state,
                gpu_after=out('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'),
                scope='Stopped only the idle owned simulator after 18 completed recordings and audit. No foreign process signaled or configured.')
    with receipt.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(simulator_stopped=True,container=CID,state=state)),flush=True)


if __name__=='__main__':main()
