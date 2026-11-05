"""Run on Ubuntu: freeze this evaluation and restart only the idle owned container.

Source files must first be copied to this study's harness-staged directory.
The exact owned ID, idle state, caps, current source identity and GPU ownership
are checked before any running source is replaced or a process is restarted.
"""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path('/home/kadir/aiworker-pouring-4070')
STUDY=ROOT/'learning/A-layout-variety-20261105-v1'
GPU='GPU-cde23916-c5fa-f18e-a343-127f732cc2f6'

def out(*args): return subprocess.check_output(args,text=True,timeout=35).strip()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    global STUDY
    p=argparse.ArgumentParser();p.add_argument('--expected-container',required=True);p.add_argument('--source-commit',required=True);p.add_argument('--extra',action='store_true');args=p.parse_args()
    if args.extra:STUDY=ROOT/'learning/A-layout-variety-extra-20261105-v1'
    assert not (STUDY/'selection.json').exists(), 'Do not redeploy a frozen evaluation'
    info=json.loads(out('docker','inspect',args.expected_container))[0]
    assert info['Id']==args.expected_container and info['Name']=='/aiworker_pouring_4070'
    assert info['Config']['Labels']['owner']=='aiworker-pouring'
    limits=info['HostConfig']
    assert limits['DeviceRequests'][0]['DeviceIDs']==[GPU]
    assert limits['NanoCpus']==4000000000 and limits['CpusetCpus']=='16-23' and limits['CpuShares']==256
    assert limits['Memory']==limits['MemorySwap']==17179869184
    health=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    assert not health['queue'] and not health.get('recording',{}).get('active') and not health.get('error')
    own={int(x.strip()) for x in out('docker','top',args.expected_container,'-eo','pid').splitlines()[1:]}
    gpu=out('nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader')
    assert all(int(row.split(',')[1]) in own for row in gpu.splitlines() if row.startswith(GPU))
    previous=ROOT/'learning/A-layout-variety-20261105-v1'
    if args.extra:
        assert json.loads((previous/'watcher-completion.json').read_text())['all_cases_complete']
        assert json.loads((previous/'final-benchmark.json').read_text())['counts']['complete']==12
    old=json.loads((previous/'selection.json' if args.extra else ROOT/'learning/A-workspace-20261104-v1/final-selection.json').read_text())
    for name,digest in old['runtime_sources'].items(): assert sha(ROOT/'src'/name)==digest,name
    assert sha(ROOT/'learning/A-workspace-20261104-v1/best.pt')==old['selected_checkpoint_sha256']
    preflights=[('preflight.json','ready')] if args.extra else [('preflight.json','ready'),('initial-object-preflight.json','clear')]
    for name,key in preflights:
        assert all(x[key] for x in json.loads((STUDY/name).read_text())['cases'])
    staged=STUDY/'harness-staged';changed=('run_fountain.py','pour_layout_extra.py') if args.extra else ('run_fountain.py','pour_recording.py','pour_layout_variety.py')
    for name in changed: ast.parse((staged/name).read_text())
    archive=STUDY/'harness-before';archive.mkdir(exist_ok=False)
    for name in old['runtime_sources']: shutil.copy2(ROOT/'src'/name,archive/name)
    (archive/'identity.json').write_text(json.dumps(old,indent=2)+'\n')
    # Idle source replacement does not change the in-memory actor. Restart just
    # this immutable ID to load the recording/queue harness; actor is unchanged.
    for name in changed: shutil.copy2(staged/name,ROOT/'src'/name)
    frozen=STUDY/'frozen-plan.json';shutil.copy2(STUDY/'proposed-plan.json',frozen)
    sources={name:sha(ROOT/'src'/name) for name in dict.fromkeys((*old['runtime_sources'],*changed))}
    assert {n for n,h in old['runtime_sources'].items() if sources[n]!=h}==({'run_fountain.py'} if args.extra else {'run_fountain.py','pour_recording.py'})
    selection=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=args.source_commit,selected_checkpoint_sha256=old['selected_checkpoint_sha256'],
        scene_plan_sha256=sha(frozen),runtime_sources=sources,
        unchanged_controller=True,changes='Only extra layout queue command; actor and all camera/physics code unchanged from original twelve.' if args.extra else 'New queue command and optional recording-only overview camera; learned actor, weights, thresholds, physics and task/wrist calibration unchanged.',
        container=args.expected_container,plan_count=len(json.loads(frozen.read_text())['test']),attempts_per_layout=1,
        primary_definition='Verified physical and visual grasp, bilateral contacts >=80% of tipping samples, >=1 nominal mL transferred, correct near-full/empty stop, >=90% final rim height or <=max(.25 nominal mL,1% initial) source, ordered completed upright return/release, no gates, zero final outside liquid and zero sampled near-table loss.',
        preflight_hashes={n:sha(STUDY/n) for n,_ in preflights})
    (STUDY/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('FROZEN '+selection['scene_plan_sha256'],flush=True)
    out('docker','stop','--timeout','15',args.expected_container)
    assert not any(row.startswith(GPU) for row in out('nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader').splitlines()), '4070 acquired by another job; leave ours stopped'
    out('docker','start',args.expected_container)
    # An existing bound guard can survive the brief restart; replace only if absent.
    guard=subprocess.run(['pgrep','-af','gpu_priority_guard.py '+args.expected_container],capture_output=True,text=True)
    if guard.returncode:
        with (ROOT/'outputs/gpu-priority-guard.log').open('a') as log:
            subprocess.Popen([sys.executable,str(ROOT/'gpu_priority_guard.py'),args.expected_container],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('RESTARTED_OWNED_CONTAINER '+args.expected_container,flush=True)

if __name__=='__main__': main()
