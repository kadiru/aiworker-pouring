"""Finite, owned-container-only calibration -> recovery regression coordinator.

Stops on failure, changed runtime, foreign GPU use or deadline. Never restarts
automatically after a priority-guard shutdown. Not an automation/service.
"""
import datetime,hashlib,json,pathlib,subprocess,sys,time,urllib.request
sys.path.insert(0,'/tmp')
from watch_wide_collection import health,isolation,command,CID,GPU,HOST_ROOT
ROOT=HOST_ROOT/'learning/A-size-20261108-v1'
STAGED=pathlib.Path('/tmp/aiworker-size-stage-tools')


def execute(*args,timeout=180):
    return subprocess.run(args,text=True,capture_output=True,check=True,timeout=timeout).stdout.strip()


def owned_idle():
    snapshot=isolation();h=health()
    assert not h['queue'] and not (h.get('recording') or {}).get('active') and not h.get('error') and not h.get('learning_error')
    own={int(x.strip()) for x in execute('docker','top',CID,'-eo','pid').splitlines()[1:]}
    assert all(int(row.split(',')[1]) in own for row in snapshot['gpu_processes'].splitlines() if row.startswith(GPU))
    return snapshot


def count(prefix):
    return list((HOST_ROOT/'outputs/fountain').glob('*/*'+prefix+'*/metrics.json'))


def wait_cases(prefix,n,seconds,run):
    deadline=time.monotonic()+seconds;previous=-1
    while time.monotonic()<deadline:
        h=health();assert h['run']==run and not h.get('error') and not h.get('learning_error'),h
        files=count(prefix)
        if len(files)!=previous:
            snapshot=isolation();previous=len(files)
            print(json.dumps(dict(stage=prefix,completed=previous,expected=n,utc=snapshot['utc'])),flush=True)
        if len(files)==n:
            assert not h['queue'] and not (h.get('recording') or {}).get('active')
            return
        if previous and not h['queue'] and not (h.get('recording') or {}).get('active'):
            raise RuntimeError('Queue ended before declared cases completed')
        time.sleep(20)
    raise TimeoutError('Finite study deadline reached; inspect without automatic reruns')


def main():
    assert not (STAGED/'coordinator-started.json').exists()
    (STAGED/'coordinator-started.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),container=CID)))
    first=health()['run'];wait_cases('SZ_A_cup_static_',9,45*60,first)
    snapshot=owned_idle();print(execute('docker','exec',CID,'/isaac-sim/python.sh','/workspace/learning/A-size-20261108-v1/check_size_stage.py'),flush=True)
    report=json.loads((ROOT/'stage-results.json').read_text())
    assert report['all_calibration_contained'],'Retain failed containment pilots; no autonomous regression queue'
    (STAGED/'runtime-after-calibration.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    for name in ('pour_source_watchdog.py','test_pour_size.py'):
        execute('docker','cp',str(STAGED/name),CID+':/workspace/learning/A-size-20261108-v1/staged/'+name)
    print(execute('docker','exec','-e','PYTHONPATH=/workspace/learning/A-size-20261108-v1/staged','-e','OMP_NUM_THREADS=4',CID,
        '/isaac-sim/python.sh','-m','unittest','test_pour_size','test_pour_guard','test_pour_finish','test_pour_source_alignment'),flush=True)
    code="""
import copy,hashlib,json,shutil,sys
from pathlib import Path
import torch
r=Path('/workspace/learning/A-size-20261108-v1');p=Path('/workspace/learning/A-finish-20261106-v1')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=json.loads((r/'candidate.json').read_text())
assert sha(r/'best.pt')==m['checkpoint_sha256']
assert all(sha(p/n)==v for n,v in m['protected_parent_files'].items())
assert all(sha(Path('/workspace/src')/n)==v for n,v in m['runtime_sources'].items())
assert not list(Path('/workspace/outputs/fountain').glob('*/*SZ_A_recovery*/parameters.json'))
archive=r/'calibration-candidate-r0';archive.mkdir()
for n in ('candidate.json','best.pt','plan.json','stage-results.json'):shutil.copy2(r/n,archive/n)
shutil.copy2(Path('/workspace/src/pour_source_watchdog.py'),archive/'pour_source_watchdog.py')
sys.path.insert(0,str(r/'staged'))
from pour_source_watchdog import DEFAULT
saved=torch.load(r/'best.pt',map_location='cpu',weights_only=False)
saved['source_watchdog']=DEFAULT;saved['model_version']='visual_size_stage_v25_recovery_only_r1'
m['runtime_sources']['pour_source_watchdog.py']=sha(r/'staged/pour_source_watchdog.py')
saved['sources']=copy.deepcopy(m['runtime_sources']);torch.save(saved,r/'best.pt')
m.update(checkpoint_sha256=sha(r/'best.pt'),source_watchdog=DEFAULT,
    timeout_selection='All 30 former V24 test traces explicitly consumed as development; 40s would interrupt two old successes, 48s interrupts none before their original stop. No native recovery attempts existed during selection.',
    previous_candidate_sha256=m['checkpoint_sha256'])
(r/'candidate.json').write_text(json.dumps(m,indent=2)+'\\n')
print(json.dumps(dict(checkpoint_sha256=m['checkpoint_sha256'],source_watchdog=DEFAULT)))
"""
    print(execute('docker','exec',CID,'/isaac-sim/python.sh','-c',code),flush=True)
    print(execute('python3',str(STAGED/'deploy_finish_study.py'),'--expected-container',CID,'--study-name','A-size-20261108-v1',
        '--phase','recovery-r1','pour_source_watchdog.py'),flush=True)
    deadline=time.monotonic()+300;new_run=None
    while time.monotonic()<deadline:
        isolation()  # A stopped container is a hard stop, never an auto-restart.
        try:h=health()
        except (OSError,ValueError):time.sleep(5);continue
        assert not h.get('error') and not h.get('learning_error')
        if h['run']!=first and h['phase']=='Ready':new_run=h['run'];break
        time.sleep(5)
    assert new_run,'Restart did not reach readiness'
    (STAGED/'runtime-before-recovery.json').write_text(json.dumps(owned_idle(),indent=2)+'\n')
    command('size_recovery_development');print('Queued six consumed-layout recovery regressions',flush=True)
    wait_cases('SZ_A_recovery_',6,150*60,new_run)
    (STAGED/'runtime-final.json').write_text(json.dumps(owned_idle(),indent=2)+'\n')
    print(execute('docker','exec',CID,'/isaac-sim/python.sh','/workspace/learning/A-size-20261108-v1/check_size_stage.py'),flush=True)
    print('Size-stage validation closed; no further jobs queued',flush=True)


if __name__=='__main__':main()
