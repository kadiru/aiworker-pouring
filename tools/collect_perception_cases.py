"""Finite host observer for the authorized 47-scene training/validation split."""
import datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=Path('/home/kadir/aiworker-pouring-4070/learning/A-perception-20261108-v1')
CID='0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'
OUT=Path('/tmp/aiworker-perception-collection')
def read(p):return json.loads(p.read_text())
def check():
    assert subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',CID],text=True).strip()=='true','Priority stop respected; no auto-restart'
    h=w.health();assert not h.get('error') and not h.get('learning_error'),h.get('error')
    return h
def main():
    resume='--resume' in sys.argv
    OUT.mkdir(exist_ok=resume);plan=read(ROOT/'plan.json');candidate=read(ROOT/'candidate.json')
    plan_sha=hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest();assert plan_sha==candidate['plan_sha256']
    names={s['name'] for s in plan['collect']};assert len(names)==47
    review=read(ROOT/'camera-review.json');assert review['passed'] and review['gripper_pilot_passed']
    snapshot='runtime-resume-'+str(time.time_ns())+'.json' if resume else 'runtime-before.json'
    if resume:assert (OUT/'runtime-before.json').exists()
    (OUT/snapshot).write_text(json.dumps(w.isolation(),indent=2)+'\n')
    deadline=time.monotonic()+10800
    while True:
        assert time.monotonic()<deadline
        h=check();g=read(ROOT/'geometry-preflight.json');passed={r['name'] for r in g['cases'] if r['passed']}
        failed=[r for r in g['cases'] if r['name'] in names and not r['passed']];assert not failed,failed
        if names.issubset(passed) and (resume or (not h['queue'] and not h['recording']['active'])):break
        print(json.dumps(dict(stage='waiting_geometry',passed_collection=len(passed&names),total=len(names),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
        time.sleep(20)
    assert hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest()==plan_sha
    w.isolation()
    if not resume:w.command('perception_collect');time.sleep(2)
    last=-1
    while True:
        assert time.monotonic()<deadline
        h=check();assert hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest()==plan_sha
        closed={read(p.parent/'parameters.json')['name'] for p in Path('/home/kadir/aiworker-pouring-4070/outputs/fountain').glob('*/*PV_A_*/metrics.json')}&names
        row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),closed=len(closed),total=len(names),phase=h['phase'],
            current=(h.get('recording') or {}).get('path'),frames=(h.get('recording') or {}).get('frames'),queue=len(h['queue']))
        (OUT/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
        if len(closed)!=last:print(json.dumps(row),flush=True);last=len(closed)
        if closed==names and not h['queue'] and not h['recording']['active']:break
        if not h['queue'] and not h['recording']['active']:
            started={read(p)['name'] for p in Path('/home/kadir/aiworker-pouring-4070/outputs/fountain').glob('*/*PV_A_*/parameters.json')}&names
            assert started==closed,'Retain and diagnose incomplete attempts; never retry them under the same ID'
            assert names-started
            w.isolation();w.command('perception_collect')
            print('Queued remaining unstarted collection cases after the server\'s 32-case queue cap.',flush=True)
        time.sleep(15)
    (OUT/'runtime-after.json').write_text(json.dumps(w.isolation(),indent=2)+'\n')
    subprocess.run(['docker','exec',CID,'/isaac-sim/python.sh','/workspace/learning/A-perception-20261108-v1/check_perception_stage.py'],check=True)
    print('COLLECTION COMPLETE. Fitting and reserved cases have not been started by this observer.',flush=True)
if __name__=='__main__':main()
