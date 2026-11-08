"""Finite host continuation: fit one component after the authorized collection.

Does not apply weights, restart simulation, queue autonomous trials or retry a
failed fit. All compute stays inside the existing guarded 4070 container.
"""
import hashlib,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=w.HOST_ROOT/'learning/A-perception-20261108-v1'
OUT=Path('/tmp/aiworker-perception-cup-fit')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir();manifest=read(ROOT/'candidate.json');plan=read(ROOT/'plan.json');helper=sha(ROOT/'fit_perception_cup.py')
    assert not manifest['trained'] and not (ROOT/'cup-component.json').exists()
    names={s['name'] for s in plan['collect']};deadline=time.monotonic()+3*3600
    while time.monotonic()<deadline:
        witness=w.isolation();h=w.health();assert not h.get('error') and not h.get('learning_error')
        assert sha(ROOT/'plan.json')==manifest['plan_sha256'] and sha(ROOT/'best.pt')==manifest['checkpoint_sha256']
        assert all(sha(w.HOST_ROOT/'src'/n)==d for n,d in manifest['runtime_sources'].items())
        if (Path('/tmp/aiworker-perception-collection/runtime-after.json')).exists() and not h['queue'] and not h['recording']['active']:
            audit=read(ROOT/'recording-audit.json')
            if not audit['started_incomplete'] and names<={r['name'] for r in audit['records']}:break
        time.sleep(20)
    else:raise TimeoutError('Collection did not close; no fitting started')
    assert sha(ROOT/'fit_perception_cup.py')==helper
    own={int(x.strip()) for x in subprocess.check_output(['docker','top',w.CID,'-eo','pid'],text=True).splitlines()[1:]}
    assert all(int(r.split(',')[1]) in own for r in witness['gpu_processes'].splitlines() if r.startswith(w.GPU))
    assert all(sha(Path(p))==d for p,d in manifest['protected_parent_files'].items())
    (OUT/'runtime-before.json').write_text(json.dumps(dict(witness,helper_sha256=helper),indent=2)+'\n')
    print('COLLECTION AUDITED; starting one cup-component fit on idle guarded 4070.',flush=True)
    with (OUT/'fit.log').open('x') as log:
        result=subprocess.run(['docker','exec',w.CID,'nice','-n','10','/isaac-sim/python.sh','/workspace/learning/A-perception-20261108-v1/fit_perception_cup.py'],stdout=log,stderr=subprocess.STDOUT)
    (OUT/'runtime-after.json').write_text(json.dumps(w.isolation(),indent=2)+'\n')
    assert sha(ROOT/'best.pt')==manifest['checkpoint_sha256'],'A component fit must not replace the active candidate'
    report=dict(exit_code=result.returncode,helper_sha256=helper,component=read(ROOT/'cup-component.json') if (ROOT/'cup-component.json').exists() else None)
    (OUT/'completion.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    assert result.returncode==0
if __name__=='__main__':main()
