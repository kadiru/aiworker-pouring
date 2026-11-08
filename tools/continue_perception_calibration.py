"""Finite host coordinator: finish fixtures, fit presence, deploy camera pilots."""
import datetime,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/tmp')
import watch_wide_collection as w
ROOT=Path('/home/kadir/aiworker-pouring-4070/learning/A-perception-20261108-v1')
CID='0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'
def run(*args):subprocess.run(args,check=True)
def native(name):run('docker','exec',CID,'/isaac-sim/python.sh','/workspace/learning/A-perception-20261108-v1/'+name)
def snapshot(name):
    code='import json,sys;from pathlib import Path;p=json.load(sys.stdin);Path(p["path"]).write_text(json.dumps(p["value"],indent=2)+"\\n")'
    subprocess.run(['docker','exec','-i',CID,'/isaac-sim/python.sh','-c',code],
        input=json.dumps(dict(path='/workspace/learning/A-perception-20261108-v1/'+name,value=w.isolation())),text=True,check=True)
def main():
    deadline=time.monotonic()+1800;previous=None
    while True:
        assert time.monotonic()<deadline,'Calibration wait deadline'
        assert subprocess.check_output(['docker','inspect','-f','{{.State.Running}}',CID],text=True).strip()=='true','Owned container stopped; do not restart after priority guard'
        h=w.health();assert not h.get('error') and not h.get('learning_error')
        key=(h.get('recording') or {}).get('path')
        if key!=previous:print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=h['phase'],recording=key,queue=h['queue'])),flush=True);previous=key
        if not h['queue'] and not (h.get('recording') or {}).get('active'):break
        time.sleep(10)
    snapshot('runtime-before-presence-fit.json')
    native('check_perception_stage.py')
    audit=json.loads((ROOT/'recording-audit.json').read_text());assert len(audit['records'])==10 and not audit['started_incomplete']
    native('fit_perception_presence.py')
    snapshot('runtime-after-presence-fit.json')
    native('prepare_perception_collection.py')
    run(sys.executable,str(ROOT/'deploy_finish_study.py'),'--expected-container',CID,'--study-name','A-perception-20261108-v1',
        '--phase','wide-camera-r2','pour_recording.py','validate_pour_recording.py','run_fountain.py','pour_perception_study.py')
    for _ in range(90):
        time.sleep(2)
        try:h=w.health()
        except Exception:continue
        assert not h.get('error')
        if h.get('phase')=='Ready':break
    else:raise RuntimeError('New simulator did not become Ready')
    w.isolation();w.command('perception_camera_pilot')
    print('QUEUED three wide-camera pilots; coordinator complete. Bulk collection requires native view review.',flush=True)
if __name__=='__main__':main()
