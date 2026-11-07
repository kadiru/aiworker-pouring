"""Finite calibration audit -> 4070 fit -> development handoff. No reserved tests."""
import datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
from watch_wide_collection import isolation,health,command,CID,HOST_ROOT
STUDY=HOST_ROOT/'learning/A-finish-20261106-v1'

def read(p):return json.loads(p.read_text())
def main():
    output=Path('/tmp/aiworker-finish-calibration-v1');output.mkdir(exist_ok=False)
    expected={s['name']:s for s in read(STUDY/'calibration-plan.json')['cases']};assert len(expected)==51
    record=dict(stage='collecting',expected=len(expected),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshots=[isolation()])
    def save():
        p=output/'progress.partial.json';p.write_text(json.dumps(record,indent=2)+'\n');p.replace(output/'progress.json')
    try:
        start=time.monotonic()
        while time.monotonic()-start<3*3600:
            h=health();assert not h.get('error') and not h.get('learning_error')
            closed=[]
            for p in (HOST_ROOT/'outputs/fountain').glob('*/*FCAL_A_*/metrics.json'):
                s=read(p.parent/'parameters.json');assert s==expected[s['name']]
                m=read(p);closed.append(dict(name=s['name'],path=str(p.parent),height=m['bottle_result']['cup_fill_fraction'],zero_loss=m['zero_spill']))
            assert len({r['name'] for r in closed})==len(closed)
            record.update(completed=len(closed),rows=closed,health={k:h.get(k) for k in ('run','phase','sim_time','recording','error')});save()
            if len(closed)==len(expected):
                assert not h['queue'] and not h['recording']['active'];break
            time.sleep(15)
        else:raise TimeoutError('Calibration collection exceeded three hours')
        isolation();record['stage']='native_calibration_audit';save()
        subprocess.run(['docker','exec',CID,'/isaac-sim/python.sh','/workspace/src/audit_finish_study.py','--output','calibration-integrity.json','--scope','calibration'],check=True,timeout=900)
        audit=read(STUDY/'calibration-integrity.json');assert audit['native_checks_passed'] and audit['valid_recordings']==len(expected)
        subprocess.run(['docker','exec',CID,'/isaac-sim/python.sh','/workspace/learning/A-finish-20261106-v1/staged/check_finish_calibration.py'],check=True,timeout=300)
        assert not (STUDY/'best.pt').exists();command('finish_train');record['stage']='fitting';save()
        for _ in range(180):
            time.sleep(5);h=health();assert not h.get('error') and not h.get('learning_error'),h.get('learning_error')
            if (STUDY/'model-summary.json').exists() and (h.get('learning') or {}).get('stage')=='fit_complete':break
        else:raise TimeoutError('Fit did not complete')
        model=read(STUDY/'model-summary.json');assert hashlib.sha256((STUDY/'best.pt').read_bytes()).hexdigest()==model['checkpoint_sha256']
        for n,digest in model['sources'].items():assert hashlib.sha256((HOST_ROOT/'src'/n).read_bytes()).hexdigest()==digest,n
        assert not list((HOST_ROOT/'outputs/fountain').glob('*/*FN30_A_dev_*/parameters.json'))
        record['snapshots'].append(isolation());command('finish_development')
        subprocess.Popen([sys.executable,'/tmp/watch_finish_trials.py','--part','development','--round','0','--output-dir','/tmp/aiworker-finish-development-r0'],stdout=open(output/'development-watch.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
        record.update(stage='development_queued',checkpoint_sha256=model['checkpoint_sha256']);save();print('FIT_COMPLETE_DEVELOPMENT_QUEUED',flush=True)
    except BaseException as e:
        record.update(stage='needs_inspection',error=str(e));save();raise
if __name__=='__main__':main()
