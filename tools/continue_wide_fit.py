"""Finite continuation from accepted references to one fit and development.

Never freezes a candidate, queues reserved tests, or changes foreign jobs.
Requires a reviewed deployment manifest and stops on collection/fit errors.
"""
import argparse,hashlib,json,pathlib,subprocess,time,urllib.request
from watch_wide_collection import isolation,health,command,CID,STUDY,HOST_ROOT

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--collection-progress',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--reuse-deployment',action='store_true');p.add_argument('--reuse-recording-audit',action='store_true')
    p.add_argument('--development-round',type=int,default=0)
    a=p.parse_args();manifest=read(pathlib.Path(a.manifest));output=pathlib.Path(a.output);assert not output.exists()
    progress=dict(stage='waiting_for_accepted_references',source_commit=manifest['source_commit']);started=time.monotonic()
    def save():output.write_text(json.dumps(progress,indent=2)+'\n')
    def staged():
        for n,sha in manifest['files'].items():assert digest(STUDY/'staged'/n)==sha,n
    try:
        staged();assert not (STUDY/'best.pt').exists() and not (STUDY/'final-selection.json').exists()
        repair_restart_started=None
        while time.monotonic()-started<8*3600:
            collection=read(pathlib.Path(a.collection_progress))
            assert collection['stage']!='needs_inspection',collection.get('error')
            # The reviewed repair owns its one idle-only simulator restart.
            # Its atomic progress record distinguishes that bounded handoff
            # from an unexpected guard stop, which still fails closed.
            if collection['stage']=='deploying_reviewed_clearance_repair':
                if repair_restart_started is None:repair_restart_started=time.monotonic()
                assert time.monotonic()-repair_restart_started<600,'Repair deployment did not return ready'
                save();time.sleep(15);continue
            try:
                isolation();h=health();assert not h.get('error') and not h.get('learning_error')
            except (AssertionError,OSError,ValueError):
                if read(pathlib.Path(a.collection_progress))['stage']=='deploying_reviewed_clearance_repair':continue
                raise
            if collection['stage']=='complete':break
            save();time.sleep(30)
        else:raise TimeoutError('Finite reference-wait limit reached')
        assert not h['queue'] and not h.get('recording',{}).get('active');staged()
        if a.reuse_deployment:
            receipt=read(STUDY/('deployment-'+manifest['deployment_phase']+'.json'))
            assert receipt['container']==CID
            assert all(receipt['new_sources'][n]==manifest['files'][n] for n in receipt['files'])
        else:
            progress['stage']='deploying_candidate_source';save()
            subprocess.run(['python3','/tmp/deploy_wide_study.py','--expected-container',CID,'--phase',manifest['deployment_phase'],*manifest['files']],check=True,timeout=180)
        for _ in range(60):
            try:
                h=health()
                if h['phase']=='Ready' and not h.get('error'):break
            except (OSError,ValueError):pass
            time.sleep(5)
        else:raise TimeoutError('Owned simulator did not return ready after deployment')
        isolation()
        for n,sha in manifest['files'].items():assert digest(HOST_ROOT/'src'/n)==sha,n
        progress['stage']='checking_native_reference_recordings';save()
        if not a.reuse_recording_audit:
            subprocess.run(['docker','exec',CID,'/isaac-sim/python.sh',
                '/workspace/learning/A-wide-20261105-v1/staged/audit_wide_study.py',
                '--output','prefit-recording-integrity.json'],check=True,timeout=900)
        integrity=read(STUDY/'prefit-recording-integrity.json')
        assert integrity['native_checks_passed'] and not integrity['pending_attempts']
        assert all(r['valid'] for r in integrity['files'])
        if a.reuse_recording_audit:
            for r in integrity['files']+integrity['sidecar_files']:
                path=HOST_ROOT/'outputs/fountain'/r['relative_path']
                assert path.stat().st_size==r['bytes'] and digest(path)==r['sha256'],r['relative_path']
            progress['reused_recording_audit_sha256']=digest(STUDY/'prefit-recording-integrity.json')
        command('wide_train');progress['stage']='fitting_on_4070';save();print('WIDE_FIT_REQUESTED',flush=True)
        for _ in range(180):
            time.sleep(5);h=health();assert not h.get('error') and not h.get('learning_error'),h.get('learning_error')
            if (h.get('learning') or {}).get('stage')=='fit_complete':break
        else:raise TimeoutError('Fit did not complete within fifteen minutes')
        model=read(STUDY/'model-summary.json');assert model['checkpoint_sha256']==digest(STUDY/'best.pt')
        assert model['device']['visible_cuda_devices']==1 and '4070' in model['device']['name']
        for n,sha in model['sources'].items():assert digest(HOST_ROOT/'src'/n)==sha,n
        progress.update(stage='fit_complete',checkpoint_sha256=model['checkpoint_sha256'],training=model['training']);save()
        rounds=[read(p).get('development_round',-1) for p in (HOST_ROOT/'outputs/fountain').glob('*/*XW30_A_dev_*/parameters.json')]
        assert a.development_round==max(rounds,default=-1)+1,'Requested development round is not the next unused round'
        isolation();command('wide_development')
        progress.update(stage='development_queued',development_round=a.development_round,cases=8,reserved_tests_started=False,elapsed_s=time.monotonic()-started);save()
        print(json.dumps(progress),flush=True)
    except BaseException as e:
        progress.update(stage='needs_inspection',error=str(e),elapsed_s=time.monotonic()-started);save();raise

if __name__=='__main__':main()
