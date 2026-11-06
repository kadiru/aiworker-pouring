"""Finite, reviewed clearance repair after the current reference queue ends.

Only retries failed scripted supervision with new identities and changed
geometry. Never retries development/reserved tests or changes foreign jobs.
"""
import argparse,datetime,hashlib,json,pathlib,subprocess,time
from watch_wide_collection import isolation,health,command,records,CID,STUDY,HOST_ROOT


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();manifest=read(pathlib.Path(a.manifest));output=pathlib.Path(a.output);assert not output.exists()
    progress=dict(stage='waiting_for_reference_queue',source_commit=manifest['source_commit'],snapshots=[])
    started=time.monotonic()
    def save():
        temporary=output.with_suffix('.partial.json');temporary.write_text(json.dumps(progress,indent=2)+'\n');temporary.replace(output)
    def wait_idle(required):
        while time.monotonic()-started<8*3600:
            h=health();assert not h.get('error') and not h.get('learning_error')
            done={r['name'] for r in records()}
            if required<=done and not h['queue'] and not h.get('recording',{}).get('active') and not h['phase'].startswith('Building'):return h
            save();time.sleep(30)
        raise TimeoutError('Finite reference repair wait expired')
    try:
        progress['snapshots'].append(isolation());save()
        assert not (STUDY/'final-selection.json').exists() and not (STUDY/'best.pt').exists()
        expected={s['name']:s for s in read(STUDY/'frozen-plan.json')['demonstrations']}
        wait_idle(set(expected));rows=records()
        assert {r['reference_case'] for r in rows}==set(expected),'Some original reference targets never completed'
        accepted={r['reference_case'] for r in rows if r['accepted']};missing=set(expected)-accepted
        # This reviewed repair addresses withdrawal drips/contact clearance.
        # Grasp failure, invalid recordings or other gates need separate diagnosis.
        failed=[r for r in rows if r['reference_case'] in missing]
        assert missing and len(missing)<=8,'No bounded reviewed spill-only repair batch'
        assert all(r['grasp'] and not r['gates'] and not r['zero_spill'] for r in failed),failed
        for name,sha in manifest['files'].items():assert digest(STUDY/'staged'/name)==sha,name
        audit=read(STUDY/'centered-return-r3-clearance.json')
        assert len(audit['cases'])==70 and all(r['ready'] for r in audit['cases'])
        for name,sha in manifest['files'].items():assert audit['source_hashes'][name]==sha,name
        bottle=read(STUDY/'bottle-cup-segmented-candidate.json')
        assert bottle['candidate'] and bottle['segmented_bottle'] and len(bottle['cases'])==70 and not any(r['hits'] for r in bottle['cases'])
        repairs=read(STUDY/'reference-repair-plan.json');new=[]
        for name in sorted(missing):
            new.append(dict(expected[name],name=name+'_clearance_r3',reference_case=name,reference_repair_round=3,
                repair_reason='Retain early tilt clearance and extra lip height; center final upright drips. Failed original is retained.'))
        assert not {r['name'] for r in rows}&{r['name'] for r in new}
        repairs['demonstrations'].extend(new)
        progress.update(stage='deploying_reviewed_clearance_repair',failed_references=failed,repair_names=[r['name'] for r in new]);save()
        progress['snapshots'].append(isolation())
        subprocess.run(['python3','/tmp/deploy_wide_study.py','--expected-container',CID,'--phase',manifest['deployment_phase'],*manifest['files']],check=True,timeout=180)
        for _ in range(60):
            try:
                h=health()
                if h['phase']=='Ready' and not h.get('error'):break
            except (OSError,ValueError):pass
            time.sleep(5)
        else:raise TimeoutError('Owned simulator did not return ready')
        for name,sha in manifest['files'].items():assert digest(HOST_ROOT/'src'/name)==sha,name
        code="""import json,pathlib,sys
r=pathlib.Path('/workspace/learning/A-wide-20261105-v1');p=r/'reference-repair-plan.json';old=r/'reference-repair-plan-before-clearance-r3.json'
assert not old.exists();old.write_bytes(p.read_bytes());p.write_text(json.dumps(json.load(sys.stdin),indent=2)+'\\n')
"""
        subprocess.run(['docker','exec','-i',CID,'/isaac-sim/python.sh','-c',code],input=json.dumps(repairs),text=True,check=True,timeout=30)
        isolation();command('wide_repair');progress['stage']='collecting_clearance_repairs';save()
        print(json.dumps(dict(repairs_queued=progress['repair_names'])),flush=True)
        time.sleep(5);wait_idle(set(progress['repair_names']));rows=records();repaired={r['name']:r for r in rows if r['name'] in progress['repair_names']}
        assert set(repaired)==set(progress['repair_names']) and all(r['accepted'] for r in repaired.values()),repaired
        assert {r['reference_case'] for r in rows if r['accepted']}==set(expected)
        progress.update(stage='complete',rows=rows,elapsed_s=time.monotonic()-started,reserved_tests_started=False)
        progress['snapshots'].append(isolation());save();print('REPAIRED_REFERENCE_COLLECTION_COMPLETE',flush=True)
    except BaseException as error:
        progress.update(stage='needs_inspection',error=str(error),elapsed_s=time.monotonic()-started);save();raise


if __name__=='__main__':main()
