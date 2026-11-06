"""Observe owned reference collection; queue remaining refs after clean repairs.

Ordinary finite host process, not a scheduled automation. Never signals other
processes, restarts containers, trains models, or queues reserved tests.
"""
import argparse,datetime,json,pathlib,subprocess,time,urllib.request,hashlib

HOST_ROOT=pathlib.Path('/home/kadir/aiworker-pouring-4070')
STUDY=HOST_ROOT/'learning/A-wide-20261105-v1'
CID='0d4953102c0be267af64193e1d5afe5cf0847054f19a1bc8897097dee246dbf1'
GPU='GPU-cde23916-c5fa-f18e-a343-127f732cc2f6'

def health():return json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=15))
def read(path):return json.loads(path.read_text())
def command(name):
    r=urllib.request.Request('http://192.168.68.60:8212/control',data=json.dumps(dict(command=name)).encode(),headers={'Content-Type':'application/json'},method='POST')
    assert urllib.request.urlopen(r,timeout=15).status==202

def isolation():
    i=json.loads(subprocess.check_output(['docker','inspect',CID],text=True,timeout=15))[0]
    assert i['Config']['Labels']['owner']=='aiworker-pouring' and i['Name']=='/aiworker_pouring_4070'
    assert i['State']['Running'],'Owned container stopped; do not automatically restart'
    c=i['HostConfig'];assert c['DeviceRequests'][0]['DeviceIDs']==[GPU]
    assert c['NanoCpus']==4000000000 and c['CpusetCpus']=='16-23' and c['CpuShares']==256
    assert c['Memory']==c['MemorySwap']==17179869184
    guard=subprocess.run(['pgrep','-af','gpu_priority_guard.py '+CID],capture_output=True,text=True)
    assert guard.returncode==0,'Owned priority guard is missing'
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),container=CID,
        limits={k:c[k] for k in ('DeviceRequests','NanoCpus','CpusetCpus','CpuShares','Memory','MemorySwap')},
        gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True,timeout=15).strip(),guard=guard.stdout.strip())

def records():
    rows=[]
    for p in sorted((HOST_ROOT/'outputs/fountain').glob('*/*T30_A_demo_*/metrics.json')):
        s=read(p.parent/'parameters.json')
        if s.get('protocol')!=17:continue
        m=read(p);r=m['bottle_result']
        rows.append(dict(name=s['name'],reference_case=s.get('reference_case',s['name']),path=str(p.parent),
            accepted=bool(m['grasp_verified'] and m['zero_spill'] and not m['gates']),gates=m['gates'],
            grasp=m['grasp_verified'],zero_spill=m['zero_spill'],stop_reason=r['stop_reason'],expected_stop=s['expected_stop'],
            final_height=r['cup_fill_fraction'],source_remaining_ml=r['bottle_ml'],spill=m['spill']))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=pathlib.Path(a.output);assert not output.exists()
    expected={s['name'] for s in read(STUDY/'frozen-plan.json')['demonstrations']}
    repair_plan=read(STUDY/'reference-repair-plan.json')['demonstrations']
    repairs={s['name'] for s in repair_plan};repair_targets={s['reference_case'] for s in repair_plan}
    report=dict(stage='waiting_for_repaired_pilots',snapshots=[isolation()],rows=[],expected_references=32)
    previous=-1;queued=False;last_snapshot=time.monotonic();start=time.monotonic()
    def save():
        temporary=output.with_suffix('.partial.json');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(output)
    try:
        while time.monotonic()-start<8*3600:
            h=health();assert h['run']==a.run,'Unexpected restart; retain work and inspect before continuing'
            assert not h.get('error') and not h.get('learning_error'),h.get('error') or h.get('learning_error')
            rows=records();report.update(rows=rows,health={k:h.get(k) for k in ('run','phase','sim_time','queue','recording','error')})
            if len(rows)!=previous:
                for r in rows:
                    if r['name'] not in report.get('announced',[]):print(json.dumps(r),flush=True)
                report['announced']=[r['name'] for r in rows];previous=len(rows)
            if time.monotonic()-last_snapshot>=120:
                report['snapshots'].append(isolation());last_snapshot=time.monotonic()
            idle=not h['queue'] and not h.get('recording',{}).get('active') and not h['phase'].startswith('Building')
            if idle and not queued:
                repaired={r['name']:r for r in rows if r['name'] in repairs}
                assert set(repaired)==repairs,'Repair queue ended before every named repair attempt completed'
                assert {r['reference_case'] for r in repaired.values() if r['accepted']}==repair_targets,'A repaired target still lacks an accepted demonstration; inspect before collecting more'
                audit_path=STUDY/'centered-return-clearance.json'
                try:audit=read(audit_path) if audit_path.exists() else {'cases':[]}
                except json.JSONDecodeError:audit={'cases':[]}
                if len(audit['cases'])<70:
                    report['stage']='waiting_for_complete_return_geometry_audit';save();time.sleep(20);continue
                assert len(audit['cases'])==70 and all(r['ready'] for r in audit['cases']),'Corrected return geometry is not clear for every planned case'
                for n in ('robot_pour.py','pour_wide_motion.py','pour_workspace_path.py'):
                    assert hashlib.sha256((HOST_ROOT/'src'/n).read_bytes()).hexdigest()==audit['source_hashes'][n],n
                command('wide_collect');queued=True;report['stage']='collecting_remaining_references'
                print('QUEUED remaining frozen reference layouts',flush=True)
            elif idle and queued:
                accepted={r['reference_case'] for r in rows if r['accepted']}
                assert accepted==expected, 'Collection finished with rejected/missing references; preserve and repair explicitly'
                report.update(stage='complete',elapsed_s=time.monotonic()-start)
                report['snapshots'].append(isolation());save();print('REFERENCE COLLECTION COMPLETE',flush=True);return
            save();time.sleep(20)
        raise TimeoutError('Eight-hour finite watcher limit reached; native state preserved')
    except BaseException as e:
        report.update(stage='needs_inspection',error=str(e),elapsed_s=time.monotonic()-start);save();raise

if __name__=='__main__':main()
