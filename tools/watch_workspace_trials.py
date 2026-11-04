"""Read-only native watcher, checking model and controller identity every40s."""
import argparse,datetime,hashlib,json,subprocess,sys,time,urllib.request
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--part',choices=('demonstrations','repairs','development','test'),required=True);p.add_argument('--round',type=int,default=0)
p.add_argument('--record-test-snapshots',action='store_true',help='Save read-only identity/resource observations at 5, 15, 25 and 30 closed tests')
args=p.parse_args()
root=Path('/home/kadir/aiworker-pouring-4070');study=root/'learning/A-workspace-20261104-v1'
reference=args.part in ('demonstrations','repairs')
plan=json.loads((study/('reference-repair-plan.json' if args.part=='repairs' else 'frozen-plan.json')).read_text())
metadata=json.loads((study/('final-selection.json' if args.part=='test' else 'reference-repair-selection.json' if args.part=='repairs' else 'collection-selection.json' if args.part=='demonstrations' else 'model-summary.json')).read_text())
sha=metadata.get('selected_checkpoint_sha256',metadata.get('checkpoint_sha256'));sources=metadata.get('runtime_sources',metadata.get('sources'))
names={x['name']+(f'_r{args.round}' if args.part=='development' else '') for x in plan['demonstrations' if reference else args.part]};seen=set();passed=0
snapdir=None
if args.record_test_snapshots:
    assert args.part=='test' and sha and len(names)==30
    snapdir=root/'study-snapshots'/('W30-'+sha[:16]);snapdir.mkdir(parents=True,exist_ok=True)

def snapshot(label):
    if snapdir is None:return
    path=snapdir/(label+'.json')
    if path.exists():
        old=json.loads(path.read_text());assert old['selected_checkpoint_sha256']==sha and old['sources']==sources
        return
    record=json.loads(subprocess.check_output([sys.executable,str(root/'snapshot_workspace_runtime.py')],timeout=30))
    assert record['selected_checkpoint_sha256']==sha and record['sources']==sources and record['guard_processes']
    record['completed_test_cases_at_observation']=sorted(seen)
    with path.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    print('SAVED_RESOURCE_SNAPSHOT '+str(path),flush=True)

for _ in range(1080):
    if sha:assert hashlib.sha256((study/'best.pt').read_bytes()).hexdigest()==sha
    for n,h in sources.items():assert hashlib.sha256((root/'src'/n).read_bytes()).hexdigest()==h,n
    try:
        for f in sorted((root/'outputs/fountain').glob('*/*W30_A_*/metrics.json')):
            spec=json.loads((f.parent/'parameters.json').read_text());name=spec['name']
            if name not in names or name in seen:continue
            m=json.loads(f.read_text())
            if sha:assert m['policy']['checkpoint_sha256']==sha
            seen.add(name);success=bool(m['grasp_verified'] and m['zero_spill'] and not m['gates']) if reference else m['surface_task_success_zero_spill'];passed+=success;b=m['bottle_result']
            print('RESULT '+json.dumps(dict(name=name,success=success,reference_only=reference,legacy=m['task_success_zero_spill'],
                grasp=m['grasp_verified'],done=m.get('policy',{}).get('done'),height=b['cup_fill_fraction'],source=b['bottle_ml'],
                reason=b['stop_reason'],spill=m['spill']['final_outside_ml'],
                ever_near_table_ml=m['spill']['ever_table_ml'],gates=m['gates'])),flush=True)
        s=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=5));m=s.get('metrics',{})
        print(json.dumps(dict(completed=len(seen),planned=len(names),passed=passed,name=(s.get('trial') or {}).get('name'),t=s.get('sim_time'),
            phase=s['phase'],spill=m.get('spilled_ml'),ever_near_table_ml=m.get('ever_table_ml'),
            source=m.get('bottle_ml'),height=m.get('cup_fill_fraction'),queue=len(s['queue']),error=s.get('error'),learning_error=s.get('learning_error'))),flush=True)
        if s.get('error') or s.get('learning_error'):raise RuntimeError('Native error; preserve partials and inspect')
        if snapdir is not None:
            for milestone in (5,15,25):
                if len(seen)>=milestone:snapshot(f'runtime-after-test{milestone:02}')
        if seen==names:
            if snapdir is not None:
                assert not s['queue'] and not (s.get('recording') or {}).get('active')
                snapshot('runtime-final')
                completion=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    completed=len(seen),passed=passed,names=sorted(seen),checkpoint_sha256=sha,
                    source_count=len(sources),all_cases_complete=True)
                q=snapdir/'watcher-completion.json'
                if not q.exists():
                    with q.open('x') as f:f.write(json.dumps(completion,indent=2)+'\n')
            print('ALL_CASES_COMPLETE',flush=True);break
    except OSError as e:print('READ_ONLY_CONNECTION_CHECK '+str(e),flush=True)
    time.sleep(40)
else:raise RuntimeError('Watcher time limit reached; preserve and inspect results')
