"""Read-only native watcher, checking model and controller identity every40s."""
import argparse,hashlib,json,time,urllib.request
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--part',choices=('demonstrations','repairs','development','test'),required=True);p.add_argument('--round',type=int,default=0);args=p.parse_args()
root=Path('/home/kadir/aiworker-pouring-4070');study=root/'learning/A-workspace-20261104-v1'
reference=args.part in ('demonstrations','repairs')
plan=json.loads((study/('reference-repair-plan.json' if args.part=='repairs' else 'frozen-plan.json')).read_text())
metadata=json.loads((study/('final-selection.json' if args.part=='test' else 'reference-repair-selection.json' if args.part=='repairs' else 'collection-selection.json' if args.part=='demonstrations' else 'model-summary.json')).read_text())
sha=metadata.get('selected_checkpoint_sha256',metadata.get('checkpoint_sha256'));sources=metadata.get('runtime_sources',metadata.get('sources'))
names={x['name']+(f'_r{args.round}' if args.part=='development' else '') for x in plan['demonstrations' if reference else args.part]};seen=set();passed=0
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
                reason=b['stop_reason'],spill=m['spill']['final_outside_ml'],gates=m['gates'])),flush=True)
        s=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=5));m=s.get('metrics',{})
        print(json.dumps(dict(completed=len(seen),planned=len(names),passed=passed,name=(s.get('trial') or {}).get('name'),t=s.get('sim_time'),
            phase=s['phase'],spill=m.get('spilled_ml'),source=m.get('bottle_ml'),height=m.get('cup_fill_fraction'),queue=len(s['queue']),error=s.get('error'),learning_error=s.get('learning_error'))),flush=True)
        if s.get('error') or s.get('learning_error'):raise RuntimeError('Native error; preserve partials and inspect')
        if seen==names:print('ALL_CASES_COMPLETE',flush=True);break
    except OSError as e:print('READ_ONLY_CONNECTION_CHECK '+str(e),flush=True)
    time.sleep(40)
else:raise RuntimeError('Watcher time limit reached; preserve and inspect results')
