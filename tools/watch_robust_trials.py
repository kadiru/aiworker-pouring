"""Compact read-only watcher for the entire thirty-case frozen batch."""
import hashlib,json,time,urllib.request
from pathlib import Path

root=Path('/home/kadir/aiworker-pouring-4070');study=root/'learning/A-robust-20261103-v1'
selection=json.loads((study/'final-selection.json').read_text());seen=set();passed=0
for _ in range(1080):
    assert hashlib.sha256((study/'best.pt').read_bytes()).hexdigest()==selection['selected_checkpoint_sha256']
    for n,h in selection['runtime_sources'].items():assert hashlib.sha256((root/'src'/n).read_bytes()).hexdigest()==h,n
    try:
        s=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=5));m=s.get('metrics',{})
        for p in sorted((root/'outputs/fountain').glob('*/*T30_A_*/metrics.json')):
            spec=json.loads((p.parent/'parameters.json').read_text());name=spec['name']
            if name in seen:continue
            r=json.loads(p.read_text());f=r['bottle_result'];seen.add(name);passed+=bool(r['surface_task_success_zero_spill'])
            print('RESULT '+json.dumps(dict(name=name,success=r['surface_task_success_zero_spill'],legacy=r['task_success_zero_spill'],
                grasp=r['grasp_verified'],done=r['policy']['done'],height=f['cup_fill_fraction'],source=f['bottle_ml'],spill=r['spill']['final_outside_ml'],gates=r['gates'])),flush=True)
        print(json.dumps(dict(completed=len(seen),passed=passed,name=(s.get('trial') or {}).get('name'),t=s.get('sim_time'),
            phase=s['phase'],spill=m.get('spilled_ml'),source=m.get('bottle_ml'),queue=len(s['queue']),error=s.get('error'),learning_error=s.get('learning_error'))),flush=True)
        if len(seen)==30:print('ALL_30_COMPLETE',flush=True);break
    except OSError as e:print('READ_ONLY_CONNECTION_CHECK '+str(e),flush=True)
    time.sleep(40)
else:raise RuntimeError('Watcher time limit reached; inspect existing results')
