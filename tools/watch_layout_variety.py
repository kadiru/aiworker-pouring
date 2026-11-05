"""Read-only Ubuntu watcher for the twelve once-only autonomous layouts."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

p=argparse.ArgumentParser();p.add_argument('--extra',action='store_true');args=p.parse_args()
root=Path('/home/kadir/aiworker-pouring-4070');study=root/('learning/A-layout-variety-extra-20261105-v1' if args.extra else 'learning/A-layout-variety-20261105-v1')
prefix='VX2_A_' if args.extra else 'V12_A_'
selection=json.loads((study/'selection.json').read_text());plan=json.loads((study/'frozen-plan.json').read_text())
names={x['name'] for x in plan['test']};seen=set();passed=0

def snapshot(label):
    record=json.loads(subprocess.check_output([sys.executable,str(root/'snapshot_workspace_runtime.py')],timeout=30))
    assert record['container']==selection['container'] and record['guard_processes']
    assert record['selected_checkpoint_sha256']==selection['selected_checkpoint_sha256']
    assert all(record['sources'][n]==h for n,h in selection['runtime_sources'].items() if n in record['sources'])
    record['completed_layouts']=sorted(seen)
    (study/(label+'.json')).write_text(json.dumps(record,indent=2)+'\n')

snapshot('runtime-start')
for _ in range(360):
    assert hashlib.sha256((root/'learning/A-workspace-20261104-v1/best.pt').read_bytes()).hexdigest()==selection['selected_checkpoint_sha256']
    for name,digest in selection['runtime_sources'].items():
        assert hashlib.sha256((root/'src'/name).read_bytes()).hexdigest()==digest,name
    for path in sorted((root/'outputs/fountain').glob(f'*/*{prefix}*/metrics.json')):
        spec=json.loads((path.parent/'parameters.json').read_text());name=spec['name']
        if name in seen or name not in names:continue
        m=json.loads(path.read_text());assert m['policy']['checkpoint_sha256']==selection['selected_checkpoint_sha256']
        assert not m['policy']['teacher_fallback']
        seen.add(name);passed+=m['surface_task_success_zero_spill'];f=m['bottle_result']
        print('RESULT '+json.dumps(dict(name=name,label=spec['layout_label'],simulator_pass=m['surface_task_success_zero_spill'],
            grasp=m['grasp_verified'],reason=f['stop_reason'],spill=m['spill']['final_outside_ml'],
            ever_near_table_ml=m['spill']['ever_table_ml'],gates=m['gates'],height=f['cup_fill_fraction'],source=f['bottle_ml'])),flush=True)
    h=json.load(urllib.request.urlopen('http://192.168.68.60:8212/health',timeout=10))
    print(json.dumps(dict(completed=len(seen),planned=len(names),simulator_passes=passed,
        name=(h.get('trial') or {}).get('name'),t=h.get('sim_time'),phase=h['phase'],
        queue=len(h['queue']),error=h.get('error'))),flush=True)
    assert not h.get('error') and not h.get('learning_error')
    for n in (1,6):
        label=f'runtime-after-{n:02}'
        if len(seen)>=n and not (study/(label+'.json')).exists():snapshot(label)
    if seen==names:
        assert not h['queue'] and not h.get('recording',{}).get('active')
        snapshot('runtime-final')
        (study/'watcher-completion.json').write_text(json.dumps(dict(all_cases_complete=True,
            completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),names=sorted(seen),
            checkpoint_sha256=selection['selected_checkpoint_sha256'],simulator_passes=passed),indent=2)+'\n')
        print('ALL_CASES_COMPLETE',flush=True);break
    time.sleep(40)
else:raise RuntimeError('Watcher limit reached; inspect preserved recordings')
