"""Read-only native trial watcher; never sends controls or changes a policy."""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=('development', 'test'), required=True)
    p.add_argument('--interval', type=float, default=40)
    p.add_argument('--study',choices=('guard','vote'),default='guard')
    args = p.parse_args()
    root = Path('/home/kadir/aiworker-pouring-4070')
    study='A-guard-20261103-v2' if args.study=='vote' else 'A-guard-20261103-v1'
    prefix='H_A_' if args.study=='vote' else 'G_A_'
    plan=json.loads((root/'learning'/study/'frozen-plan.json').read_text())
    expected = 6 if args.split == 'test' else len(plan['regression'])+len(plan['validation'])
    for _ in range(180):
        assert hashlib.sha256((root / 'learning' / study / 'best.pt').read_bytes()).hexdigest() == args.checkpoint
        s = json.load(urllib.request.urlopen('http://192.168.68.60:8212/health', timeout=5))
        m = s.get('metrics', {})
        print(json.dumps(dict(name=(s.get('trial') or {}).get('name'), phase=s['phase'],
            t=s.get('sim_time'), cup=m.get('cup_fill_fraction'), source=m.get('bottle_ml'),
            spill=m.get('spilled_ml'), verified_grasp=m.get('grasp_verified'),
            reason=m.get('stop_reason'), remaining_queue=s['queue'],
            recording_active=s.get('recording', {}).get('active'),
            error=s.get('error'), learning_error=s.get('learning_error'))), flush=True)
        rows = []
        for name in glob.glob(str(root / f'outputs/fountain/*/*{prefix}*/metrics.json')):
            q = Path(name)
            result = json.loads(q.read_text())
            spec = json.loads((q.parent / 'parameters.json').read_text())
            if result['policy']['checkpoint_sha256'] != args.checkpoint:
                continue
            if (spec['study_split'] == 'test') != (args.split == 'test'):
                continue
            f = result['bottle_result']
            rows.append(dict(name=spec['name'], grasp=result['grasp_verified'],
                done=result['policy']['done'], surface=f['cup_fill_fraction'],
                source=f['bottle_ml'], outside=result['spill']['final_outside_ml'],
                success=result['surface_task_success_zero_spill'],
                legacy_success=result['task_success_zero_spill'], gates=result['gates']))
        print('completed ' + json.dumps(rows), flush=True)
        if len(rows) == expected:
            print('ALL_REQUESTED_CASES_COMPLETE', flush=True)
            return
        time.sleep(args.interval)
    raise RuntimeError('Read-only monitoring limit reached; inspect existing records')


if __name__ == '__main__':
    main()
