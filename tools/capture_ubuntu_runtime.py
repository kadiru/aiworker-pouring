"""Read-only Ubuntu snapshot. Run over SSH; redirect stdout to local evidence."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path('/home/kadir/aiworker-pouring-4070')
GPU = 'GPU-cde23916-c5fa-f18e-a343-127f732cc2f6'


def run(*args):
    return subprocess.check_output(args, text=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',choices=('guard','vote','robust'),default='guard');args=p.parse_args()
    study={'guard':'A-guard-20261103-v1','vote':'A-guard-20261103-v2','robust':'A-robust-20261103-v1'}[args.study]
    c = json.loads(run('docker', 'inspect', 'aiworker_pouring_4070'))[0]
    cid = c['Id']
    assert c['Config']['Labels']['owner'] == 'aiworker-pouring'
    limits = {k: c['HostConfig'][k] for k in (
        'NanoCpus', 'CpuShares', 'CpusetCpus', 'Memory', 'MemorySwap', 'DeviceRequests')}
    assert limits['NanoCpus'] == 4000000000 and limits['CpuShares'] == 256
    assert limits['CpusetCpus'] == '16-23'
    assert limits['Memory'] == limits['MemorySwap'] == 17179869184
    assert limits['DeviceRequests'][0]['DeviceIDs'] == [GPU]
    s = json.load(urllib.request.urlopen('http://192.168.68.60:8212/health', timeout=5))
    sources = ('pour_guard_actor.py', 'pour_guard_train.py', 'pour_guard_models.py',
        'pour_guard_vision.py', 'pour_guard_trajectory.py', 'pour_guard_study.py',
        'pour_guard_data.py', 'pour_stop_actor.py', 'pour_stop_models.py',
        'pour_stop_sequence.py', 'run_fountain.py', 'robot_pour.py', 'pour_recording.py',
        'bottle_pour.py', 'pour_fluid.py', 'pour_geometry.py', 'vessel_appearance.py')
    if args.study in ('vote','robust'):sources+=('pour_guard_vote_study.py',)
    if args.study=='robust':sources+=('pour_robust_study.py',)
    result = dict(
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        container=cid, owner='aiworker-pouring', limits=limits,
        guard_processes=run('pgrep', '-af', 'gpu_priority_guard.py ' + cid).splitlines(),
        owned_processes=run('docker', 'top', cid, '-eo', 'pid,ni,args'),
        gpu_compute_processes=run('nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'),
        gpu_utilization=run('nvidia-smi', '--query-gpu=name,uuid,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'),
        memory=run('free', '-m'),
        selected_checkpoint_sha256=hashlib.sha256((ROOT / 'learning' / study / 'best.pt').read_bytes()).hexdigest(),
        sources={n: hashlib.sha256((ROOT / 'src' / n).read_bytes()).hexdigest() for n in sources},
        asset_sha256=hashlib.sha256((ROOT / 'assets/FFW_BG2_without_mimic.usd').read_bytes()).hexdigest(),
        health={k: s.get(k) for k in ('run', 'phase', 'sim_time', 'queue', 'recording', 'error', 'learning_error', 'learning')})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
