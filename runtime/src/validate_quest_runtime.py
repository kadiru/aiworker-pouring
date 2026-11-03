"""Bounded operator check of only the restored AIWorker service."""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/kadir/aiworker-pouring-4070')
BASE = 'http://192.168.68.60:8212'
def health():
    with urllib.request.urlopen(BASE + '/health', timeout=5) as response:
        return json.load(response)
def command(value):
    request = urllib.request.Request(BASE + '/control', data=json.dumps({'command': value}).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 202
def wait_for(predicate):
    for _ in range(60):
        try:
            state = health()
            if predicate(state): return state
        except (OSError, ValueError):
            pass
        time.sleep(0.5)
    raise RuntimeError('AIWorker state did not reach the requested condition')
def output(args):
    return subprocess.check_output(args, text=True, timeout=15).strip()

before = wait_for(lambda s: s.get('ready') and s.get('frame', 0) >= 30)
assert not before['recording'] and not before['paused']
command('pause')
try:
    paused = wait_for(lambda s: s.get('paused'))
    time.sleep(2)
    held = health()
    assert held['sim_time_s'] == paused['sim_time_s']
    assert held['frame'] == paused['frame']
    assert held['joint_positions'] == paused['joint_positions']
    assert held['fluid'] == paused['fluid']
    assert held['preview_updated_unix'] > paused['preview_updated_unix']
finally:
    if health()['paused']: command('pause')
resumed = wait_for(lambda s: not s['paused'] and s['frame'] > held['frame'] + 5)
info = json.loads(output(['docker', 'inspect', 'aiworker_pouring_4070']))[0]
assert info['Config']['Labels']['owner'] == 'aiworker-pouring'
visible = output(['docker', 'exec', info['Id'], 'nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'])
assert visible == 'GPU-cde23916-c5fa-f18e-a343-127f732cc2f6'
old = json.loads(output(['docker', 'inspect', 'isaacsim6_home']))[0]
report = {'passed': True, 'checked_unix': time.time(), 'container_id': info['Id'],
          'visible_gpu': visible, 'limits': {k: info['HostConfig'][k] for k in
          ('NanoCpus', 'CpuShares', 'CpusetCpus', 'Memory', 'MemorySwap', 'ShmSize', 'PidsLimit')},
          'pause': {'time_before': paused['sim_time_s'], 'time_held': held['sim_time_s'],
                    'time_resumed': resumed['sim_time_s'], 'preview_kept_updating': True,
                    'fluid_and_joints_unchanged_while_paused': True},
          'health': resumed, 'old_container': {'running': old['State']['Running'], 'started_at': old['State']['StartedAt']},
          'gpu_processes': output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader']),
          'resources': output(['docker', 'stats', '--no-stream', '--format', '{{json .}}', info['Id']])}
(ROOT / 'outputs/quest-runtime-verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'health'}, indent=2))
