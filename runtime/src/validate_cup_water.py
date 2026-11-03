"""Bounded idle-scene checks; resets only this AIWorker simulation."""
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

def idle(state):
    assert not state['recording'] and not any(state['controllers'].values()), 'Operator is active'

def wait_for(predicate, timeout=55):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = health()
        idle(state)
        if predicate(state):
            return state
        time.sleep(.5)
    raise RuntimeError('AIWorker did not reach the requested state')

def command(name):
    idle(health())
    request = urllib.request.Request(BASE + '/control', data=json.dumps({'command':name}).encode(),
                                     headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 202

def check(state):
    fluid, render = state['fluid'], state['fluid_render']
    assert fluid['total'] == fluid['source_cup'] == 1331, fluid
    assert fluid['outside_both'] == fluid['below_table'] == fluid['target_cup'] == 0, fluid
    assert fluid['moved_from_seed_m'] > .005, fluid
    assert render['mesh_vertices'] > 100 and render['mesh_faces'] > 100, render
    assert render['mesh_visible'] and not render['particle_spheres_visible'], render
    assert render['material'] == '/World/PourWaterSurface', render
    assert abs(state['source_cup_pose'][2] - 1.060) < .003, state['source_cup_pose']

before = wait_for(lambda s: s.get('ready') and s['frame'] > 80 and s.get('fluid_render'))
idle(before)
check(before)
resets = []
for index in range(3):
    reset_count = health()['reset_count']
    command('reset')
    reset = wait_for(lambda s: s['reset_count'] > reset_count)
    settled = wait_for(lambda s: s['frame'] > reset['frame'] + 80)
    check(settled)
    resets.append({k:settled[k] for k in ('reset_count','frame','fluid','fluid_render','source_cup_pose')})
    print(json.dumps({'reset':index+1,'fluid':settled['fluid'],'surface':settled['fluid_render']}), flush=True)
start = health()
time.sleep(5)
end = health()
idle(end)
fps = (end['frame']-start['frame'])/(end['updated_unix']-start['updated_unix'])
command('cup_view')
time.sleep(3)
with urllib.request.urlopen(BASE+'/snapshot.jpg', timeout=5) as response:
    (ROOT/'outputs/cup-water-closeup.jpg').write_bytes(response.read())
info = json.loads(subprocess.check_output(['docker','inspect','aiworker_pouring_4070']))[0]
assert info['Config']['Labels']['owner'] == 'aiworker-pouring'
report = {'passed':True, 'checked_unix':time.time(), 'evidence_kind':'simulation',
          'grasp_verified':False, 'pour_verified':False,
          'container_id':info['Id'], 'cup_geometry':end['cup_geometry'],
          'wet_reset_checks':resets, 'measured_action_steps_per_s':fps,
          'limits':{k:info['HostConfig'][k] for k in ('NanoCpus','CpuShares','CpusetCpus','Memory','MemorySwap','PidsLimit')},
          'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True),
          'resources':subprocess.check_output(['docker','stats','--no-stream','--format','{{json .}}',info['Id']],text=True)}
(ROOT/'outputs/cup-water-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
