"""Bounded in-process-runtime comparison, no second simulator/GPU process.

Run on the Ubuntu host. Requires idle controllers and no active recording.
Synthetic drop tests are not robot grasp/pour demonstrations.
"""
import argparse, json, time, urllib.request
from pathlib import Path
BASE='http://192.168.68.60:8212'
parser=argparse.ArgumentParser()
parser.add_argument('--ids',default='1,2,3,4,5')
parser.add_argument('--seconds',type=float,default=6)
parser.add_argument('--prefix',default='five-liquid')
args=parser.parse_args()
out=Path('/home/kadir/aiworker-pouring-4070/outputs')
def state():
    return json.load(urllib.request.urlopen(BASE+'/health',timeout=5))
def command(c):
    req=urllib.request.Request(BASE+'/control',data=json.dumps({'command':c}).encode(),headers={'Content-Type':'application/json'})
    urllib.request.urlopen(req,timeout=5).close()
def safe(s):
    if s['recording'] or any(s['controllers'].values()): raise RuntimeError('Operator active; validation stopped')
    if not s['ready'] or s.get('error'): raise RuntimeError('Runtime not ready')
def photo(name):
    (out/name).write_bytes(urllib.request.urlopen(BASE+'/snapshot.jpg',timeout=5).read())
s=state();safe(s)
if s['paused']: raise RuntimeError('Resume first')
report={'evidence_kind':'simulation; synthetic standardized drops, not robot pours','duration_sim_s':args.seconds,'trials':[]}
for liquid in map(int,args.ids.split(',')):
    safe(state())
    command('spill_'+str(liquid)); command('spill_view')
    start=time.monotonic(); samples=[]; first=None
    while time.monotonic()-start < 180:
        s=state();safe(s)
        test=s.get('spill_test',{})
        if test.get('id')==liquid and test.get('phase')=='released':
            elapsed=s['sim_time_s']-test['start_sim_time']
            if first is None: first=(time.monotonic(),s['frame'])
            f=s['fluid']['liquids'][liquid-1]
            r=s.get('fluid_render',{}).get('liquids',[{}]*5)[liquid-1]
            samples.append({'sim_s':elapsed,'fluid':f,'render':r})
            if elapsed>=args.seconds: break
        time.sleep(1)
    else: raise RuntimeError('Spill test timed out')
    photo(f'{args.prefix}-{liquid}-spill.jpg')
    trial={'id':liquid,'samples':samples,'final':samples[-1],
           'max_under_footprint':max(v['fluid']['under_table_footprint'] for v in samples),
           'particles_conserved':all(v['fluid']['total']==v['fluid']['particle_count'] for v in samples),
           'action_steps_per_wall_second':(s['frame']-first[1])/(time.monotonic()-first[0])}
    final=trial['final']; f=final['fluid']; r=final['render']
    trial['passed']=(trial['particles_conserved'] and f['suspected_table_crossings']==0
        and f['on_table']>.5*f['total'] and r['mesh_above_table_vertices']>0
        and r['tabletop_particles_near_mesh']>=.99*f['on_table'] and not r['particle_spheres_visible'])
    report['trials'].append(trial)
    (out/(args.prefix+'-validation.json')).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:trial[k] for k in ['id','max_under_footprint','particles_conserved','action_steps_per_wall_second']}),flush=True)
    print(json.dumps(trial['final']),flush=True)
command('reset');command('comparison_view')
report['passed']=all(t['passed'] for t in report['trials'])
(out/(args.prefix+'-validation.json')).write_text(json.dumps(report,indent=2))
print('COMPARISON_COMPLETE',flush=True)
assert report['passed'], 'Inspect particle retention / crossings / surface coverage'
