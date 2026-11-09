"""Trace lost particles in closed records; read-only scoring, never fitting."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path[:0]=['/workspace/src','/workspace']
import numpy as np
from scipy.spatial.transform import Rotation
from bottle_pour import BOTTLE,CUP,vessel_mask
from robot_pour import FRAME
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--part',default='diagnostic');a.add_argument('--name');a.add_argument('--output',type=Path,required=True);a=a.parse_args();assert not a.output.exists()
expected={s['name'] for s in read(ROOT/'plan.json')[a.part]};rows=[]
if a.name:
    assert a.name in expected;expected={a.name}
if a.part=='test':
    assert a.name,'Reserved inspection requires one explicitly closed case'
    frozen=read(ROOT/'final-selection.json')
    assert sha(ROOT/'plan.json')==frozen['plan_sha256'] and sha(ROOT/'best.pt')==frozen['checkpoint_sha256']
    assert all(sha(Path('/workspace/src')/n)==d for n,d in frozen['runtime_sources'].items())
for p in sorted(Path('/workspace/outputs/fountain').glob('*/*SR_A_*/metrics.json')):
    spec=read(p.parent/'parameters.json')
    if spec['name'] not in expected:continue
    samples=read(p.parent/'samples.json');decisions=read(p.parent/'policy-decisions.json')
    npz=p.parent/'trajectory.npz';data=np.load(npz);times=data['times'];cup=np.array(spec['cup_world'])
    final=data[f'positions_{len(times)-1:03}'];lost=set();last_source=None;last_cup=None;history=[]
    for i,t in enumerate(times):
        s=min(samples,key=lambda s:abs(s['t']-t));assert abs(s['t']-t)<1e-6
        points=data[f'positions_{i:03}'];v=data[f'velocities_{i:03}']
        rot=Rotation.from_quat(s['bottle_quat_xyzw']).as_matrix();base=np.array(s['bottle_pose_xyz'])
        local=(points-cup)@FRAME;source=vessel_mask((points-base)@rot,BOTTLE);inside=vessel_mask(local,CUP)&~source
        near=local[:,2]<.0035;lost.update(np.flatnonzero(near).tolist())
        history.append((t,local,v@FRAME,source,inside))
    s=samples[-1];rot=Rotation.from_quat(s['bottle_quat_xyzw']).as_matrix()
    outside=~(vessel_mask((final-cup)@FRAME,CUP)|vessel_mask((final-np.array(s['bottle_pose_xyz']))@rot,BOTTLE))
    lost.update(np.flatnonzero(outside).tolist());traces=[]
    for idx in sorted(lost):
        trace=[];firsttable=next((j for j,h in enumerate(history) if h[1][idx,2]<.0035),len(history)-1)
        for j,h in enumerate(history):
            t,pos,vel,source,inside=h
            if max(0,firsttable-30)<=j<=min(len(history)-1,firsttable+3):
                trace.append(dict(t=float(t),xyz_cup_m=pos[idx].tolist(),velocity_cup_m_s=vel[idx].tolist(),
                    state='source' if source[idx] else 'cup' if inside[idx] else 'outside'))
        traces.append(dict(index=idx,trace=trace))
    exit_times={};entries={}
    for j in range(1,len(history)):
        t,pos,vel,source,inside=history[j];before=history[j-1]
        for idx in np.flatnonzero(before[3]&~source):exit_times[int(idx)]=float(t)
        crossing=(before[1][:,2]>=CUP[-1][0])&(pos[:,2]<CUP[-1][0])&~source
        for idx in np.flatnonzero(crossing):
            u=(before[1][idx,2]-CUP[-1][0])/(before[1][idx,2]-pos[idx,2])
            xy=before[1][idx,:2]*(1-u)+pos[idx,:2]*u
            entries[int(idx)]=dict(t=float(t),xy_cup_m=xy.tolist(),radial_m=float(np.linalg.norm(xy)),inside_at_next_sample=bool(inside[idx]))
    latest=sorted(exit_times,key=exit_times.get)[-100:]
    late_drops=[dict(index=i,last_source_exit_s=exit_times[i],rim_crossing=entries.get(i)) for i in latest]
    stop=next((d['t'] for d in decisions if d['stop_latched']),None)
    rows.append(dict(name=spec['name'],recording=str(p.parent),trajectory_sha256=sha(npz),samples_sha256=sha(p.parent/'samples.json'),
        frames=len(times),stop_time=stop,lost_particles=traces,last_100_source_exits=late_drops,final_height=s['cup_fill_fraction'],
        maximum_height=max(x['cup_fill_fraction'] for x in samples),final_near_table_ml=s['ever_table_ml']))
report=dict(cases=rows,complete=len(rows)==len(expected),scope='20Hz observed particle tracks; no contact impulse measurement or causal force attribution',helper_sha256=sha(Path(__file__)))
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([dict(name=r['name'],lost_ids=[t['index'] for t in r['lost_particles']],frames=r['frames']) for r in rows]))
