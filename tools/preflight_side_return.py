"""CPU-only side-return IK, joint-step and full gripper/bench/cup screen."""
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--limit',type=int,default=0);a.add_argument('--kinematics-only',action='store_true');a=a.parse_args()
assert not a.output.exists()
ROOT=Path('/workspace/learning/A-side-return-20261108-v1')
sys.path[:0]=[str(ROOT/'staged'),'/workspace/src','/workspace/learning/A-finish-20261106-v1/staged','/workspace']
from check_workspace_clearance import geometry,intersections
from preflight_finish_study import vessel,object_hits
from wide_gripper_geometry import finger_meshes
from robot_pour import build_path
from pour_wide_motion import project_target
from pour_side_return import paths
from scipy.spatial import ConvexHull
import numpy as np
import torch
torch.set_num_threads(1)
stage,_,base_meshes=geometry()
# Conservative full finger-motion envelopes, rather than just nominal closure.
all_fingers=[finger_meshes(stage,angles) for angles in ((.64,.75,.64,.75),(.75,.95,.75,.95),(.64,.95,.64,.95),(.75,.75,.75,.75))]
closed=list(base_meshes)
for j in range(4):
    points=np.concatenate([f[j][2] for f in all_fingers]);h=ConvexHull(points)
    closed.append((6,all_fingers[0][j][1],points[h.vertices],h.equations))
opened=base_meshes+finger_meshes(stage,[0,0,0,0])
plan=json.loads((ROOT/'plan.json').read_text());specs=plan['development']+plan['test'];rows=[]
if a.limit:specs=specs[:a.limit]
report=dict(kind='CPU sampled geometric screen; no fluid, drive dynamics or self-collision proof',cases=rows,complete=False,
    sources={n:hashlib.sha256((ROOT/'staged'/n).read_bytes()).hexdigest() for n in ('pour_side_return.py',)})
for spec in specs:
    approach,lift,tilt,arm=build_path(stage,spec,include_planner=True)
    cup=np.array(spec['cup_world']);source=cup+[-spec['source_xy'][1],spec['source_xy'][0],0.]
    row=dict(name=spec['name'],returns=[],passed=True)
    for phase in (60.,80.,131.):
        r=dict(phase_deg=phase,passed=False)
        try:
            start=project_target(arm,np.radians(tilt[min(round(phase/3),len(tilt)-1)]),source[:2],cup[:2],cup[2],phase,spec['mouth_z'],late_mouth_shift_m=.008)
            times,path,place,retreat,info=paths(arm,start,np.radians(approach[0]),cup[:2],cup[2],{})
            r.update(info,poses=len(path)+len(place)+len(retreat),bench=[],cup=[],placed_bottle=[])
            if not a.kinematics_only:
                for label,qs,hulls in (('uprighting',path,closed),('placing',place,closed),('retreating',retreat,opened)):
                    for i,q in enumerate(qs):
                        hits=intersections(arm,hulls,q)
                        if hits:r['bench'].append(dict(stage=label,i=i,hits=hits))
                        hits=object_hits(arm,hulls,q,vessel(cup,'cup'))
                        if hits:r['cup'].append(dict(stage=label,i=i,hits=hits))
                        if label=='retreating' and i>0:
                            hits=object_hits(arm,hulls,q,vessel(np.r_[info['placement_xy'],cup[2]],'bottle',margin=0.))
                            if hits:r['placed_bottle'].append(dict(i=i,hits=hits))
            r['passed']=not(r['bench'] or r['cup'] or r['placed_bottle'])
        except ValueError as e:r['error']=str(e)
        row['returns'].append(r);row['passed'] &= r['passed']
    rows.append(row);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(name=row['name'],passed=row['passed'],returns=[{k:r.get(k) for k in ('phase_deg','side','max_planned_joint_step_rad','passed','error')} for r in row['returns']])),flush=True)
report.update(complete=True,passed=all(r['passed'] for r in rows),cuda_initialized=torch.cuda.is_initialized())
a.output.write_text(json.dumps(report,indent=2)+'\n');assert not report['cuda_initialized']
