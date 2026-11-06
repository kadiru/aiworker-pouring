"""Compare old versus raised ready pose, without changing frozen object layouts."""
import sys,os,argparse
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path[:0]=['/workspace/learning/A-wide-20261105-v1/staged','/workspace']
from check_workspace_clearance import geometry, intersections
from preflight_wide_study import vessel, object_hits
from robot_pour import build_path,interpolate,ROOT as ROBOT
from pour_wide_study import ROOT,plan
from pxr import UsdGeom,UsdPhysics
from scipy.spatial import ConvexHull
import numpy as np
import json

parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--report',default='overhead-ready-clearance.json')
args=parser.parse_args()
stage,planner,meshes=geometry();cache=UsdGeom.XformCache()
inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT+'/right_arm/arm_r_link7'))).T)
for prim in stage.Traverse():
    name=str(prim.GetPath())
    if '/right_arm/gripper_r_' not in name or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
    points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
    m=inv@np.array(cache.GetLocalToWorldTransform(prim)).T;local=points@m[:3,:3].T+m[:3,3]
    hull=ConvexHull(local);meshes.append((6,name,local[hull.vertices],hull.equations))
assert len(meshes)==12
rows=[]
for spec in sum((plan()[k] for k in ('demonstrations','development','test')),[])[:args.limit]:
    cup=np.array(spec['cup_world']);bottle=cup+[-spec['source_xy'][1],spec['source_xy'][0],0.]
    shapes={k:vessel(c,k) for k,c in [('cup',cup),('bottle',bottle)]}
    old=np.radians(build_path(stage,dict(spec,home_clearance_m=0.))[0])
    approach=np.radians(build_path(stage,spec)[0]);home=approach[0]
    row=dict(name=spec['name'],old_ready={k:object_hits(planner,meshes,old[0],sh) for k,sh in shapes.items()},
             raised_ready={k:object_hits(planner,meshes,home,sh) for k,sh in shapes.items()},bench=[],cup=[],early_bottle=[])
    for u in np.linspace(0.,1.,41):
        q=interpolate(approach,u)
        if intersections(planner,meshes,q):row['bench'].append(float(u))
        if object_hits(planner,meshes,q,shapes['cup']):row['cup'].append(float(u))
        # Open fingers deliberately surround the bottle at the end; report
        # early approaches separately instead of interpreting grasp contact.
        if u<.85 and object_hits(planner,meshes,q,shapes['bottle']):row['early_bottle'].append(float(u))
    row['ready']=not any(row['raised_ready'].values()) and not row['bench'] and not row['cup'] and not row['early_bottle']
    rows.append(row)
    print(json.dumps(row),flush=True)
    (ROOT/args.report).write_text(json.dumps(dict(meshes=12,home_raise_m=.10,level_transit=True,
        vessel_shell_m=.0024,combined_contact_margin_m=.002,
        scope='CPU geometry, not native contact guarantees. Frozen vessel positions unchanged; all arm and four finger meshes included.',cases=rows),indent=2)+'\n')
assert all(r['ready'] for r in rows)
