"""Recheck frozen fixtures under protocol-18 wide-geometry compatibility."""
import sys,os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path[:0]=['/workspace/learning/A-finish-20261106-v1/staged','/workspace/src','/workspace']
from preflight_finish_study import geometry,vessel,object_hits,intersections
from robot_pour import build_path,interpolate,ROOT as ROBOT
from pour_finish_study import ROOT,plan,sha
from pxr import UsdGeom,UsdPhysics
from scipy.spatial import ConvexHull
import numpy as np,json
stage,planner,meshes=geometry();cache=UsdGeom.XformCache()
inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT+'/right_arm/arm_r_link7'))).T)
for prim in stage.Traverse():
    name=str(prim.GetPath())
    if '/right_arm/gripper_r_' not in name or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
    points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get());m=inv@np.array(cache.GetLocalToWorldTransform(prim)).T
    local=points@m[:3,:3].T+m[:3,3];hull=ConvexHull(local);meshes.append((6,name,local[hull.vertices],hull.equations))
rows=[]
for spec in plan()['development']+plan()['test']:
    approach,_,_=build_path(stage,spec);approach=np.radians(approach)
    cup=np.array(spec['cup_world']);bottle=cup+[-spec['source_xy'][1],spec['source_xy'][0],0.]
    initial={kind:object_hits(planner,meshes,approach[0],vessel(center,kind)) for kind,center in [('cup',cup),('bottle',bottle)]}
    bench=[];cups=[]
    for u in np.linspace(0,1,41):
        q=interpolate(approach,u)
        if intersections(planner,meshes,q):bench.append(float(u))
        if object_hits(planner,meshes,q,vessel(cup,'cup')):cups.append(float(u))
    r=dict(name=spec['name'],initial=initial,bench=bench,cup=cups,ready=not any(initial.values()) and not bench and not cups);rows.append(r);print(json.dumps(r),flush=True)
    (ROOT/'protocol18-clearance.json').write_text(json.dumps(dict(source_sha256=sha(ROOT/'staged/robot_pour.py'),cases=rows),indent=2)+'\n')
assert len(rows)==38 and all(r['ready'] for r in rows)
