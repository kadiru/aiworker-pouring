"""CPU check of the over-cup return and fixed vessel/bench geometry."""
import os,sys
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path[:0]=['/workspace/learning/A-finish-20261106-v1/staged','/workspace']
from check_workspace_clearance import geometry,intersections
from preflight_finish_study import vessel,object_hits
from robot_pour import build_path,interpolate,ROOT as ROBOT
from pour_wide_motion import centered_return_path
from pour_finish_study import ROOT,plan
from pxr import UsdGeom,UsdPhysics
from scipy.spatial import ConvexHull
from wide_gripper_geometry import finger_meshes,REST_ERRORS
import argparse,json,numpy as np,itertools,hashlib
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--name');p.add_argument('--output',default='centered-return-clearance.json');p.add_argument('--open-hand',action='store_true');a=p.parse_args()
source_hashes={n:hashlib.sha256(next(p for p in (ROOT/'staged'/n,Path('/workspace/src')/n,Path('/workspace')/n) if p.exists()).read_bytes()).hexdigest() for n in ('robot_pour.py','pour_wide_motion.py','pour_wide_geometry.py','pour_workspace_path.py','wide_gripper_geometry.py','audit_finish_return.py')}
stage,planner,meshes=geometry();cache=UsdGeom.XformCache()
inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT+'/right_arm/arm_r_link7'))).T)
for prim in stage.Traverse():
    name=str(prim.GetPath())
    if '/right_arm/gripper_r_' not in name or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
    points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get());m=inv@np.array(cache.GetLocalToWorldTransform(prim)).T
    local=points@m[:3,:3].T+m[:3,3];hull=ConvexHull(local);meshes.append((6,name,local[hull.vertices],hull.equations))
assert len(meshes)==12
if not a.open_hand:
    meshes=meshes[:8]
    # Conservative independent ranges contain the observed held-finger values
    # (driven ~0.66–0.73, passive ~0.79–0.92 rad) with extra margin.
    for angles in itertools.product((.64,.75),(.75,.95),(.64,.75),(.75,.95)):
        meshes.extend(finger_meshes(stage,angles))
rows=[]
for spec in sum((plan()[k] for k in ('demonstrations','development','test')),[]):
    if a.name and spec['name']!=a.name:continue
    _,lift,tilt,arm=build_path(stage,spec,include_planner=True)
    cup=np.array(spec['cup_world']);source=cup+[-spec['source_xy'][1],spec['source_xy'][0],0.]
    # A 165-degree start covers the full tilt range, including all full stops.
    times,path=centered_return_path(arm,np.radians(tilt[-1]),165.,source[:2],cup[:2],cup[2],np.radians(lift[-1]),spec['mouth_z'],drain_hold_s=0.)
    row=dict(name=spec['name'],bench=[],cup=[],poses=len(path),duration=float(times[-1]))
    for t,q in zip(times,path):
        hits=intersections(arm,meshes,q)
        if hits:row['bench'].append(dict(t=float(t),hits=hits))
        hits=object_hits(arm,meshes,q,vessel(cup,'cup'))
        if hits:row['cup'].append(dict(t=float(t),hits=hits))
    row['ready']=not row['bench'] and not row['cup'];rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/a.output).write_text(json.dumps(dict(source_hashes=source_hashes,hulls=len(meshes),closed_grip_ranges_rad=None if a.open_hand else [[.64,.75],[.75,.95],[.64,.75],[.75,.95]],rest_transform_discrepancies=REST_ERRORS,cases=rows,
        scope='USD-joint-articulated finger hulls along centered return/transfer, including all 16 range corners for closed grasps. Authored rest discrepancies explicitly reported; joint frames govern FK. Excludes dynamics/self-collision/particle validation.'),indent=2)+'\n')
assert rows and all(r['ready'] for r in rows)
