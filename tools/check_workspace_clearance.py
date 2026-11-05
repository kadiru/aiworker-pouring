"""CPU-only convex collision-mesh/bench intersection for approach paths.

Historical check: geometry() includes eight arm/palm collision meshes but
omits the four articulated finger meshes. It does not certify full-hand
clearance, replace dynamic native tests, or test robot self-collision.
The newer preflight_layout_extra.py explicitly adds all four finger meshes.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
bindings=list(Path('/isaac-sim/extscache').glob('omni.usd.libs-*'))
assert len(bindings)==1
if not os.environ.get('AIWORKER_USD_BOOTSTRAPPED'):
    env=dict(os.environ,AIWORKER_USD_BOOTSTRAPPED='1')
    env['LD_LIBRARY_PATH']=str(bindings[0]/'bin')+':'+env.get('LD_LIBRARY_PATH','')
    os.execve(sys.executable,[sys.executable,*sys.argv],env)
sys.path.extend((str(bindings[0]),'/workspace/src'))
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
from robot_pour import ArmPath,build_path,interpolate,ROOT as ROBOT_ROOT
from pour_workspace_path import clearance_approach
from pour_workspace_study import ROOT,plan


def geometry():
    stage=Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World/Robot','Xform').GetReferences().AddReference('/workspace/assets/FFW_BG2_without_mimic.usd')
    cache=UsdGeom.XformCache()
    joints=[UsdPhysics.RevoluteJoint.Get(stage,ROBOT_ROOT+'/joints/arm_r_joint'+str(i)) for i in range(1,8)]
    lift=UsdPhysics.PrismaticJoint.Get(stage,ROBOT_ROOT+'/joints/lift_joint')
    planner=ArmPath([j.GetLocalPos0Attr().Get() for j in joints],[j.GetAxisAttr().Get() for j in joints],
        [(j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()) for j in joints],lift.GetLocalPos0Attr().Get())
    meshes=[]
    for i,j in enumerate(joints):
        body=stage.GetPrimAtPath(j.GetBody1Rel().GetTargets()[0])
        inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(body)).T)
        for p in Usd.PrimRange(body):
            if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh):
                points=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
                m=inv@np.array(cache.GetLocalToWorldTransform(p)).T
                local=points@m[:3,:3].T+m[:3,3];hull=ConvexHull(local)
                meshes.append((i,str(p.GetPath()),local[hull.vertices],hull.equations))
    return stage,planner,meshes


def intersections(planner,meshes,q):
    xyz=planner.shoulder.copy();rot=np.eye(3);trans=[]
    for delta,axis,angle in zip(planner.offsets,planner.axes,q):
        xyz=xyz+rot@delta;rot=rot@Rotation.from_rotvec(axis*angle).as_matrix()
        trans.append((xyz.copy(),rot.copy()))
    lo=np.array([.305,-.43,1.016]);hi=np.array([.685,.03,1.04]);hits=[]
    for i,name,local,eq in meshes:
        xyz,rot=trans[i];points=local@rot.T+xyz
        if not (np.minimum(points.max(0),hi)>np.maximum(points.min(0),lo)+1e-7).all():continue
        normal=eq[:,:3]@rot.T;rhs=-eq[:,3]+normal@xyz
        result=linprog(np.zeros(3),A_ub=np.vstack((normal,np.eye(3),-np.eye(3))),
            b_ub=np.r_[rhs,hi, -lo],bounds=[(None,None)]*3,method='highs')
        assert result.status in (0,2),result.message
        if result.success:hits.append(name)
    return hits


def main():
    stage,planner,meshes=geometry();rows=[]
    for spec in plan()['demonstrations']:
        old=np.radians(build_path(stage,spec)[0])
        raised=clearance_approach(planner,old[0],old[-1])
        row=dict(name=spec['name'])
        for label,path in (('straight_joint',old),('raised_cartesian',raised)):
            hits=[]
            for u in np.linspace(0,1,81):
                colliding=intersections(planner,meshes,interpolate(path,u))
                if colliding:hits.append(dict(fraction=float(u),meshes=colliding))
            row[label]=dict(sampled_poses=81,intersections=hits)
        rows.append(row)
        print(json.dumps(dict(name=row['name'],old_hits=len(row['straight_joint']['intersections']),raised_hits=len(row['raised_cartesian']['intersections']))),flush=True)
    report=dict(method='Open-gripper configured convex collision hull intersection with the fixed bench, via linear-program feasibility at 81 joint-interpolated poses per approach.',
        scope='CPU geometric evidence; excludes self-collision, dynamics, drive tracking and native contact offsets. Native trials still required.',
        clearance_m=.04,cases=rows,helper_sha256=hashlib.sha256(Path('/workspace/src/pour_workspace_path.py').read_bytes()).hexdigest())
    (ROOT/'approach-clearance-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    assert next(x for x in rows if x['name']=='W30_A_demo_train_08')['straight_joint']['intersections'], 'The known failed approach must remain a positive geometric regression control'
    assert all(not x['raised_cartesian']['intersections'] for x in rows)


if __name__=='__main__':main()
