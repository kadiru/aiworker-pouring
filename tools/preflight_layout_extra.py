"""Pre-test full-hand fixture and approach checks for the two extra layouts."""
import json
from check_workspace_clearance import geometry,intersections
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdPhysics
from robot_pour import build_path,ArmPath,WRIST,interpolate,ROOT as ROBOT
from pour_workspace_path import clearance_approach
from pour_layout_extra import ROOT,make_plan
from pour_layout_variety import plan as original_plan

ROOT.mkdir(parents=True,exist_ok=True);assert not (ROOT/'selection.json').exists()
p=make_plan(original_plan()['test'][0]);stage,planner,meshes=geometry();cache=UsdGeom.XformCache()
inv=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(ROBOT+'/right_arm/arm_r_link7'))).T)
added=[]
for prim in stage.Traverse():
    name=str(prim.GetPath())
    if '/right_arm/gripper_r_' not in name or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
    points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get());m=inv@np.array(cache.GetLocalToWorldTransform(prim)).T
    local=points@m[:3,:3].T+m[:3,3];hull=ConvexHull(local);meshes.append((6,name,local[hull.vertices],hull.equations));added.append(name)
assert len(added)==4 and len(meshes)==12
rows=[]
for spec,layout in zip(p['test'],p['layouts_world_xy']):
    approach,lift,tilt=build_path(stage,spec);home=np.radians(approach[0]);xyz=planner.shoulder.copy();rot=np.eye(3);trans=[]
    for delta,axis,angle in zip(planner.offsets,planner.axes,home):
        xyz=xyz+rot@delta;rot=rot@Rotation.from_rotvec(axis*angle).as_matrix();trans.append((xyz.copy(),rot.copy()))
    hits=[]
    for kind,profile in [('bottle',[(.004,.025),(.082,.025),(.107,.009),(.135,.009)]),('cup',[(.004,.020),(.038,.024)])]:
        center=np.r_[layout[kind],1.04];points=np.array([[r*np.cos(a),r*np.sin(a),z] for z,r in profile for a in np.arange(32)*2*np.pi/32])+center;eqv=ConvexHull(points).equations
        for i,name,local,eq in meshes:
            xyz,rot=trans[i];mp=local@rot.T+xyz
            if not (np.minimum(mp.max(0),points.max(0))>np.maximum(mp.min(0),points.min(0))+1e-7).all():continue
            normal=eq[:,:3]@rot.T;rhs=-eq[:,3]+normal@xyz
            result=linprog(np.zeros(3),A_ub=np.vstack((normal,eqv[:,:3])),b_ub=np.r_[rhs,-eqv[:,3]],bounds=[(None,None)]*3,method='highs')
            assert result.status in (0,2),result.message
            if result.success:hits.append(dict(object=kind,mesh=name))
    arm=ArmPath(planner.offsets,['XYZ'[int(np.argmax(a))] for a in planner.axes],np.degrees(planner.limits.T),planner.shoulder)
    base=np.r_[layout['bottle'],1.04];grasp=np.radians(arm.solve_pose(base+[0,0,.048],WRIST))
    for z in np.linspace(0,.044,9):arm.solve_pose(base+[0,0,.048+z],WRIST)
    path=clearance_approach(arm,home,grasp,.04);bench=[]
    for u in np.linspace(0,1,81):
        hit=intersections(arm,meshes,interpolate(path,u))
        if hit:bench.append(dict(fraction=float(u),meshes=hit))
    row=dict(name=spec['name'],initial_intersections=hits,bench_intersections=bench,
        reference_ik_reachable=True,actor_grasp_lift_reachable=True,checked_mesh_count=len(meshes),ready=not hits and not bench)
    rows.append(row);print(json.dumps(row),flush=True)
(ROOT/'preflight.json').write_text(json.dumps(dict(scope='Pre-test CPU full-hand geometry and IK. Includes all four open finger meshes; native dynamics still required.',cases=rows),indent=2)+'\n')
(ROOT/'proposed-plan.json').write_text(json.dumps(p,indent=2)+'\n')
assert all(r['ready'] for r in rows)
